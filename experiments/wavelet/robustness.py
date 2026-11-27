"""Run controlled wavelet ablation, family comparison and paired bootstrap."""
from __future__ import annotations
import json, tempfile, sys
from datetime import datetime, timezone
from pathlib import Path
import boto3, lightgbm as lgb, numpy as np, pandas as pd, polars as pl
from botocore.config import Config

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.train_model import RAW, env, load_features_from_frame, metric_pack
from experiments.wavelet.features import build_wavelet_features

OUT = "wavelet-experiment-artifacts"
FAMILIES = ("db2", "db4", "sym4", "coif1")
WCOLS = [
    "wavelet_approx_energy", "wavelet_detail_energy_l1",
    "wavelet_detail_energy_l2", "wavelet_coefficient_entropy",
    "wavelet_high_low_energy_ratio", "wavelet_smoothed_mean_32",
]
BOOT_REPS, BOOT_SEED = 2000, 20261127

def client():
    return boto3.client("s3",
        endpoint_url=f"https://{env('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
        aws_access_key_id=env("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=env("R2_SECRET_ACCESS_KEY"),
        region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}))

def fit_predict(tr, va, ev, cols, cats, label):
    frames = [tr.copy(), va.copy(), ev.copy()]
    for frame in frames:
        for c in cats:
            frame[c] = frame[c].astype("category")
    tr, va, ev = frames
    model = lgb.train(
        {"objective":"regression", "metric":"l1", "learning_rate":0.05,
         "num_leaves":63, "max_depth":-1, "min_data_in_leaf":100,
         "feature_fraction":0.85, "bagging_fraction":0.85, "bagging_freq":1,
         "lambda_l2":2.0, "verbosity":-1, "seed":42, "num_threads":4},
        lgb.Dataset(tr[cols], label=tr.sale_amount, categorical_feature=cats, free_raw_data=False),
        num_boost_round=1200,
        valid_sets=[lgb.Dataset(va[cols], label=va.sale_amount, categorical_feature=cats)],
        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(200)])
    pv = np.maximum(model.predict(va[cols], num_iteration=model.best_iteration), 0)
    pe = np.maximum(model.predict(ev[cols], num_iteration=model.best_iteration), 0)
    print(f"{label}: best_iteration={model.best_iteration}")
    return model, pv, pe

def bootstrap(y, baseline, candidates):
    y = np.asarray(y, dtype=float)
    baseline = np.maximum(np.asarray(baseline, dtype=float), 0)
    base_abs = np.abs(y-baseline)
    base_sq = np.square(y-baseline)
    rng = np.random.default_rng(BOOT_SEED)
    draws = {name: {m: [] for m in ("mae","rmse","wape")} for name in candidates}
    for _ in range(BOOT_REPS):
        idx = rng.integers(0, len(y), size=len(y))
        denom_y = max(float(np.abs(y[idx]).sum()), 1e-12)
        b_mae = float(base_abs[idx].mean())
        b_rmse = float(np.sqrt(base_sq[idx].mean()))
        b_wape = float(base_abs[idx].sum()/denom_y)
        for name, pred in candidates.items():
            err = np.abs(y[idx]-pred[idx])
            c_mae = float(err.mean())
            c_rmse = float(np.sqrt(np.square(y[idx]-pred[idx]).mean()))
            c_wape = float(err.sum()/denom_y)
            draws[name]["mae"].append((b_mae-c_mae)/max(b_mae,1e-12)*100)
            draws[name]["rmse"].append((b_rmse-c_rmse)/max(b_rmse,1e-12)*100)
            draws[name]["wape"].append((b_wape-c_wape)/max(b_wape,1e-12)*100)
    out = {}
    for name, metrics in draws.items():
        out[name] = {}
        for metric, vals in metrics.items():
            a = np.asarray(vals)
            lo, hi = np.percentile(a, [2.5,97.5])
            out[name][metric] = {
                "mean_improvement_pct":float(a.mean()),
                "ci_95_low_pct":float(lo), "ci_95_high_pct":float(hi),
                "probability_improvement":float((a>0).mean()),
                "ci_excludes_zero":bool(lo>0 or hi<0)}
    return {
        "method":"paired nonparametric row bootstrap; same sampled indices for all candidates",
        "replicates":BOOT_REPS, "seed":BOOT_SEED, "evaluation_observations":len(y),
        "comparisons":out,
        "caveat":"Rows within store-product series and dates may be dependent. Row-bootstrap CIs can be optimistic; use series-cluster bootstrap before strong inferential claims."
    }

def split(frame, valid_dates, eval_dates):
    frame = frame.sort(["store_id","product_id","date"])
    tr = frame.filter(~pl.col("date").is_in(list(valid_dates)) & ~pl.col("date").cast(pl.String).is_in(list(eval_dates)))
    va = frame.filter(pl.col("date").is_in(list(valid_dates)))
    ev = frame.filter(pl.col("date").cast(pl.String).is_in(list(eval_dates)))
    return [x.to_pandas() for x in (tr,va,ev)]

def main():
    s3, bucket = client(), env("R2_BUCKET")
    with tempfile.TemporaryDirectory(prefix="frn-wavelet-robust-") as td:
        root=Path(td); raw={}
        for name,key in RAW.items():
            path=root/f"{name}.parquet"; print(f"Downloading {key}")
            s3.download_file(bucket,key,str(path)); raw[name]=pl.read_parquet(path)
        eval_dates=set(raw["eval"].get_column("dt").unique().to_list())
        combined=pl.concat([raw["train"],raw["eval"]],how="vertical_relaxed")
        base=load_features_from_frame(combined)
        wave_input=combined.select(["store_id","product_id","dt","sale_amount"]).with_columns(
            pl.col("dt").str.to_date(strict=False).alias("date")).drop("dt").to_pandas()
        base_cols=[c for c in base.columns if c not in ("sale_amount","date")]
        cats=[c for c in ("city_id","store_id","management_group_id","first_category_id",
            "second_category_id","third_category_id","product_id","day_of_week","month") if c in base_cols]
        family_results=[]; family_preds={}; family_valid={}; family_eval_frame={}
        db2_frame=None; valid_dates=None; train_dates=None
        for family in FAMILIES:
            print(f"Building trailing 32-day {family} features...")
            wf=build_wavelet_features(wave_input,wavelet_name=family)
            wf["date"]=pd.to_datetime(wf["date"]).dt.date
            wpl=pl.from_pandas(wf).with_columns(pl.col("date").cast(pl.Date))
            frame=base.join(wpl,on=["store_id","product_id","date"],how="inner").sort(["store_id","product_id","date"])
            if db2_frame is None:
                dates=frame.filter(~pl.col("date").cast(pl.String).is_in(list(eval_dates))).select(
                    pl.col("date").unique().sort()).to_series().to_list()
                valid_dates=set(dates[-14:])
                db2_frame=frame
            else:
                frame=frame.join(db2_frame.select(["store_id","product_id","date"]),
                    on=["store_id","product_id","date"],how="inner").sort(["store_id","product_id","date"])
            tr,va,ev=split(frame,valid_dates,eval_dates)
            yv,ye=va.sale_amount.to_numpy(),ev.sale_amount.to_numpy()
            model,pv,pe=fit_predict(tr,va,ev,base_cols+WCOLS,cats,f"family {family}")
            family_eval_frame[family]=ev
            family_valid[family]=pv; family_preds[family]=pe
            family_results.append({
                "family":family,
                **{f"validation_{k}":v for k,v in metric_pack(yv,pv).items()},
                **{f"evaluation_{k}":v for k,v in metric_pack(ye,pe).items()},
                "best_iteration":int(model.best_iteration)})
        # Use the db2 common rows as the comparison reference.
        tr,va,ev=split(db2_frame,valid_dates,eval_dates)
        yv,ye=va.sale_amount.to_numpy(),ev.sale_amount.to_numpy()
        base_model,base_v,base_e=fit_predict(tr,va,ev,base_cols,cats,"same-row LightGBM baseline")
        baseline_metrics={"validation":metric_pack(yv,base_v),"evaluation":metric_pack(ye,base_e)}
        # Leave-one-feature-out ablation, retraining each model on same rows.
        ablations=[]
        for removed in WCOLS:
            cols=base_cols+[c for c in WCOLS if c!=removed]
            model,pv,pe=fit_predict(tr,va,ev,cols,cats,f"ablation remove {removed}")
            vp,ep=metric_pack(yv,pv),metric_pack(ye,pe)
            full=metric_pack(ye,family_preds["db2"])
            ablations.append({
                "removed_feature":removed,
                **{f"validation_{k}":v for k,v in vp.items()},
                **{f"evaluation_{k}":v for k,v in ep.items()},
                "mae_change_vs_full_db2_pct":(ep["mae"]/full["mae"]-1)*100,
                "wape_change_vs_full_db2_pct":(ep["wape_pct"]/full["wape_pct"]-1)*100,
                "best_iteration":int(model.best_iteration)})
        boot=bootstrap(ye,base_e,{f"wavelet_{f}":family_preds[f] for f in FAMILIES})
        config={
            "families":list(FAMILIES),"features":WCOLS,"history_window_days":32,
            "decomposition_level":2,"train_rows":len(tr),"validation_rows":len(va),"evaluation_rows":len(ev),
            "validation_period":{"start":str(min(valid_dates)),"end":str(max(valid_dates))},
            "evaluation_period":{"start":str(ev.date.min()),"end":str(ev.date.max())},
            "ablation":"leave-one-wavelet-feature-out; retrain per feature",
            "comparison":"same chronological splits and same eligible store-product-date rows",
            "bootstrap_replicates":BOOT_REPS,"bootstrap_seed":BOOT_SEED,
            "target":"sale_amount","target_semantics":"Observed sales, not necessarily uncensored demand",
            "leakage_control":"Each trailing 32-observation window ends at t-1",
            "generated_at_utc":datetime.now(timezone.utc).isoformat()}
        predictions=pd.DataFrame({
            "date":ev.date.astype(str),"store_id":ev.store_id.astype(str),
            "product_id":ev.product_id.astype(str),"actual_sale_amount":ye,
            "seasonal_naive_lag7":ev.sales_lag_7.to_numpy(),
            "lightgbm_same_rows":base_e,
            **{f"wavelet_{f}":family_preds[f] for f in FAMILIES}})
        artifacts={
            "wavelet_family_comparison.csv":pd.DataFrame(family_results),
            "wavelet_ablation.csv":pd.DataFrame(ablations),
            "wavelet_bootstrap_significance.json":boot,
            "wavelet_robustness_config.json":config,
            "wavelet_robustness_predictions.csv":predictions,
            "wavelet_robustness_metrics.json":{"baseline":baseline_metrics,"families":family_results,
                "ablation":ablations,"bootstrap":boot,"config":config}}
        for filename,payload in artifacts.items():
            path=root/filename
            if filename.endswith(".csv"): payload.to_csv(path,index=False)
            else: path.write_text(json.dumps(payload,indent=2))
            key=f"{OUT}/{filename}"
            s3.upload_file(str(path),bucket,key,ExtraArgs={"ContentType":"text/csv" if filename.endswith(".csv") else "application/json"})
            head=s3.head_object(Bucket=bucket,Key=key)
            if head["ContentLength"]!=path.stat().st_size: raise RuntimeError(f"R2 verification failed: {key}")
            print(f"Verified s3://{bucket}/{key} ({path.stat().st_size:,} bytes)")
        print(json.dumps({"baseline":baseline_metrics,"families":family_results,"ablation":ablations,"bootstrap":boot},indent=2))

if __name__=="__main__": main()
