"""Feature selection, wavelet signal ablation, LightGBM tuning and rolling backtests.

Uses causal features only. Three latest non-overlapping 7-day evaluation windows,
each preceded by a 14-day validation window. Results are written to Cloudflare R2.
"""
from __future__ import annotations
import json, tempfile, sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
import boto3, lightgbm as lgb, numpy as np, pandas as pd, polars as pl
from botocore.config import Config

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.train_model import RAW, env, load_features_from_frame, metric_pack
from experiments.wavelet.engineered_pipeline import add_engineered_features, wavelet_features, GROUP

OUT = "wavelet-selection-backtest-artifacts"
SEED = 20261129
BOOT_REPS = 2000
WINDOW = 32
FAMILY = "db2"
SIGNALS = ("sales", "sales_residual_7", "sales_momentum_7_28")
PARAM_GRID = [
    {"learning_rate": 0.04, "num_leaves": 31, "min_data_in_leaf": 150, "feature_fraction": 0.85, "lambda_l2": 3.0},
    {"learning_rate": 0.04, "num_leaves": 63, "min_data_in_leaf": 150, "feature_fraction": 0.85, "lambda_l2": 5.0},
    {"learning_rate": 0.025, "num_leaves": 63, "min_data_in_leaf": 100, "feature_fraction": 0.90, "lambda_l2": 5.0},
    {"learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 250, "feature_fraction": 0.90, "lambda_l2": 8.0},
]

def client():
    return boto3.client("s3", endpoint_url=f"https://{env('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
        aws_access_key_id=env("R2_ACCESS_KEY_ID"), aws_secret_access_key=env("R2_SECRET_ACCESS_KEY"),
        region_name="auto", config=Config(signature_version="s3v4", retries={"max_attempts":5,"mode":"standard"}))

def model_params(p):
    return {"objective":"regression","metric":"l1","verbosity":-1,"seed":42,"num_threads":4,
        "bagging_fraction":0.85,"bagging_freq":1, **p}

def align_categories(frames, cats):
    frames = [f.copy() for f in frames]
    for c in cats:
        levels = pd.Index(pd.concat([f[c].astype(str) for f in frames]).dropna().unique())
        dtype = pd.CategoricalDtype(categories=levels)
        for f in frames: f[c] = f[c].astype(str).astype(dtype)
    return frames

def fit(train, valid, test, cols, cats, params, label, importance=False):
    tr, va, te = align_categories([train,valid,test], cats)
    dtr = lgb.Dataset(tr[cols], label=tr.sale_amount, categorical_feature=cats, free_raw_data=False)
    dva = lgb.Dataset(va[cols], label=va.sale_amount, categorical_feature=cats, reference=dtr, free_raw_data=False)
    m = lgb.train(model_params(params), dtr, num_boost_round=1400, valid_sets=[dva],
        callbacks=[lgb.early_stopping(90, verbose=False)])
    pv = np.maximum(m.predict(va[cols], num_iteration=m.best_iteration),0)
    pt = np.maximum(m.predict(te[cols], num_iteration=m.best_iteration),0)
    gain = dict(zip(cols, m.feature_importance(importance_type="gain").tolist())) if importance else None
    print(f"{label}: iter={m.best_iteration}, val_mae={np.mean(np.abs(va.sale_amount.to_numpy()-pv)):.6f}")
    return m,pv,pt,gain

def metrics(y,p):
    return {k:float(v) for k,v in metric_pack(np.asarray(y),np.asarray(p)).items()}

def date_folds(all_dates, n=3, horizon=7, validation=14):
    ds = sorted(pd.to_datetime(list(all_dates)).normalize().unique())
    # Select the latest 3 consecutive, non-overlapping evaluation windows.
    folds=[]
    for k in range(n,0,-1):
        end_idx = len(ds) - (k-1)*horizon
        start_idx = end_idx-horizon
        if start_idx <= validation+10: continue
        ev = ds[start_idx:end_idx]
        va = ds[start_idx-validation:start_idx]
        train_end = start_idx-validation
        tr = ds[:train_end]
        folds.append((tr,va,ev))
    return folds

def two_way_date_series_bootstrap(pred_frame, baseline_col, candidate_cols):
    """Paired bootstrap with independent series-cluster and contiguous date-block weights."""
    rng=np.random.default_rng(SEED)
    df=pred_frame.copy()
    series=(df.store_id.astype(str)+"::"+df.product_id.astype(str)).to_numpy()
    unique_series, si=np.unique(series,return_inverse=True)
    dates=pd.to_datetime(df.date).dt.normalize().to_numpy()
    unique_dates=np.array(sorted(pd.unique(dates)))
    date_pos=np.searchsorted(unique_dates,dates)
    # Two-day moving blocks, circular at the end; folds remain separately represented by dates.
    block_len=2
    blocks=[np.arange(i,min(i+block_len,len(unique_dates))) for i in range(0,len(unique_dates),block_len)]
    out={c:{"mae":[],"rmse":[],"wape":[]} for c in candidate_cols}
    y=df.actual.to_numpy(float); b=df[baseline_col].to_numpy(float)
    for _ in range(BOOT_REPS):
        sw=np.bincount(rng.integers(0,len(unique_series),len(unique_series)),minlength=len(unique_series))
        chosen=rng.integers(0,len(blocks),len(blocks))
        dw=np.zeros(len(unique_dates),dtype=float)
        for bi in chosen: dw[blocks[bi]]+=1
        w=sw[si]*dw[date_pos]
        if w.sum()==0: continue
        den=max(np.sum(w*np.abs(y)),1e-12)
        be=np.abs(y-b)
        bm=np.sqrt(np.sum(w*(y-b)**2)/w.sum())
        base={"mae":np.sum(w*be)/w.sum(),"rmse":bm,"wape":np.sum(w*be)/den}
        for c in candidate_cols:
            p=df[c].to_numpy(float); e=np.abs(y-p)
            vals={"mae":np.sum(w*e)/w.sum(),"rmse":np.sqrt(np.sum(w*(y-p)**2)/w.sum()),"wape":np.sum(w*e)/den}
            for metric in out[c]: out[c][metric].append((base[metric]-vals[metric])/max(base[metric],1e-12)*100)
    result={}
    for c,ms in out.items():
        result[c]={}
        for metric,vals in ms.items():
            a=np.asarray(vals); lo,hi=np.percentile(a,[2.5,97.5])
            result[c][metric]={"mean_improvement_pct":float(a.mean()),"ci_95_low_pct":float(lo),
                "ci_95_high_pct":float(hi),"probability_improvement":float((a>0).mean()),
                "ci_excludes_zero":bool(lo>0 or hi<0)}
    return {"method":"paired two-way bootstrap: store-product clusters + contiguous 2-day date blocks",
        "replicates":BOOT_REPS,"seed":SEED,"series_clusters":int(len(unique_series)),
        "unique_dates":int(len(unique_dates)),"comparisons_vs_tuned_engineered":result,
        "caveat":"Only three rolling evaluation windows are available in this run; date-block inference remains provisional."}

def main():
    s3,bucket=client(),env("R2_BUCKET")
    with tempfile.TemporaryDirectory(prefix="frn-wavelet-selection-") as td:
        root=Path(td); raw={}
        for split,key in RAW.items():
            path=root/f"{split}.parquet"; print(f"Downloading {key}"); s3.download_file(bucket,key,str(path)); raw[split]=pl.read_parquet(path)
        combined=pl.concat([raw["train"],raw["eval"]],how="vertical_relaxed)
        base=load_features_from_frame(combined)
        frame=add_engineered_features(base)
        frame=frame.sort(GROUP+["date"])
        # Restrict all candidates to a shared 64-observation causal history.
        eligibility=combined.select(GROUP+["dt"]).with_columns(pl.col("dt").str.to_date(strict=False).alias("date")).sort(GROUP+["date"]).with_columns(
            pl.col("date").cum_count().over(GROUP).alias("_n")).filter(pl.col("_n")>64).select(GROUP+["date"])
        frame=frame.join(eligibility,on=GROUP+["date"],how="inner").sort(GROUP+["date"])
        wave_raw=combined.select(GROUP+["dt","sale_amount"]).with_columns(pl.col("dt").str.to_date(strict=False).alias("date")).drop("dt")
        wf=wavelet_features(wave_raw,WINDOW,FAMILY)
        wfpl=pl.from_pandas(wf).with_columns(pl.col("date").cast(pl.Date))
        frame=frame.join(wfpl,on=GROUP+["date"],how="inner").sort(GROUP+["date"])
        all_pd=frame.to_pandas(); all_pd["date"]=pd.to_datetime(all_pd["date"])
        base_cols=[c for c in all_pd.columns if c not in ("sale_amount","date") and not c.startswith("wv_")]
        cats=[c for c in ("city_id","store_id","management_group_id","first_category_id","second_category_id","third_category_id","product_id","day_of_week","month") if c in base_cols]
        wave_cols={s:[c for c in all_pd.columns if c.startswith(f"wv_{s}_")] for s in SIGNALS}
        folds=date_folds(all_pd.date.unique(),n=3,horizon=7,validation=14)
        if len(folds)<3: raise RuntimeError(f"Need three complete rolling folds; found {len(folds)}")
        prediction_rows=[]; fold_reports=[]; feature_gain={}
        for fold_no,(tr_dates,va_dates,ev_dates) in enumerate(folds,1):
            tr=all_pd[all_pd.date.isin(tr_dates)].copy(); va=all_pd[all_pd.date.isin(va_dates)].copy(); ev=all_pd[all_pd.date.isin(ev_dates)].copy()
            if min(len(tr),len(va),len(ev))==0: raise RuntimeError(f"Empty split in fold {fold_no}")
            # Tune on the 14-day validation window, then freeze selected config for evaluation.
            tune=[]
            for i,p in enumerate(PARAM_GRID):
                _,pv,_,_=fit(tr,va,va,base_cols,cats,p,f"fold{fold_no}/tune{i+1}")
                tune.append((float(np.mean(np.abs(va.sale_amount.to_numpy()-pv))),p))
            best_params=min(tune,key=lambda x:x[0])[1]
            _,pv,pe,gain=fit(tr,va,ev,base_cols,cats,best_params,f"fold{fold_no}/tuned_engineered",True)
            # Remove weak engineered features using gain-based selection, retain all original baseline predictors.
            original=[c for c in base_cols if c not in {
                "sales_lag_2","sales_lag_3","sales_lag_14","sales_lag_28","sales_lag_56",
                "sales_roll_mean_14","sales_roll_mean_56","sales_roll_std_14","sales_roll_std_28","sales_roll_std_56",
                "sales_roll_median_7","sales_roll_median_28","sales_roll_sum_28","sales_zero_rate_28",
                "sales_momentum_7_28","sales_cv_28","sales_lag_ratio_1_7","dow_sin","dow_cos","doy_sin","doy_cos"}]
            engineered=[c for c in base_cols if c not in original]
            positive=sorted([gain.get(c,0) for c in engineered if gain.get(c,0)>0],reverse=True)
            threshold=positive[max(0,int(len(positive)*0.35)-1)] if positive else float("inf")
            selected=original+[c for c in engineered if gain.get(c,0)>=threshold and gain.get(c,0)>0]
            _,svp,spe,_=fit(tr,va,ev,selected,cats,best_params,f"fold{fold_no}/selected_features")
            # Wavelet signal ablations: each signal alone, all pairs, and all three.
            signal_sets=[(s,) for s in SIGNALS]+[(SIGNALS[0],SIGNALS[1]),(SIGNALS[0],SIGNALS[2]),(SIGNALS[1],SIGNALS[2]),SIGNALS]
            fold_predictions={"tuned_engineered":pe,"selected_features":spe}
            for sigs in signal_sets:
                wcols=sum([wave_cols[s] for s in sigs],[])
                name="wavelet_"+"+".join(s.replace("sales_","") for s in sigs)
                _,_,wp,_=fit(tr,va,ev,selected+wcols,cats,best_params,f"fold{fold_no}/{name}")
                fold_predictions[name]=wp
            for model_name,pred in fold_predictions.items():
                for idx,(_,row) in enumerate(ev.iterrows()):
                    prediction_rows.append({"fold":fold_no,"date":row.date.date().isoformat(),"store_id":str(row.store_id),
                        "product_id":str(row.product_id),"actual":float(row.sale_amount),"model":model_name,"prediction":float(pred[idx])})
            fold_reports.append({"fold":fold_no,"train_start":str(min(tr_dates).date()),"train_end":str(max(tr_dates).date()),
                "validation_start":str(min(va_dates).date()),"validation_end":str(max(va_dates).date()),
                "evaluation_start":str(min(ev_dates).date()),"evaluation_end":str(max(ev_dates).date()),
                "best_hyperparameters":best_params,"baseline_feature_count":len(base_cols),
                "selected_feature_count":len(selected),"removed_engineered_features":[c for c in engineered if c not in selected],
                "validation_tuning_mae":float(min(tune,key=lambda x:x[0])[0])})
            feature_gain[str(fold_no)]={c:float(gain.get(c,0)) for c in engineered}
        pred=pd.DataFrame(prediction_rows)
        wide=pred.pivot(index=["fold","date","store_id","product_id","actual"],columns="model",values="prediction").reset_index()
        baseline="tuned_engineered"
        model_cols=[c for c in wide.columns if c not in ["fold","date","store_id","product_id","actual",baseline]]
        comparisons=[]
        for c in [baseline]+model_cols:
            comparisons.append({"model":c,**metrics(wide.actual,wide[c])})
        boot=two_way_date_series_bootstrap(wide,baseline,model_cols)
        report={"folds":fold_reports,"metrics_pooled_across_rolling_windows":comparisons,
            "feature_gain_by_fold":feature_gain,"wavelet_family":FAMILY,"wavelet_window":WINDOW,
            "wavelet_signal_sets":["sales","sales_residual_7","sales_momentum_7_28","sales+sales_residual_7",
                "sales+sales_momentum_7_28","sales_residual_7+sales_momentum_7_28","all_three"],
            "hyperparameter_candidates":PARAM_GRID,"date_block_bootstrap":boot,
            "evaluation_design":"Three non-overlapping 7-day evaluation windows; each preceded by a 14-day validation window; expanding training window.",
            "leakage_control":"Features are shifted; DWT windows end at t-1; validation only selects hyperparameters and feature thresholds; evaluation windows are not used for tuning.",
            "generated_at_utc":datetime.now(timezone.utc).isoformat()}
        artifacts={"rolling_wavelet_model_comparison.csv":pd.DataFrame(comparisons),
            "rolling_wavelet_predictions.csv":wide,"rolling_wavelet_feature_gain.json":feature_gain,
            "rolling_wavelet_backtest_report.json":report}
        for filename,payload in artifacts.items():
            path=root/filename
            if filename.endswith(".csv"): payload.to_csv(path,index=False); ctype="text/csv"
            else: path.write_text(json.dumps(payload,indent=2,default=str)); ctype="application/json"
            key=f"{OUT}/{filename}"; s3.upload_file(str(path),bucket,key,ExtraArgs={"ContentType":ctype})
            head=s3.head_object(Bucket=bucket,Key=key)
            if head["ContentLength"]!=path.stat().st_size: raise RuntimeError(f"R2 verification failed: {key}")
            print(f"Verified s3://{bucket}/{key} ({path.stat().st_size:,} bytes)")
        print(json.dumps(report,indent=2,default=str))

if __name__=="__main__": main()
