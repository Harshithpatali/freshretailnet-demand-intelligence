"""Train leakage-aware P10/P50/P90 LightGBM quantile models and publish to R2."""
import json, os, tempfile
from pathlib import Path
from datetime import datetime, timezone
import boto3, lightgbm as lgb, numpy as np, pandas as pd, polars as pl
from botocore.config import Config

QUANTILES=(0.10,0.50,0.90)
def env(k):
    v=os.getenv(k)
    if not v: raise RuntimeError(f"Missing secret {k}")
    return v
def client():
    return boto3.client("s3",endpoint_url=f"https://{env('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
        aws_access_key_id=env("R2_ACCESS_KEY_ID"),aws_secret_access_key=env("R2_SECRET_ACCESS_KEY"),
        region_name="auto",config=Config(signature_version="s3v4",retries={"max_attempts":5,"mode":"standard"}))
def make_features(df):
    required={"store_id","product_id","dt","sale_amount","hours_sale","hours_stock_status","stock_hour6_22_cnt"}
    if required-set(df.columns): raise ValueError(f"Missing columns: {sorted(required-set(df.columns))}")
    df=df.with_columns([
        pl.col("dt").str.to_date(strict=False).alias("date"),
        pl.col("hours_sale").list.sum().alias("hourly_sales_sum"),
        pl.col("hours_sale").list.mean().alias("hourly_sales_mean"),
        pl.col("hours_sale").list.max().alias("hourly_sales_peak"),
        pl.col("hours_sale").list.len().alias("hourly_slots"),
        pl.col("hours_stock_status").list.sum().alias("stock_status_sum"),
        pl.col("hours_stock_status").list.len().alias("stock_status_slots"),
    ]).with_columns([
        pl.col("date").dt.weekday().alias("day_of_week"),pl.col("date").dt.day().alias("day_of_month"),
        pl.col("date").dt.month().alias("month"),pl.col("date").dt.ordinal_day().alias("day_of_year"),
        pl.col("date").dt.week().alias("week_of_year"),
    ]).sort(["store_id","product_id","date"])
    grp=["store_id","product_id"]; expr=[]
    for lag in (1,7,14,28):
        expr += [pl.col("sale_amount").shift(lag).over(grp).alias(f"sales_lag_{lag}"),
                 pl.col("stock_status_sum").shift(lag).over(grp).alias(f"stock_status_lag_{lag}")]
    df=df.with_columns(expr).with_columns([
        pl.col("sale_amount").shift(1).rolling_mean(window_size=7,min_samples=1).over(grp).alias("sales_roll_mean_7"),
        pl.col("sale_amount").shift(1).rolling_mean(window_size=28,min_samples=1).over(grp).alias("sales_roll_mean_28"),
        pl.col("sale_amount").shift(1).rolling_std(window_size=7,min_samples=2).over(grp).fill_null(0).alias("sales_roll_std_7")])
    excluded={"dt","date","sale_amount","hours_sale","hours_stock_status","hourly_sales_sum",
      "hourly_sales_mean","hourly_sales_peak","hourly_slots","stock_status_sum","stock_status_slots","stock_hour6_22_cnt"}
    feats=[c for c in df.columns if c not in excluded]
    return df.drop_nulls(subset=["sales_lag_1","sales_lag_7"]).select(["date","store_id","product_id","sale_amount","stock_hour6_22_cnt"]+feats)
def pinball(y,p,q):
    e=np.asarray(y)-np.asarray(p)
    return float(np.maximum(q*e,(q-1)*e).mean())
def metrics(y,p):
    y=np.asarray(y)
    return {"pinball_loss":{f"p{int(q*100)}":pinball(y,p[q],q) for q in QUANTILES},
      "p10_p90_coverage_pct":float(((y>=p[.10])&(y<=p[.90])).mean()*100),
      "p10_p90_mean_width":float((p[.90]-p[.10]).mean()),
      "quantile_crossing_rate_pct":float(((p[.10]>p[.50])|(p[.50]>p[.90])).mean()*100),
      "median_mae":float(np.abs(y-p[.50]).mean()),
      "median_wape_pct":float(np.abs(y-p[.50]).sum()/max(np.abs(y).sum(),1e-9)*100)}
def conformal_adjustment(y,lower,upper,coverage=.80):
    """Finite-sample split conformal adjustment using validation residual scores."""
    y=np.asarray(y); lower=np.asarray(lower); upper=np.asarray(upper)
    scores=np.maximum(lower-y,y-upper)
    rank=min(len(scores),int(np.ceil((len(scores)+1)*coverage)))
    return float(np.sort(scores)[rank-1])
def calibrated_interval(y,p,qhat):
    lower=np.maximum(0,p[.10]-qhat)
    upper=np.maximum(lower,p[.90]+qhat)
    y=np.asarray(y)
    return {"coverage_pct":float(((y>=lower)&(y<=upper)).mean()*100),
            "mean_width":float((upper-lower).mean()),
            "adjustment":float(qhat)},lower,upper
def strata(frame,p):
    z=pd.DataFrame({"exposure":frame.stock_hour6_22_cnt.to_numpy(),"actual":frame.sale_amount.to_numpy(),
                    "p10":p[.10],"p90":p[.90]})
    bins=[-np.inf,0,2,5,9,15,np.inf]
    labels=["0_hours","1_2_hours","3_5_hours","6_9_hours","10_15_hours","16_hours"]
    z["group"]=pd.cut(z.exposure,bins=bins,labels=labels)
    return {str(k):{"rows":int(len(g)),"coverage_pct":float(((g.actual>=g.p10)&(g.actual<=g.p90)).mean()*100),
                    "mean_interval_width":float((g.p90-g.p10).mean())}
            for k,g in z.groupby("group",observed=True)}
def main():
    s3=client(); bucket=env("R2_BUCKET")
    with tempfile.TemporaryDirectory(prefix="frn-quantiles-") as td:
        root=Path(td); raw={}
        for split,key in [("train","raw/train.parquet"),("eval","raw/eval.parquet")]:
            f=root/f"{split}.parquet"; print(f"Downloading {key}"); s3.download_file(bucket,key,str(f)); raw[split]=pl.read_parquet(f)
        eval_dates=set(raw["eval"]["dt"].unique().to_list())
        all_df=make_features(pl.concat([raw["train"],raw["eval"]],how="vertical_relaxed"))
        ev=all_df.filter(pl.col("date").cast(pl.String).is_in(list(eval_dates)))
        tr=all_df.filter(~pl.col("date").cast(pl.String).is_in(list(eval_dates)))
        dates=tr.select(pl.col("date").unique().sort()).to_series().to_list()
        if len(dates)<30: raise ValueError("Insufficient chronological history")
        vd=set(dates[-14:])
        a=tr.filter(~pl.col("date").is_in(list(vd))).drop("date").to_pandas()
        v=tr.filter(pl.col("date").is_in(list(vd))).drop("date").to_pandas()
        e=ev.to_pandas()
        target="sale_amount"; diag={"stock_hour6_22_cnt"}
        feats=[c for c in a.columns if c not in ({target}|diag)]
        cats=[c for c in ("city_id","store_id","management_group_id","first_category_id","second_category_id",
                          "third_category_id","product_id","day_of_week","month") if c in feats]
        for d in (a,v,e):
            for c in cats: d[c]=d[c].astype("category")
        Xtr,ytr=a[feats],a[target]; Xv,yv=v[feats],v[target]; Xe,ye=e[feats],e[target]
        models={}; vp={}; ep={}; paths=[]
        for q in QUANTILES:
            name=f"p{int(q*100)}"; print(f"Training {name}")
            params={"objective":"quantile","alpha":q,"metric":"quantile","learning_rate":.05,
              "num_leaves":63,"max_depth":-1,"min_data_in_leaf":100,"feature_fraction":.85,
              "bagging_fraction":.85,"bagging_freq":1,"lambda_l2":2.0,"verbosity":-1,"seed":42,"num_threads":4}
            m=lgb.train(params,lgb.Dataset(Xtr,label=ytr,categorical_feature=cats,free_raw_data=False),
              num_boost_round=1200,valid_sets=[lgb.Dataset(Xv,label=yv,categorical_feature=cats)],
              callbacks=[lgb.early_stopping(80),lgb.log_evaluation(100)])
            models[q]=m; vp[q]=np.maximum(m.predict(Xv,num_iteration=m.best_iteration),0)
            ep[q]=np.maximum(m.predict(Xe,num_iteration=m.best_iteration),0)
            f=root/f"lightgbm_{name}.txt"; m.save_model(str(f)); paths.append(f)
        qhat=conformal_adjustment(yv,vp[.10],vp[.90],coverage=.80)
        val_cal,val_lo,val_hi=calibrated_interval(yv,vp,qhat)
        eval_cal,eval_lo,eval_hi=calibrated_interval(ye,ep,qhat)
        calibrated_ep={.10:eval_lo,.50:ep[.50],.90:eval_hi}
        result={"validation":metrics(yv,vp),"evaluation":metrics(ye,ep),
          "conformal_calibration":{"method":"split conformalized quantile regression",
            "nominal_coverage_pct":80.0,"validation_adjustment":qhat,
            "validation_calibrated_interval":val_cal,
            "evaluation_calibrated_interval":eval_cal,
            "caveat":"Calibration is estimated on a prior time window; temporal distribution shift can invalidate nominal future coverage."},
          "evaluation_calibrated_stockout_exposure_strata":strata(e,calibrated_ep),
          "evaluation_stockout_exposure_strata":strata(e,ep),
          "validation_period":{"start":str(min(vd)),"end":str(max(vd))},
          "evaluation_period":{"start":str(ev.get_column("date").min()),"end":str(ev.get_column("date").max())},
          "rows":{"train":len(a),"validation":len(v),"evaluation":len(e)},
          "best_iterations":{f"p{int(q*100)}":int(models[q].best_iteration) for q in QUANTILES},
          "features":feats,"categorical_features":cats,"quantiles":list(QUANTILES),
          "nominal_interval_coverage_pct":80.0,"target":"sale_amount",
          "target_semantics":"Observed daily sales, not guaranteed uncensored demand.",
          "leakage_note":"Contemporaneous hourly sales and stock status are excluded from predictors. Current-day stock exposure is only used for post-hoc stratification.",
          "stockout_note":"Coverage by stockout exposure is descriptive, not causal. Confirm status encoding against dataset technical documentation.",
          "generated_at_utc":datetime.now(timezone.utc).isoformat()}
        mf=root/"quantile_metrics.json"; mf.write_text(json.dumps(result,indent=2,default=str))
        pf=root/"evaluation_quantile_predictions.csv"
        pd.DataFrame({"date":e["date"].astype(str),"store_id":e["store_id"],"product_id":e["product_id"],
                      "actual_sale_amount":ye.to_numpy(),"p10":ep[.10],"p50":ep[.50],"p90":ep[.90],
                      "p10_calibrated":eval_lo,"p90_calibrated":eval_hi,
                      "stock_hour6_22_cnt":e["stock_hour6_22_cnt"].to_numpy()}).to_csv(pf,index=False)
        for f in paths+[mf,pf]:
            key=f"quantile-model-artifacts/{f.name}"
            typ="application/json" if f.suffix==".json" else "text/csv" if f.suffix==".csv" else "text/plain"
            s3.upload_file(str(f),bucket,key,ExtraArgs={"ContentType":typ})
            size=s3.head_object(Bucket=bucket,Key=key)["ContentLength"]
            if size!=f.stat().st_size: raise RuntimeError(f"R2 verification failed: {key}")
            print(f"Verified s3://{bucket}/{key} ({size:,} bytes)")
        print(json.dumps(result,indent=2,default=str))
if __name__=="__main__": main()
