"""Train leakage-aware FreshRetailNet daily demand models and publish artifacts to R2."""
from __future__ import annotations
import json, os, tempfile, hashlib
from pathlib import Path
from datetime import datetime, timezone

import boto3
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from botocore.config import Config
from sklearn.metrics import mean_absolute_error, mean_squared_error

RAW = {"train": "raw/train.parquet", "eval": "raw/eval.parquet"}
OUT = "model-artifacts"

def env(k):
    v=os.getenv(k)
    if not v: raise RuntimeError(f"Missing secret {k}")
    return v

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def r2_client():
    return boto3.client("s3", endpoint_url=f"https://{env('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
        aws_access_key_id=env("R2_ACCESS_KEY_ID"), aws_secret_access_key=env("R2_SECRET_ACCESS_KEY"),
        region_name="auto", config=Config(signature_version="s3v4",retries={"max_attempts":5,"mode":"standard"}))

def load_features_from_frame(df: pl.DataFrame) -> pl.DataFrame:
    required={"store_id","product_id","dt","sale_amount","hours_sale","hours_stock_status"}
    missing=required-set(df.columns)
    if missing: raise ValueError(f"Required columns absent: {sorted(missing)}")
    # Expand the daily 24-hour vectors into stable daily summary features.
    df=df.with_columns([
        pl.col("dt").str.to_date(strict=False).alias("date"),
        pl.col("hours_sale").list.sum().alias("hourly_sales_sum"),
        pl.col("hours_sale").list.mean().alias("hourly_sales_mean"),
        pl.col("hours_sale").list.max().alias("hourly_sales_peak"),
        pl.col("hours_sale").list.len().alias("hourly_slots"),
        pl.col("hours_stock_status").list.sum().alias("stock_status_sum"),
        pl.col("hours_stock_status").list.len().alias("stock_status_slots"),
    ])
    df=df.with_columns([
        pl.col("date").dt.weekday().alias("day_of_week"),
        pl.col("date").dt.day().alias("day_of_month"),
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.ordinal_day().alias("day_of_year"),
        pl.col("date").dt.week().alias("week_of_year"),
    ])
    df=df.sort(["store_id","product_id","date"])
    group=["store_id","product_id"]
    # Only past sales and past stock-status summaries are used for predictors.
    lag_expr=[]
    for lag in (1,7,14,28):
        lag_expr.append(pl.col("sale_amount").shift(lag).over(group).alias(f"sales_lag_{lag}"))
        lag_expr.append(pl.col("stock_status_sum").shift(lag).over(group).alias(f"stock_status_lag_{lag}"))
    df=df.with_columns(lag_expr)
    df=df.with_columns([
        pl.col("sale_amount").shift(1).rolling_mean(window_size=7,min_samples=1).over(group).alias("sales_roll_mean_7"),
        pl.col("sale_amount").shift(1).rolling_mean(window_size=28,min_samples=1).over(group).alias("sales_roll_mean_28"),
        pl.col("sale_amount").shift(1).rolling_std(window_size=7,min_samples=2).over(group).fill_null(0).alias("sales_roll_std_7"),
    ])
    # Do not include current-day hourly sales or current-day stock observations:
    # both are contemporaneous with the target and would leak the answer.
    excluded={"dt","date","sale_amount","hours_sale","hours_stock_status",
              "hourly_sales_sum","hourly_sales_mean","hourly_sales_peak","hourly_slots",
              "stock_status_sum","stock_status_slots","stock_hour6_22_cnt"}
    features=[c for c in df.columns if c not in excluded]
    df=df.drop_nulls(subset=["sales_lag_1","sales_lag_7"])
    return df.select(["date","sale_amount"]+features)

def metric_pack(y,p):
    p=np.maximum(np.asarray(p),0)
    y=np.asarray(y)
    denom=max(float(np.abs(y).sum()),1e-9)
    return {"mae":float(mean_absolute_error(y,p)),
            "rmse":float(np.sqrt(mean_squared_error(y,p))),
            "wape_pct":float(np.abs(y-p).sum()/denom*100),
            "bias_pct":float((p-y).sum()/max(float(y.sum()),1e-9)*100)}

def main():
    bucket=env("R2_BUCKET"); s3=r2_client()
    with tempfile.TemporaryDirectory(prefix="frn-model-") as td:
        root=Path(td); raw_frames={}
        for split,key in RAW.items():
            local=root/f"{split}.parquet"
            print(f"Downloading {key}")
            s3.download_file(bucket,key,str(local))
            raw_frames[split]=pl.read_parquet(local)
        # Build lags on combined chronological history so eval uses train context.
        eval_dates=set(raw_frames["eval"].get_column("dt").unique().to_list())
        combined=pl.concat([raw_frames["train"],raw_frames["eval"]],how="vertical_relaxed")
        all_features=load_features_from_frame(combined)
        eval_df=all_features.filter(pl.col("date").cast(pl.String).is_in(list(eval_dates)))
        train=all_features.filter(~pl.col("date").cast(pl.String).is_in(list(eval_dates)))
        print(f"train: {train.height:,}; eval: {eval_df.height:,}; features={len(train.columns)-2}")
        # Chronological holdout from the latest 14 distinct training dates.
        dates=train.select(pl.col("date").unique().sort()).to_series().to_list()
        if len(dates)<30: raise ValueError("Insufficient unique training dates for a 14-day holdout and lag history")
        valid_dates=set(dates[-14:])
        train_pd=train.filter(~pl.col("date").is_in(list(valid_dates))).drop("date").to_pandas()
        valid_pd=train.filter(pl.col("date").is_in(list(valid_dates))).drop("date").to_pandas()
        eval_pd=eval_df.drop("date").to_pandas()
        target="sale_amount"
        feature_cols=[c for c in train_pd.columns if c!=target]
        # Categorical IDs are handled as categorical features by LightGBM.
        cat_cols=[c for c in ("city_id","store_id","management_group_id","first_category_id","second_category_id","third_category_id","product_id","day_of_week","month") if c in feature_cols]
        for frame in (train_pd,valid_pd,eval_pd):
            for c in cat_cols:
                frame[c]=frame[c].astype("category")
        Xtr=train_pd[feature_cols]; ytr=train_pd[target]
        Xv=valid_pd[feature_cols]; yv=valid_pd[target]
        Xe=eval_pd[feature_cols]; ye=eval_pd[target]
        params={"objective":"regression","metric":"l1","learning_rate":0.05,"num_leaves":63,
                "max_depth":-1,"min_data_in_leaf":100,"feature_fraction":0.85,
                "bagging_fraction":0.85,"bagging_freq":1,"lambda_l2":2.0,
                "verbosity":-1,"seed":42,"num_threads":4}
        model=lgb.train(params,lgb.Dataset(Xtr,label=ytr,categorical_feature=cat_cols,free_raw_data=False),
                        num_boost_round=1200,
                        valid_sets=[lgb.Dataset(Xv,label=yv,categorical_feature=cat_cols,reference=None)],
                        callbacks=[lgb.early_stopping(80),lgb.log_evaluation(100)])
        pred_v=model.predict(Xv,num_iteration=model.best_iteration)
        pred_e=model.predict(Xe,num_iteration=model.best_iteration)
        # Naive seasonal baseline: same entity's sale seven days earlier.
        valid_base=valid_pd["sales_lag_7"].to_numpy()
        eval_base=eval_pd["sales_lag_7"].to_numpy()
        metrics={"validation":{"model":metric_pack(yv,pred_v),"seasonal_naive_lag7":metric_pack(yv,valid_base)},
                 "evaluation":{"model":metric_pack(ye,pred_e),"seasonal_naive_lag7":metric_pack(ye,eval_base)},
                 "validation_period":{"start":str(min(valid_dates)),"end":str(max(valid_dates))},
                 "evaluation_period":{"start":str(eval_df["date"].min()),"end":str(eval_df["date"].max())},
                 "rows":{"train":len(train_pd),"validation":len(valid_pd),"evaluation":len(eval_pd)},
                 "best_iteration":int(model.best_iteration),"features":feature_cols,
                 "categorical_features":cat_cols,
                 "target":"sale_amount",
                 "target_semantics":"Observed daily sales, not guaranteed uncensored demand.",
                 "stockout_note":"Current-day stock status and hourly sales are excluded from predictors; only lagged stock-status summaries are used. Stock-status coding semantics require source documentation before interpreting direction.",
                 "generated_at_utc":datetime.now(timezone.utc).isoformat()}
        model_path=root/"lightgbm_model.txt"; model.save_model(str(model_path))
        metrics_path=root/"metrics.json"; metrics_path.write_text(json.dumps(metrics,indent=2,default=str))
        importance=pd.DataFrame({"feature":feature_cols,"gain":model.feature_importance(importance_type="gain"),
                                  "split":model.feature_importance(importance_type="split")}).sort_values("gain",ascending=False)
        imp_path=root/"feature_importance.csv"; importance.to_csv(imp_path,index=False)
        pred_path=root/"evaluation_predictions.csv"
        pd.DataFrame({"actual_sale_amount":ye.to_numpy(),"prediction":np.maximum(pred_e,0),
                      "seasonal_naive_lag7":eval_base}).to_csv(pred_path,index=False)
        for p in (model_path,metrics_path,imp_path,pred_path):
            key=f"{OUT}/{p.name}"
            s3.upload_file(str(p),bucket,key,ExtraArgs={"ContentType":"application/json" if p.suffix==".json" else "text/csv" if p.suffix==".csv" else "text/plain"})
            head=s3.head_object(Bucket=bucket,Key=key)
            if head["ContentLength"]!=p.stat().st_size: raise RuntimeError(f"R2 upload verification failed: {key}")
            print(f"Verified s3://{bucket}/{key} ({p.stat().st_size:,} bytes)")
        print(json.dumps(metrics,indent=2,default=str))
if __name__=="__main__": main()
