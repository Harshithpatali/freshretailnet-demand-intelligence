"""Demand segmentation and direct multi-horizon retail forecasting experiment.

Uses only lagged sales/stock/covariate history for predictors. Forecast horizons
are evaluated against future target dates, with a chronological validation block.
Optional price/promotion columns are detected from raw schema and used only lagged.
"""
from __future__ import annotations
import json, os, re, sys, tempfile
from pathlib import Path
from datetime import datetime, timezone
import boto3, lightgbm as lgb, numpy as np, pandas as pd, polars as pl
from botocore.config import Config
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.train_model import RAW, env, load_features_from_frame

OUT="demand-segmentation-multihorizon-artifacts"
GROUP=["store_id","product_id"]
SEED=20261130
HORIZONS=(1,7,14)
PARAMS=[
 {"learning_rate":0.04,"num_leaves":31,"min_data_in_leaf":150,"feature_fraction":0.85,"lambda_l2":3.0},
 {"learning_rate":0.04,"num_leaves":63,"min_data_in_leaf":150,"feature_fraction":0.85,"lambda_l2":5.0},
 {"learning_rate":0.025,"num_leaves":63,"min_data_in_leaf":100,"feature_fraction":0.90,"lambda_l2":5.0},
]

def r2():
 return boto3.client("s3",endpoint_url=f"https://{env('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
  aws_access_key_id=env("R2_ACCESS_KEY_ID"),aws_secret_access_key=env("R2_SECRET_ACCESS_KEY"),
  region_name="auto",config=Config(signature_version="s3v4",retries={"max_attempts":5,"mode":"standard"}))

def add_demand_profile(frame):
 frame=frame.sort(GROUP+["date"])
 past=pl.col("sale_amount").shift(1)
 frame=frame.with_columns([
  past.rolling_sum(window_size=56,min_samples=1).over(GROUP).alias("_sales_sum_56"),
  (past>0).cast(pl.Float64).rolling_sum(window_size=56,min_samples=1).over(GROUP).alias("_nonzero_56"),
  past.rolling_mean(window_size=56,min_samples=2).over(GROUP).alias("_mean_56"),
  past.rolling_var(window_size=56,min_samples=2).over(GROUP).fill_null(0).alias("_var_56"),
 ])
 frame=frame.with_columns([
  (56/(pl.col("_nonzero_56")+1e-6)).alias("demand_adi_56"),
  (pl.col("_var_56")/(pl.col("_mean_56")**2+1e-6)).alias("demand_cv2_56"),
  (1-pl.col("_nonzero_56")/56).clip(0,1).alias("demand_zero_share_56"),
 ])
 frame=frame.with_columns(
  pl.when((pl.col("demand_adi_56")<1.32)&(pl.col("demand_cv2_56")<0.49)).then(pl.lit("smooth"))
   .when((pl.col("demand_adi_56")>=1.32)&(pl.col("demand_cv2_56")<0.49)).then(pl.lit("intermittent"))
   .when((pl.col("demand_adi_56")<1.32)&(pl.col("demand_cv2_56")>=0.49)).then(pl.lit("erratic"))
   .otherwise(pl.lit("lumpy")).alias("demand_segment")
 )
 return frame.drop(["_sales_sum_56","_nonzero_56","_mean_56","_var_56"])

def metrics(y,p):
 y=np.asarray(y,float); p=np.maximum(np.asarray(p,float),0); e=np.abs(y-p)
 return {"mae":float(e.mean()),"rmse":float(np.sqrt(np.mean((y-p)**2))),
  "wape_pct":float(100*e.sum()/max(np.abs(y).sum(),1e-9)),
  "bias_pct":float(100*(p-y).sum()/max(y.sum(),1e-9))}

def category_align(frames,cats):
 fs=[x.copy() for x in frames]
 for c in cats:
  levels=pd.Index(pd.concat([x[c].astype(str) for x in fs]).dropna().unique())
  dtype=pd.CategoricalDtype(categories=levels)
  for x in fs: x[c]=x[c].astype(str).astype(dtype)
 return fs

def fit_one(tr,va,ev,features,cats):
 tr,va,ev=category_align([tr,va,ev],cats)
 best=None
 for p in PARAMS:
  dtr=lgb.Dataset(tr[features],label=tr.target,categorical_feature=cats,free_raw_data=False)
  dva=lgb.Dataset(va[features],label=va.target,categorical_feature=cats,reference=dtr,free_raw_data=False)
  model=lgb.train({"objective":"regression","metric":"l1","verbosity":-1,"seed":SEED,"num_threads":4,
   "bagging_fraction":0.85,"bagging_freq":1,**p},dtr,num_boost_round=1400,valid_sets=[dva],
   callbacks=[lgb.early_stopping(90,verbose=False)])
  pred=np.maximum(model.predict(va[features],num_iteration=model.best_iteration),0)
  score=float(np.mean(np.abs(va.target.to_numpy()-pred)))
  if best is None or score<best[0]: best=(score,p,model)
 score,p,model=best
 pred=np.maximum(model.predict(ev[features],num_iteration=model.best_iteration),0)
 gain=pd.DataFrame({"feature":features,"gain":model.feature_importance(importance_type="gain")}).sort_values("gain",ascending=False)
 return pred,{"validation_mae":score,"best_params":p,"best_iteration":int(model.best_iteration),
  "feature_importance":gain.to_dict("records")}

def make_horizon_frame(frame,h):
 group=GROUP
 frame=frame.sort(group+["date"])
 frame=frame.with_columns([
  pl.col("sale_amount").shift(-h).over(group).alias("target"),
  pl.col("date").shift(-h).over(group).alias("target_date"),
 ])
 frame=frame.with_columns([
  pl.col("target_date").dt.weekday().alias("target_day_of_week"),
  pl.col("target_date").dt.day().alias("target_day_of_month"),
  pl.col("target_date").dt.month().alias("target_month"),
  pl.col("target_date").dt.ordinal_day().alias("target_day_of_year"),
  (2*np.pi*pl.col("target_date").dt.weekday()/7).sin().alias("target_dow_sin"),
  (2*np.pi*pl.col("target_date").dt.weekday()/7).cos().alias("target_dow_cos"),
  (2*np.pi*pl.col("target_date").dt.ordinal_day()/365.25).sin().alias("target_doy_sin"),
  (2*np.pi*pl.col("target_date").dt.ordinal_day()/365.25).cos().alias("target_doy_cos"),
 ]).drop_nulls(["target","target_date"])
 return frame

def main():
 s3,bucket=r2(),env("R2_BUCKET")
 with tempfile.TemporaryDirectory(prefix="frn-demand-segments-") as td:
  root=Path(td); raw={}
  for split,key in RAW.items():
   path=root/f"{split}.parquet"; print(f"Downloading {key}")
   s3.download_file(bucket,key,str(path)); raw[split]=pl.read_parquet(path)
  raw_cols=raw["train"].columns
  combined=pl.concat([raw["train"],raw["eval"]],how="diagonal_relaxed").sort(GROUP+["dt"])
  base=load_features_from_frame(combined)
  frame=add_demand_profile(base)
  # Optional covariates: use only one-day-lagged numeric fields whose names indicate price/promotion.
  raw_schema=combined.schema
  optional=[]
  for c,dtype in raw_schema.items():
   if re.search(r"price|promo|discount|markdown",c,re.I) and c not in GROUP+["dt","sale_amount"]:
    if dtype in (pl.Float32,pl.Float64,pl.Int8,pl.Int16,pl.Int32,pl.Int64,pl.UInt8,pl.UInt16,pl.UInt32,pl.UInt64):
     optional.append(c)
  if optional:
   lagged=[pl.col(c).shift(1).over(GROUP).alias(f"{c}_lag1") for c in optional]
   frame=frame.with_columns(lagged)
  # Remove contemporaneous optional price/promo fields; retain their lagged versions only.
  drop_now=[c for c in optional if c in frame.columns]
  if drop_now: frame=frame.drop(drop_now)
  horizon_results=[]; pred_rows=[]; segment_rows=[]; audits=[]
  all_dates=sorted(pd.to_datetime(frame["date"].to_list()).normalize().unique())
  eval_source_dates=sorted(pd.to_datetime(raw["eval"]["dt"].to_list()).normalize().unique())
  for h in HORIZONS:
   hf=make_horizon_frame(frame,h)
   dates=sorted(pd.to_datetime(hf["target_date"].to_list()).normalize().unique())
   eval_dates=[d for d in dates if d in set(eval_source_dates)]
   if len(eval_dates)<3:
    audits.append({"horizon":h,"status":"skipped","reason":f"Only {len(eval_dates)} evaluation target dates available"})
    continue
   eval_days=eval_dates[-min(7,len(eval_dates)):]
   eval_start=eval_days[0]
   prior=[d for d in dates if d<eval_start]
   if len(prior)<14:
    audits.append({"horizon":h,"status":"skipped","reason":"Insufficient pre-evaluation dates for validation"})
    continue
   val_days=prior[-7:]
   train_days=prior[:-7]
   if len(train_days)<30:
    audits.append({"horizon":h,"status":"skipped","reason":"Insufficient training dates"})
    continue
   pdh=hf.to_pandas(); pdh["target_date"]=pd.to_datetime(pdh["target_date"])
   tr=pdh[pdh.target_date.isin(train_days)].copy()
   va=pdh[pdh.target_date.isin(val_days)].copy()
   ev=pdh[pdh.target_date.isin(eval_days)].copy()
   # Exclude target and date identity fields; use future calendar fields, past sales and past covariates.
   exclude={"target","target_date","date","sale_amount","dt","hours_sale","hours_stock_status"}
   features=[c for c in pdh.columns if c not in exclude and not c.startswith("hours_")]
   cats=[c for c in ["city_id","store_id","management_group_id","first_category_id","second_category_id",
    "third_category_id","product_id","day_of_week","month","demand_segment","target_day_of_week","target_month"] if c in features]
   pred,fit_report=fit_one(tr,va,ev,features,cats)
   naive_col=f"sales_lag_{h}"
   if naive_col not in ev.columns: naive_col="sales_lag_7" if "sales_lag_7" in ev.columns else "sales_lag_1"
   model_metrics=metrics(ev.target,pred); naive_metrics=metrics(ev.target,ev[naive_col].fillna(0))
   improvement={k:float((naive_metrics[k]-model_metrics[k])/max(abs(naive_metrics[k]),1e-9)*100) for k in ("mae","rmse","wape_pct")}
   horizon_results.append({"horizon_days":h,"evaluation_dates":{"start":str(min(eval_days).date()),"end":str(max(eval_days).date()),"count":len(eval_days)},
    "train_dates":len(train_days),"validation_dates":{"start":str(min(val_days).date()),"end":str(max(val_days).date())},
    "model":model_metrics,"seasonal_naive":naive_metrics,"improvement_vs_naive_pct":improvement,
    "fit":{k:v for k,v in fit_report.items() if k!="feature_importance"},
    "feature_importance_top20":fit_report["feature_importance"][:20],
    "train_rows":len(tr),"validation_rows":len(va),"evaluation_rows":len(ev),
    "feature_count":len(features),"optional_lagged_price_promo_columns":[f"{c}_lag1" for c in optional]})
   for i,(_,row) in enumerate(ev.iterrows()):
    pred_rows.append({"horizon_days":h,"origin_date":row.date.date().isoformat(),"target_date":row.target_date.date().isoformat(),
     "store_id":str(row.store_id),"product_id":str(row.product_id),"actual":float(row.target),
     "prediction":float(pred[i]),"seasonal_naive":float(row[naive_col] if pd.notna(row[naive_col]) else 0),
     "demand_segment":str(row.demand_segment)})
   seg=pd.DataFrame({"actual":ev.target.to_numpy(),"prediction":pred,"segment":ev.demand_segment.astype(str).to_numpy()})
   for segment,g in seg.groupby("segment"):
    segment_rows.append({"horizon_days":h,"demand_segment":segment,"rows":len(g),**metrics(g.actual,g.prediction)})
   audits.append({"horizon":h,"status":"completed","evaluation_dates":len(eval_days),"train_dates":len(train_days)})
  if not horizon_results: raise RuntimeError(f"No forecast horizons could be evaluated: {audits}")
  report={"experiment":"demand segmentation + direct multi-horizon forecasting","horizons":horizon_results,
   "segment_metrics":segment_rows,"horizon_audit":audits,"raw_columns":raw_cols,
   "optional_price_promotion_columns_detected":optional,
   "holiday_data_available":any(re.search(r"holiday|festival",c,re.I) for c in raw_cols),
   "holiday_note":"No external holiday calendar was introduced. Only target-date calendar features are used; holiday features require a documented source field/calendar.",
   "demand_segmentation":"Trailing 56-observation ADI/CV² categories: smooth, intermittent, erratic, lumpy; computed from history ending at origin_date-1.",
   "forecast_design":"Separate direct LightGBM model per horizon; target date shifted within store-product; validation is the seven dates immediately before the held-out evaluation dates.",
   "leakage_control":"No contemporaneous target, hourly sales, stock status, or price/promotion values are predictors. Optional numeric price/promotion fields are lagged one observation.",
   "generated_at_utc":datetime.now(timezone.utc).isoformat()}
  outputs={"demand_multihorizon_report.json":report,"demand_multihorizon_predictions.csv":pd.DataFrame(pred_rows),
   "demand_segment_metrics.csv":pd.DataFrame(segment_rows)}
  for name,obj in outputs.items():
   path=root/name
   if name.endswith(".csv"): obj.to_csv(path,index=False); ctype="text/csv"
   else: path.write_text(json.dumps(obj,indent=2,default=str)); ctype="application/json"
   key=f"{OUT}/{name}"; s3.upload_file(str(path),bucket,key,ExtraArgs={"ContentType":ctype})
   if s3.head_object(Bucket=bucket,Key=key)["ContentLength"]!=path.stat().st_size: raise RuntimeError(f"R2 verification failed: {key}")
   print(f"Verified s3://{bucket}/{key} ({path.stat().st_size:,} bytes)")
  print(json.dumps(report,indent=2,default=str))

if __name__=="__main__": main()
