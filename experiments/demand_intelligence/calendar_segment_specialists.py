"""Target-date known calendar signals + demand-segment-specific forecasting.

Evaluation is chronological and directly compares the repository's standard
LightGBM strategy with an enhanced holiday/activity model and segment specialists.
Holiday/activity values are joined from the raw target-date row (known calendar
attributes), never from target sales or contemporaneous stock observations.
"""
from __future__ import annotations
import json, re, sys, tempfile
from pathlib import Path
from datetime import datetime, timezone
import boto3, lightgbm as lgb, numpy as np, pandas as pd, polars as pl
from botocore.config import Config
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from src.train_model import RAW, env, load_features_from_frame

OUT="demand-segment-strategy-artifacts"; GROUP=["store_id","product_id"]
SEED=20261201; HORIZONS=(1,7,14)
PARAMS={"learning_rate":0.04,"num_leaves":63,"min_data_in_leaf":150,
        "feature_fraction":0.85,"bagging_fraction":0.85,"bagging_freq":1,
        "lambda_l2":5.0,"verbosity":-1,"seed":SEED,"num_threads":4}

def s3client():
 return boto3.client("s3",endpoint_url=f"https://{env('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
  aws_access_key_id=env("R2_ACCESS_KEY_ID"),aws_secret_access_key=env("R2_SECRET_ACCESS_KEY"),
  region_name="auto",config=Config(signature_version="s3v4",retries={"max_attempts":5,"mode":"standard"}))

def metrics(y,p):
 y=np.asarray(y,float); p=np.maximum(np.asarray(p,float),0); e=np.abs(y-p)
 return {"mae":float(e.mean()),"rmse":float(np.sqrt(np.mean((y-p)**2))),
  "wape_pct":float(100*e.sum()/max(np.abs(y).sum(),1e-9)),
  "bias_pct":float(100*(p-y).sum()/max(y.sum(),1e-9))}

def segment_features(frame):
 frame=frame.sort(GROUP+["date"])
 past=pl.col("sale_amount").shift(1)
 frame=frame.with_columns([
  (past>0).cast(pl.Float64).rolling_sum(window_size=56,min_samples=1).over(GROUP).alias("_nz"),
  past.rolling_mean(window_size=56,min_samples=2).over(GROUP).alias("_mu"),
  past.rolling_var(window_size=56,min_samples=2).over(GROUP).fill_null(0).alias("_var")])
 frame=frame.with_columns([
  (56/(pl.col("_nz")+1e-6)).alias("demand_adi_56"),
  (pl.col("_var")/(pl.col("_mu")**2+1e-6)).alias("demand_cv2_56")])
 frame=frame.with_columns(pl.when((pl.col("demand_adi_56")<1.32)&(pl.col("demand_cv2_56")<0.49)).then(pl.lit("smooth"))
  .when((pl.col("demand_adi_56")>=1.32)&(pl.col("demand_cv2_56")<0.49)).then(pl.lit("intermittent"))
  .when((pl.col("demand_adi_56")<1.32)&(pl.col("demand_cv2_56")>=0.49)).then(pl.lit("erratic"))
  .otherwise(pl.lit("lumpy")).alias("demand_segment"))
 return frame.drop(["_nz","_mu","_var"])

def align(frames,cats):
 fs=[f.copy() for f in frames]
 for c in cats:
  levels=pd.Index(pd.concat([f[c].astype(str) for f in fs]).dropna().unique())
  dtype=pd.CategoricalDtype(categories=levels)
  for f in fs: f[c]=f[c].astype(str).astype(dtype)
 return fs

def fit_reg(tr,va,ev,features,cats,objective="regression_l1"):
 tr,va,ev=align([tr,va,ev],cats)
 params={**PARAMS,"objective":"regression","metric":"l1"}
 if objective=="huber": params.update(objective="huber",alpha=0.9,metric="l1")
 dtr=lgb.Dataset(tr[features],label=tr.target,categorical_feature=cats,free_raw_data=False)
 dva=lgb.Dataset(va[features],label=va.target,categorical_feature=cats,reference=dtr,free_raw_data=False)
 m=lgb.train(params,dtr,num_boost_round=1400,valid_sets=[dva],callbacks=[lgb.early_stopping(90,verbose=False)])
 return np.maximum(m.predict(ev[features],num_iteration=m.best_iteration),0),int(m.best_iteration)

def fit_hurdle(tr,va,ev,features,cats):
 tr,va,ev=align([tr,va,ev],cats)
 # Occurrence classifier plus positive-demand size model; avoids treating zeros as continuous noise.
 ybin=(tr.target.to_numpy()>0).astype(int)
 dtr=lgb.Dataset(tr[features],label=ybin,categorical_feature=cats,free_raw_data=False)
 dva=lgb.Dataset(va[features],label=(va.target.to_numpy()>0).astype(int),categorical_feature=cats,reference=dtr,free_raw_data=False)
 cls=lgb.train({**PARAMS,"objective":"binary","metric":"binary_logloss"},dtr,num_boost_round=900,
   valid_sets=[dva],callbacks=[lgb.early_stopping(60,verbose=False)])
 pos=tr[tr.target>0]
 if len(pos)<100: return np.zeros(len(ev)),int(cls.best_iteration)
 # positive-size regressor
 pos, va2, ev2=align([pos,va,ev],cats)
 dpos=lgb.Dataset(pos[features],label=pos.target,categorical_feature=cats,free_raw_data=False)
 dval=lgb.Dataset(va2[features],label=va2.target,categorical_feature=cats,reference=dpos,free_raw_data=False)
 reg=lgb.train({**PARAMS,"objective":"regression","metric":"l1"},dpos,num_boost_round=1100,
   valid_sets=[dval],callbacks=[lgb.early_stopping(70,verbose=False)])
 prob=cls.predict(ev[features],num_iteration=cls.best_iteration)
 size=np.maximum(reg.predict(ev2[features],num_iteration=reg.best_iteration),0)
 return prob*size,int(max(cls.best_iteration,reg.best_iteration))

def main():
 s3,bucket=s3client(),env("R2_BUCKET")
 with tempfile.TemporaryDirectory(prefix="frn-segment-strategy-") as td:
  root=Path(td); raw={}
  for split,key in RAW.items():
   p=root/f"{split}.parquet"; print(f"Downloading {key}"); s3.download_file(bucket,key,str(p)); raw[split]=pl.read_parquet(p)
  combined=pl.concat([raw["train"],raw["eval"]],how="diagonal_relaxed").sort(GROUP+["dt"])
  # Standard production feature builder; adds only past sales/stock features.
  base=load_features_from_frame(combined)
  prof=segment_features(base)
  # Attach target-date known calendar flags from raw calendar row.
  calcols=[c for c in ["holiday_flag","activity_flag"] if c in combined.columns]
  cal=combined.select([pl.col("dt").str.to_date(strict=False).alias("target_date")]+[pl.col(c) for c in calcols]).unique(subset=["target_date"])
  frame=prof.with_columns(pl.col("date").alias("origin_date"))
  # Flags joined later against target_date, so target date calendar information is explicit.
  raw_eval_dates=set(pd.to_datetime(raw["eval"]["dt"].to_list()).normalize().unique())
  results=[]; pred_rows=[]; segment_rows=[]; audit=[]
  for h in HORIZONS:
   hf=frame.sort(GROUP+["origin_date"]).with_columns([
    pl.col("sale_amount").shift(-h).over(GROUP).alias("target"),
    pl.col("origin_date").shift(-h).over(GROUP).alias("target_date")]).drop_nulls(["target","target_date"])
   # Add future-known holiday/activity flags by target_date; never join target sales or stock data.
   if calcols: hf=hf.join(cal,on="target_date",how="left")
   hf=hf.with_columns([
    pl.col("target_date").dt.weekday().alias("target_day_of_week"),
    pl.col("target_date").dt.day().alias("target_day_of_month"),
    pl.col("target_date").dt.month().alias("target_month"),
    (2*np.pi*pl.col("target_date").dt.weekday()/7).sin().alias("target_dow_sin"),
    (2*np.pi*pl.col("target_date").dt.weekday()/7).cos().alias("target_dow_cos")])
   dates=sorted(pd.to_datetime(hf["target_date"].to_list()).normalize().unique())
   eval_dates=[d for d in dates if d in raw_eval_dates]
   if len(eval_dates)<3: audit.append({"horizon":h,"status":"skipped","reason":"fewer than 3 eval dates"}); continue
   ed=eval_dates[-min(7,len(eval_dates)):]; prior=[d for d in dates if d<ed[0]]
   if len(prior)<37: audit.append({"horizon":h,"status":"skipped","reason":"insufficient chronological history"}); continue
   vd=prior[-7:]; td=prior[:-7]
   pdf=hf.to_pandas(); pdf["target_date"]=pd.to_datetime(pdf.target_date)
   tr=pdf[pdf.target_date.isin(td)].copy(); va=pdf[pdf.target_date.isin(vd)].copy(); ev=pdf[pdf.target_date.isin(ed)].copy()
   excluded={"target","target_date","origin_date","date","sale_amount","dt","hours_sale","hours_stock_status"}
   allfeat=[c for c in pdf.columns if c not in excluded and not c.startswith("hours_")]
   basefeat=[c for c in allfeat if c not in calcols and c not in {"demand_adi_56","demand_cv2_56","demand_segment"}]
   cats=[c for c in ["city_id","store_id","management_group_id","first_category_id","second_category_id","third_category_id","product_id","day_of_week","month","demand_segment","target_day_of_week","target_month"] if c in allfeat]
   # Current LightGBM baseline: same horizon rows/splits, standard features, no target-date calendar flags.
   pbase,itbase=fit_reg(tr,va,ev,basefeat,cats)
   # Calendar enhanced global model: all products, future-known holiday/activity.
   pcal,itcal=fit_reg(tr,va,ev,allfeat,cats)
   model_preds={"current_lightgbm_baseline":pbase,"calendar_enhanced_global":pcal}
   strategy_reports={}
   # Specialist strategies for each segment; intermittent uses hurdle as well as lumpy.
   for seg in ["smooth","erratic","lumpy","intermittent"]:
    tr_s=tr[tr.demand_segment==seg]; va_s=va[va.demand_segment==seg]; ev_s=ev[ev.demand_segment==seg]
    if len(tr_s)<300 or len(va_s)<30 or len(ev_s)==0:
     # Fallback to global calendar model, explicitly audited.
     strategy_reports[seg]={"strategy":"global_calendar_fallback","train_rows":len(tr_s),"validation_rows":len(va_s),"eval_rows":len(ev_s)}
     continue
    if seg in ("lumpy","intermittent"):
     ps,it=fit_hurdle(tr_s,va_s,ev_s,allfeat,cats); strategy="two_stage_occurrence_size"
    elif seg=="erratic":
     ps,it=fit_reg(tr_s,va_s,ev_s,allfeat,cats,objective="huber"); strategy="huber_regression"
    else:
     ps,it=fit_reg(tr_s,va_s,ev_s,allfeat,cats); strategy="segment_lightgbm"
    pcal[ev.index.get_indexer(ev_s.index)]=ps
    strategy_reports[seg]={"strategy":strategy,"train_rows":len(tr_s),"validation_rows":len(va_s),"eval_rows":len(ev_s),"best_iteration":it}
   # pcal now holds routed segment-specialist predictions where enough data existed.
   model_preds["calendar_plus_segment_specialists"]=pcal
   for name,pred in model_preds.items():
    results.append({"horizon_days":h,"model":name,**metrics(ev.target,pred)})
    for i,(_,row) in enumerate(ev.iterrows()):
     pred_rows.append({"horizon_days":h,"target_date":row.target_date.date().isoformat(),"origin_date":row.origin_date.date().isoformat(),
      "store_id":str(row.store_id),"product_id":str(row.product_id),"actual":float(row.target),"model":name,
      "prediction":float(pred[i]),"demand_segment":str(row.demand_segment)})
   for seg,g in ev.assign(prediction=model_preds["calendar_plus_segment_specialists"]).groupby("demand_segment"):
    segment_rows.append({"horizon_days":h,"model":"calendar_plus_segment_specialists","demand_segment":seg,"rows":len(g),**metrics(g.target,g.prediction)})
   audit.append({"horizon":h,"status":"completed","train_rows":len(tr),"validation_rows":len(va),"evaluation_rows":len(ev),
    "train_dates":len(td),"validation_dates":len(vd),"evaluation_dates":len(ed),"evaluation_start":str(ed[0].date()),"evaluation_end":str(ed[-1].date()),
    "calendar_columns_used":calcols,"holiday_activity_source":"raw target-date calendar row; future-known flags only",
    "baseline_features":len(basefeat),"calendar_features":len(allfeat),"segment_strategies":strategy_reports,
    "baseline_best_iteration":itbase,"calendar_global_best_iteration":itcal})
  if not results: raise RuntimeError(f"No horizons evaluated: {audit}")
  report={"experiment":"target-date holiday/activity + segment-specialist forecasting",
   "pooled_model_metrics":results,"segment_metrics":segment_rows,"fold_audit":audit,
   "segment_strategy_design":{"smooth":"segment-specific LightGBM regression","erratic":"Huber-robust LightGBM","lumpy":"two-stage occurrence classifier x positive-size regressor","intermittent":"two-stage occurrence classifier x positive-size regressor"},
   "calendar_leakage_policy":"holiday_flag and activity_flag are joined from the raw target-date calendar row only. They are allowed only if the source calendar flags are known before the forecast is issued; no target sales, hours_sale, current stock or target-day realized weather are joined.",
   "comparison_policy":"Current baseline and enhanced models use identical target rows, horizons, validation dates and evaluation dates. Baseline uses repository standard historical features; enhanced adds target-date calendar flags and routed segment specialists.",
   "caveats":"Only the available evaluation dates are scored. Confirm operational availability timestamps for holiday/activity flags before production use.",
   "generated_at_utc":datetime.now(timezone.utc).isoformat()}
  outputs={"target_calendar_segment_report.json":report,"target_calendar_segment_predictions.csv":pd.DataFrame(pred_rows),
   "target_calendar_segment_metrics.csv":pd.DataFrame(results),"target_calendar_segment_by_demand.csv":pd.DataFrame(segment_rows)}
  for name,obj in outputs.items():
   p=root/name
   if name.endswith(".csv"): obj.to_csv(p,index=False); typ="text/csv"
   else: p.write_text(json.dumps(obj,indent=2,default=str)); typ="application/json"
   key=f"{OUT}/{name}"; s3.upload_file(str(p),bucket,key,ExtraArgs={"ContentType":typ})
   if s3.head_object(Bucket=bucket,Key=key)["ContentLength"]!=p.stat().st_size: raise RuntimeError(f"R2 verify failed {key}")
   print(f"Verified s3://{bucket}/{key} ({p.stat().st_size:,} bytes)")
  print(json.dumps(report,indent=2,default=str))
if __name__=="__main__": main()
