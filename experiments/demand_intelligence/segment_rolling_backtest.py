"""Multi-window rolling backtest with validation-gated demand-segment specialists.

Each fold is strictly chronological: expanding train -> 7-day validation ->
7-day test. For each horizon and segment, a specialist is routed into the test
predictions only when its validation MAE beats the production LightGBM baseline
by the configured margin. Test outcomes never influence routing decisions.
"""
from __future__ import annotations
import json, sys, tempfile
from pathlib import Path
from datetime import datetime, timezone
import boto3, lightgbm as lgb, numpy as np, pandas as pd, polars as pl
from botocore.config import Config

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from src.train_model import RAW, env, load_features_from_frame

OUT="segment-rolling-backtest-artifacts"
GROUP=["store_id","product_id"]
HORIZONS=(1,7,14)
FOLDS=3
BLOCK_DAYS=7
MIN_VALIDATION_GAIN_PCT=0.5
MIN_TRAIN_DATES=56
SEED=20261202
PARAMS={"learning_rate":0.04,"num_leaves":63,"min_data_in_leaf":150,
 "feature_fraction":0.85,"bagging_fraction":0.85,"bagging_freq":1,
 "lambda_l2":5.0,"verbosity":-1,"seed":SEED,"num_threads":4}

def client():
 return boto3.client("s3",endpoint_url=f"https://{env('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
  aws_access_key_id=env("R2_ACCESS_KEY_ID"),aws_secret_access_key=env("R2_SECRET_ACCESS_KEY"),
  region_name="auto",config=Config(signature_version="s3v4",retries={"max_attempts":5,"mode":"standard"}))

def score(y,p):
 y=np.asarray(y,float); p=np.maximum(np.asarray(p,float),0)
 return {"mae":float(np.mean(np.abs(y-p))),"rmse":float(np.sqrt(np.mean((y-p)**2))),
  "wape_pct":float(100*np.abs(y-p).sum()/max(np.abs(y).sum(),1e-9)),
  "bias_pct":float(100*(p-y).sum()/max(y.sum(),1e-9))}

def add_segments(frame):
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

def fit_model(tr,va,features,cats,objective="regression_l1",binary=False):
 cats=[c for c in cats if c in features]
 tr,va=align([tr,va],cats)
 params={**PARAMS}
 if binary: params.update(objective="binary",metric="binary_logloss")
 else:
  params.update(objective="huber" if objective=="huber" else "regression",
                metric="l1")
  if objective=="huber": params["alpha"]=0.9
 label=(tr.target.to_numpy()>0).astype(int) if binary else tr.target
 vlabel=(va.target.to_numpy()>0).astype(int) if binary else va.target
 dtr=lgb.Dataset(tr[features],label=label,categorical_feature=cats,free_raw_data=False)
 dva=lgb.Dataset(va[features],label=vlabel,categorical_feature=cats,reference=dtr,free_raw_data=False)
 model=lgb.train(params,dtr,num_boost_round=1100,valid_sets=[dva],
  callbacks=[lgb.early_stopping(70,verbose=False)])
 return model

def predict_frame(model, frame, features, cats):
 x=frame[features].copy()
 active=[c for c in cats if c in features]
 stored=model.pandas_categorical or []
 if len(stored)!=len(active):
  raise RuntimeError(f"LightGBM categorical metadata mismatch: model={len(stored)} features={len(active)}")
 for col,levels in zip(active,stored):
  x[col]=pd.Categorical(x[col].astype(str),categories=levels)
 return model.predict(x,num_iteration=model.best_iteration)

def predict_specialist(tr,va,ev,features,cats,seg):
 if seg in ("lumpy","intermittent"):
  cls=fit_model(tr,va,features,cats,binary=True)
  pos=tr[tr.target>0]
  if len(pos)<100: return np.zeros(len(ev))
  reg=fit_model(pos,va[va.target>0] if (va.target>0).sum()>=20 else pos,features,cats)
  return np.maximum(predict_frame(cls,ev,features,cats),0)*np.maximum(predict_frame(reg,ev,features,cats),0)
 model=fit_model(tr,va,features,cats,objective="huber" if seg=="erratic" else "regression")
 return np.maximum(predict_frame(model,ev,features,cats),0)

def main():
 s3,bucket=client(),env("R2_BUCKET")
 with tempfile.TemporaryDirectory(prefix="frn-rolling-segments-") as td:
  root=Path(td); raw={}
  for split,key in RAW.items():
   p=root/f"{split}.parquet"; print(f"Downloading {key}")
   s3.download_file(bucket,key,str(p)); raw[split]=pl.read_parquet(p)
  combined=pl.concat([raw["train"],raw["eval"]],how="diagonal_relaxed").sort(GROUP+["dt"])
  frame=add_segments(load_features_from_frame(combined))
  calcols=[c for c in ["holiday_flag","activity_flag"] if c in combined.columns]
  if calcols:
   cal=combined.select([pl.col("dt").str.to_date(strict=False).alias("target_date")]+[pl.col(c) for c in calcols]).unique(subset=["target_date"])
  audits=[]; fold_rows=[]; selection_rows=[]; prediction_rows=[]
  for h in HORIZONS:
   hf=frame.sort(GROUP+["date"]).with_columns([
    pl.col("sale_amount").shift(-h).over(GROUP).alias("target"),
    pl.col("date").shift(-h).over(GROUP).alias("target_date")]).drop_nulls(["target","target_date"])
   if calcols: hf=hf.join(cal,on="target_date",how="left")
   hf=hf.with_columns([
    pl.col("target_date").dt.weekday().alias("target_day_of_week"),
    pl.col("target_date").dt.day().alias("target_day_of_month"),
    pl.col("target_date").dt.month().alias("target_month"),
    (2*np.pi*pl.col("target_date").dt.weekday()/7).sin().alias("target_dow_sin"),
    (2*np.pi*pl.col("target_date").dt.weekday()/7).cos().alias("target_dow_cos")])
   pdf=hf.to_pandas(); pdf["target_date"]=pd.to_datetime(pdf.target_date)
   dates=sorted(pdf.target_date.dt.normalize().unique())
   # Reserve the latest 3 disjoint 7-day blocks as test windows; each has its own
   # immediately preceding 7-day validation block and expanding historical train.
   test_blocks=[dates[-(FOLDS-i)*BLOCK_DAYS:-(FOLDS-i-1)*BLOCK_DAYS if (FOLDS-i-1)>0 else None] for i in range(FOLDS)]
   for fi,ed in enumerate(test_blocks,1):
    if len(ed)!=BLOCK_DAYS: raise RuntimeError(f"h={h} fold={fi}: expected {BLOCK_DAYS} test dates, got {len(ed)}")
    first=ed[0]; prior=[d for d in dates if d<first]
    if len(prior)<MIN_TRAIN_DATES+BLOCK_DAYS: raise RuntimeError(f"h={h} fold={fi}: insufficient pre-test history ({len(prior)} dates)")
    vd=prior[-BLOCK_DAYS:]; td=prior[:-BLOCK_DAYS]
    tr=pdf[pdf.target_date.dt.normalize().isin(td)].copy()
    va=pdf[pdf.target_date.dt.normalize().isin(vd)].copy()
    ev=pdf[pdf.target_date.dt.normalize().isin(ed)].copy()
    excluded={"target","target_date","date","sale_amount","dt","hours_sale","hours_stock_status","origin_date"}
    allfeat=[c for c in pdf.columns if c not in excluded and not c.startswith("hours_")]
    exp_only=set(calcols)|{"demand_adi_56","demand_cv2_56","demand_segment","target_day_of_week","target_day_of_month","target_month","target_dow_sin","target_dow_cos"}
    basefeat=[c for c in allfeat if c not in exp_only]
    cats=[c for c in ["city_id","store_id","management_group_id","first_category_id","second_category_id","third_category_id","product_id","day_of_week","month","demand_segment","target_day_of_week","target_month"] if c in allfeat]
    baseline=fit_model(tr,va,basefeat,cats)
    pb=np.maximum(predict_frame(baseline,ev,basefeat,cats),0)
    # Fit each specialist once per fold, evaluate on validation for routing, then
    # apply the already-decided route to the untouched test block.
    routed=pb.copy()
    for seg in ["smooth","erratic","lumpy","intermittent"]:
     tr_s=tr[tr.demand_segment==seg]; va_s=va[va.demand_segment==seg]; ev_s=ev[ev.demand_segment==seg]
     if len(tr_s)<300 or len(va_s)<30 or len(ev_s)==0:
      selection_rows.append({"horizon_days":h,"fold":fi,"segment":seg,"selected":False,"reason":"insufficient segment sample"})
      continue
     # Baseline segment score and challenger validation score are calculated on the same rows.
     bval=np.maximum(predict_frame(baseline,va_s,basefeat,cats),0)
     sp_val=predict_specialist(tr_s,va_s,va_s,allfeat,cats,seg)
     bmae=score(va_s.target,bval)["mae"]; smae=score(va_s.target,sp_val)["mae"]
     gain=100*(bmae-smae)/max(bmae,1e-9)
     chosen=gain>=MIN_VALIDATION_GAIN_PCT
     reason="validation_mae_gain_meets_threshold" if chosen else "validation_gain_below_threshold"
     selection_rows.append({"horizon_days":h,"fold":fi,"segment":seg,"selected":bool(chosen),
       "validation_baseline_mae":bmae,"validation_specialist_mae":smae,"validation_gain_pct":gain,"reason":reason})
     if chosen:
      sp_test=predict_specialist(tr_s,va_s,ev_s,allfeat,cats,seg)
      mask=ev.demand_segment.eq(seg).to_numpy()
      if int(mask.sum())!=len(sp_test): raise RuntimeError("specialist routing row mismatch")
      routed[mask]=sp_test
    fold_rows.extend([
     {"horizon_days":h,"fold":fi,"model":"current_lightgbm_baseline",**score(ev.target,pb)},
     {"horizon_days":h,"fold":fi,"model":"validation_gated_segment_specialists",**score(ev.target,routed)}])
    for idx,(_,r) in enumerate(ev.iterrows()):
     prediction_rows.append({"horizon_days":h,"fold":fi,"target_date":r.target_date.date().isoformat(),
      "store_id":str(r.store_id),"product_id":str(r.product_id),"segment":str(r.demand_segment),
      "actual":float(r.target),"baseline_prediction":float(pb[idx]),"selected_prediction":float(routed[idx])})
    audits.append({"horizon_days":h,"fold":fi,"train_dates":len(td),"validation_dates":len(vd),"test_dates":len(ed),
      "train_start":str(min(td).date()),"train_end":str(max(td).date()),
      "validation_start":str(min(vd).date()),"validation_end":str(max(vd).date()),
      "test_start":str(min(ed).date()),"test_end":str(max(ed).date()),
      "train_rows":len(tr),"validation_rows":len(va),"test_rows":len(ev)})
  fdf=pd.DataFrame(fold_rows); sdf=pd.DataFrame(selection_rows); pdf=pd.DataFrame(prediction_rows)
  pooled=[]
  for (h,model),g in pdf.groupby("horizon_days"):
   for name,col in [("current_lightgbm_baseline","baseline_prediction"),("validation_gated_segment_specialists","selected_prediction")]:
    pooled.append({"horizon_days":int(h),"model":name,**score(g.actual,g[col])})
  stability=[]
  for (h,seg),g in sdf.groupby(["horizon_days","segment"]):
   wins=int(g.selected.sum()); folds=int(len(g))
   stable=wins>=max(2,int(np.ceil(2*folds/3)))
   stability.append({"horizon_days":int(h),"segment":seg,"validation_wins":wins,"eligible_folds":folds,
    "stable_specialist":bool(stable),"selection_frequency_pct":100*wins/max(folds,1)})
  report={"experiment":"multi-window rolling backtest with validation-gated segment specialists",
   "fold_design":{"folds":FOLDS,"test_block_days":BLOCK_DAYS,"validation_block_days":BLOCK_DAYS,
    "training":"expanding chronological history","test_windows_disjoint":True},
   "selection_policy":{"metric":"segment validation MAE","minimum_gain_pct":MIN_VALIDATION_GAIN_PCT,
    "stability_rule":"recommend a segment specialist only if validation selects it in at least 2 of 3 folds",
    "test_isolation":"test blocks are never used to select a specialist; only validation scores control routing"},
   "pooled_test_metrics":pooled,"fold_test_metrics":fold_rows,"selection_diagnostics":stability,
   "fold_audit":audits,"selection_details":selection_rows,
   "segment_strategy":{"smooth":"LightGBM regression","erratic":"Huber LightGBM","lumpy":"occurrence classifier x positive-size regressor","intermittent":"occurrence classifier x positive-size regressor"},
   "calendar_features":calcols,"calendar_policy":"Only target-date calendar flags from raw calendar rows are used by specialists; verify operational availability before forecast time.",
   "caveats":"Backtest uses the latest three disjoint 7-day windows available in the combined historical dataset. Confirm stability on a later live period before production promotion.",
   "generated_at_utc":datetime.now(timezone.utc).isoformat()}
  outputs={"segment_rolling_backtest_report.json":report,"segment_rolling_backtest_folds.csv":fdf,
   "segment_rolling_backtest_selection.csv":sdf,"segment_rolling_backtest_predictions.csv":pdf}
  for name,obj in outputs.items():
   path=root/name
   if name.endswith(".csv"): obj.to_csv(path,index=False); ctype="text/csv"
   else: path.write_text(json.dumps(obj,indent=2,default=str)); ctype="application/json"
   key=f"{OUT}/{name}"; s3.upload_file(str(path),bucket,key,ExtraArgs={"ContentType":ctype})
   if s3.head_object(Bucket=bucket,Key=key)["ContentLength"]!=path.stat().st_size: raise RuntimeError(f"R2 verification failed: {key}")
   print(f"Verified s3://{bucket}/{key} ({path.stat().st_size:,} bytes)")
  print(json.dumps(report,indent=2,default=str))

if __name__=="__main__": main()
