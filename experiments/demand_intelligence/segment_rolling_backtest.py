"""Multi-window rolling backtest with validation-gated demand-segment specialists.

Each fold is strictly chronological: expanding train -> 7-day validation ->
horizon-adaptive test window. For each horizon and segment, a specialist is routed into the test
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
FOLDS=int(os.getenv("ROLLING_FOLDS","5"))
BLOCK_DAYS=7
BOOTSTRAP_REPS=500
BOOTSTRAP_SEED=20261128
MIN_VALIDATION_GAIN_PCT=0.5
MIN_TRAIN_DATES=56
MIN_HURDLE_VALIDATION_POSITIVES=20
CALIBRATION_PRIOR_ROWS=2000
CALIBRATION_MIN_FACTOR=0.80
CALIBRATION_MAX_FACTOR=1.25
# Match src/train_model.py so the benchmark represents the current production baseline.
SEED=42
PARAMS={"objective":"regression","metric":"l1","learning_rate":0.05,
 "num_leaves":63,"max_depth":-1,"min_data_in_leaf":100,
 "feature_fraction":0.85,"bagging_fraction":0.85,"bagging_freq":1,
 "lambda_l2":2.0,"verbosity":-1,"seed":SEED,"num_threads":4}
NUM_BOOST_ROUND=1200
EARLY_STOPPING_ROUNDS=80

def client():
 return boto3.client("s3",endpoint_url=f"https://{env('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
  aws_access_key_id=env("R2_ACCESS_KEY_ID"),aws_secret_access_key=env("R2_SECRET_ACCESS_KEY"),
  region_name="auto",config=Config(signature_version="s3v4",retries={"max_attempts":5,"mode":"standard"}))

def score(y,p):
 y=np.asarray(y,float); p=np.maximum(np.asarray(p,float),0)
 return {"mae":float(np.mean(np.abs(y-p))),"rmse":float(np.sqrt(np.mean((y-p)**2))),
  "wape_pct":float(100*np.abs(y-p).sum()/max(np.abs(y).sum(),1e-9)),
  "bias_pct":float(100*(p-y).sum()/max(y.sum(),1e-9))}

def paired_cluster_bootstrap(frame, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
 """Paired store-product cluster bootstrap for calibrated vs production baseline.

 Resamples entity clusters, retaining all dates/folds for each sampled entity.
 This quantifies cross-sectional uncertainty; it does not replace temporal
 uncertainty across future time windows.
 """
 rng=np.random.default_rng(seed)
 summaries=[]
 for h,g in frame.groupby("horizon_days",sort=True):
  clusters=g[GROUP].astype(str).agg("::".join,axis=1)
  tmp=g.assign(_cluster=clusters)
  # Per-cluster sufficient statistics for both models.
  rows=[]
  for _,c in tmp.groupby("_cluster",sort=False):
   y=c.actual.to_numpy(float)
   b=np.maximum(c.baseline_prediction.to_numpy(float),0)
   p=np.maximum(c.calibrated_prediction.to_numpy(float),0)
   rows.append([
    len(c), np.abs(y-b).sum(), np.abs(y-p).sum(),
    ((y-b)**2).sum(), ((y-p)**2).sum(),
    np.abs(y).sum(), y.sum(), (b-y).sum(), (p-y).sum(),
    b.sum(), p.sum()
   ])
  a=np.asarray(rows,dtype=float)
  n=len(a)
  if n<2: raise RuntimeError(f"h={h}: need at least two store-product clusters")
  # Metrics are computed from the same sampled clusters for a paired comparison.
  draws=rng.integers(0,n,size=(reps,n))
  vals=[]
  for ix in draws:
   z=a[ix].sum(axis=0)
   count=max(z[0],1.0); denom=max(z[5],1e-9)
   base_mae=z[1]/count; cal_mae=z[2]/count
   base_rmse=np.sqrt(z[3]/count); cal_rmse=np.sqrt(z[4]/count)
   base_wape=100*z[1]/denom; cal_wape=100*z[2]/denom
   base_bias=100*z[7]/max(z[6],1e-9); cal_bias=100*z[8]/max(z[6],1e-9)
   vals.append([
    100*(base_mae-cal_mae)/max(base_mae,1e-9),
    100*(base_rmse-cal_rmse)/max(base_rmse,1e-9),
    base_wape-cal_wape,
    cal_bias-base_bias
   ])
  v=np.asarray(vals)
  point=score(g.actual,g.calibrated_prediction)
  base=score(g.actual,g.baseline_prediction)
  names=["mae_improvement_pct","rmse_improvement_pct","wape_improvement_pp","bias_change_pp"]
  point_values=[
   100*(base["mae"]-point["mae"])/max(base["mae"],1e-9),
   100*(base["rmse"]-point["rmse"])/max(base["rmse"],1e-9),
   base["wape_pct"]-point["wape_pct"],
   point["bias_pct"]-base["bias_pct"]
  ]
  for j,name in enumerate(names):
   summaries.append({"horizon_days":int(h),"comparison":"calibrated_vs_current_lightgbm",
    "estimand":name,"point_estimate":float(point_values[j]),
    "ci_95_lower":float(np.quantile(v[:,j],0.025)),
    "ci_95_upper":float(np.quantile(v[:,j],0.975)),
    "bootstrap_replicates":int(reps),"store_product_clusters":int(n)})
 return summaries

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
 model=lgb.train(params,dtr,num_boost_round=NUM_BOOST_ROUND,valid_sets=[dva],
  callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS,verbose=False)])
 return model

def predict_frame(model, frame, features, cats):
 x=frame[features].copy()
 active=[c for c in cats if c in features]
 stored=model.pandas_categorical or []
 if len(stored)!=len(active):
  raise RuntimeError(f"LightGBM categorical metadata mismatch: model={len(stored)} features={len(active)}")
 for col,levels in zip(active,stored):
  values=x[col].astype(str)
  # Categories unseen during training are represented as missing at inference.
  values=values.where(values.isin(levels),np.nan)
  x[col]=pd.Categorical(values,categories=levels)
 return model.predict(x,num_iteration=model.best_iteration)

def fit_predict_specialist_pair(tr,va,ev,features,cats,seg):
 """Fit each specialist once, then return predictions for validation and test."""
 if seg in ("lumpy","intermittent"):
  cls=fit_model(tr,va,features,cats,binary=True)
  pos=tr[tr.target>0]
  pos_va=va[va.target>0]
  if len(pos)<100 or len(pos_va)<MIN_HURDLE_VALIDATION_POSITIVES:
   raise RuntimeError(
    f"Insufficient hurdle-regression examples: train_positive={len(pos)}, "
    f"validation_positive={len(pos_va)}"
   )
  reg=fit_model(pos,pos_va,features,cats)
  val_pred=np.maximum(predict_frame(cls,va,features,cats),0)*np.maximum(predict_frame(reg,va,features,cats),0)
  test_pred=np.maximum(predict_frame(cls,ev,features,cats),0)*np.maximum(predict_frame(reg,ev,features,cats),0)
  return val_pred,test_pred
 model=fit_model(tr,va,features,cats,objective="huber" if seg=="erratic" else "regression")
 return (np.maximum(predict_frame(model,va,features,cats),0),
         np.maximum(predict_frame(model,ev,features,cats),0))

def main():
 s3,bucket=client(),env("R2_BUCKET")
 with tempfile.TemporaryDirectory(prefix="frn-rolling-segments-") as td:
  root=Path(td); raw={}
  for split,key in RAW.items():
   p=root/f"{split}.parquet"; print(f"Downloading {key}")
   s3.download_file(bucket,key,str(p)); raw[split]=pl.read_parquet(p)
  combined=pl.concat([raw["train"],raw["eval"]],how="diagonal_relaxed").sort(GROUP+["dt"])
  frame=add_segments(load_features_from_frame(combined))
  # Keep origin-date calendar fields intact for production-baseline parity.
  # Target-date fields get explicit names so they cannot collide with origin features.
  calcols=[c for c in ["holiday_flag","activity_flag"] if c in combined.columns]
  target_cal_cols=[f"target_{c}" for c in calcols]
  if calcols:
   cal=combined.select(
    [pl.col(c) for c in GROUP]+
    [pl.col("dt").str.to_date(strict=False).alias("target_date")]+
    [pl.col(c).alias(f"target_{c}") for c in calcols]
   ).unique(subset=GROUP+["target_date"])
  audits=[]; fold_rows=[]; selection_rows=[]; prediction_rows=[]; calibration_audit=[]
  for h in HORIZONS:
   hf=frame.sort(GROUP+["date"]).with_columns([
    pl.col("sale_amount").shift(-h).over(GROUP).alias("target"),
    pl.col("date").shift(-h).over(GROUP).alias("target_date")]).drop_nulls(["target","target_date"])
   rows_before_gap_check=hf.height
   # A row shift is only an h-day horizon when the entity's target date is
   # exactly h calendar days ahead. Exclude irregular gaps rather than mislabel them.
   hf=hf.filter((pl.col("target_date")-pl.col("date")).dt.total_days()==h)
   dropped_gap_rows=rows_before_gap_check-hf.height
   print(f"h={h}: excluded {dropped_gap_rows:,} rows with non-{h}-day target gaps")
   if hf.is_empty():
    raise RuntimeError(f"h={h}: no exact {h}-calendar-day target pairs remain")
   if calcols: hf=hf.join(cal,on=GROUP+["target_date"],how="left")
   hf=hf.with_columns([
    pl.col("target_date").dt.weekday().alias("target_day_of_week"),
    pl.col("target_date").dt.day().alias("target_day_of_month"),
    pl.col("target_date").dt.month().alias("target_month"),
    (2*np.pi*pl.col("target_date").dt.weekday()/7).sin().alias("target_dow_sin"),
    (2*np.pi*pl.col("target_date").dt.weekday()/7).cos().alias("target_dow_cos")])
   pdf=hf.to_pandas(); pdf["target_date"]=pd.to_datetime(pdf.target_date)
   dates=sorted(pdf.target_date.dt.normalize().unique())
   # Reserve the latest three disjoint test windows; each has its own
   # immediately preceding 7-day validation block and expanding historical train.
   # Keep three disjoint test folds, shrinking each test block when the
   # horizon shift leaves fewer dates. Training and validation lengths stay fixed.
   required_prior=MIN_TRAIN_DATES+BLOCK_DAYS
   max_test_days=(len(dates)-required_prior)//FOLDS
   effective_block_days=min(BLOCK_DAYS,max_test_days)
   if effective_block_days<1:
    raise RuntimeError(
     f"h={h}: insufficient dates ({len(dates)}) for "
     f"{MIN_TRAIN_DATES} train + {BLOCK_DAYS} validation + "
     f"{FOLDS} non-empty test folds"
    )
   total_test_days=FOLDS*effective_block_days
   block_start=len(dates)-total_test_days
   test_blocks=[
    dates[block_start+i*effective_block_days:block_start+(i+1)*effective_block_days]
    for i in range(FOLDS)
   ]
   print(f"h={h}: {len(dates)} target dates; {FOLDS} test folds x "
         f"{effective_block_days} days; min train={MIN_TRAIN_DATES}, validation={BLOCK_DAYS}")
   selection_history={seg:[] for seg in ["smooth","erratic","lumpy","intermittent"]}
   for fi,ed in enumerate(test_blocks,1):
    if len(ed)!=effective_block_days: raise RuntimeError(f"h={h} fold={fi}: expected {effective_block_days} test dates, got {len(ed)}")
    first=ed[0]; prior=[d for d in dates if d<first]
    if len(prior)<MIN_TRAIN_DATES+BLOCK_DAYS: raise RuntimeError(f"h={h} fold={fi}: insufficient pre-test history ({len(prior)} dates)")
    vd=prior[-BLOCK_DAYS:]; td=prior[:-BLOCK_DAYS]
    tr=pdf[pdf.target_date.dt.normalize().isin(td)].copy()
    va=pdf[pdf.target_date.dt.normalize().isin(vd)].copy()
    ev=pdf[pdf.target_date.dt.normalize().isin(ed)].copy()
    # Derive production features from the exact feature-loader output. Remove
    # only the three demand-segmentation fields added by this experiment.
    segment_feature_cols=["demand_adi_56","demand_cv2_56","demand_segment"]
    basefeat=[c for c in frame.columns
              if c not in {"date","sale_amount",*segment_feature_cols}]
    target_date_feature_cols=["target_day_of_week","target_day_of_month",
                              "target_month","target_dow_sin","target_dow_cos"]
    allfeat=basefeat+segment_feature_cols+target_cal_cols+target_date_feature_cols
    missing_features=set(allfeat)-set(pdf.columns)
    if missing_features:
     raise RuntimeError(f"h={h}: feature columns missing after target/calendar joins: {sorted(missing_features)}")
    cats=[c for c in ["city_id","store_id","management_group_id","first_category_id","second_category_id","third_category_id","product_id","day_of_week","month","demand_segment","target_day_of_week","target_month"] if c in allfeat]
    baseline=fit_model(tr,va,basefeat,cats)
    pb=np.maximum(predict_frame(baseline,ev,basefeat,cats),0)
    # Fit each specialist once per fold, evaluate on validation for routing, then
    # apply the already-decided route to the untouched test block.
    routed=pb.copy()
    pval_base=np.maximum(predict_frame(baseline,va,basefeat,cats),0)
    val_routed=pval_base.copy()
    specialist_test_predictions={}
    specialist_val_predictions={}
    current_fold_selected={}
    for seg in ["smooth","erratic","lumpy","intermittent"]:
     tr_s=tr[tr.demand_segment==seg]; va_s=va[va.demand_segment==seg]; ev_s=ev[ev.demand_segment==seg]
     if len(tr_s)<300 or len(va_s)<30 or len(ev_s)==0:
      selection_rows.append({"horizon_days":h,"fold":fi,"segment":seg,"selected":False,"reason":"insufficient segment sample"})
      current_fold_selected[seg]=False
      continue
     if seg in ("lumpy","intermittent") and int((va_s.target>0).sum())<MIN_HURDLE_VALIDATION_POSITIVES:
      selection_rows.append({"horizon_days":h,"fold":fi,"segment":seg,"selected":False,
       "reason":"insufficient positive validation rows for hurdle regressor",
       "validation_positive_rows":int((va_s.target>0).sum())})
      current_fold_selected[seg]=False
      continue
     # Baseline segment score and challenger validation score are calculated on the same rows.
     bval=np.maximum(predict_frame(baseline,va_s,basefeat,cats),0)
     sp_val,sp_test=fit_predict_specialist_pair(tr_s,va_s,ev_s,allfeat,cats,seg)
     bmae=score(va_s.target,bval)["mae"]; smae=score(va_s.target,sp_val)["mae"]
     gain=100*(bmae-smae)/max(bmae,1e-9)
     chosen=gain>=MIN_VALIDATION_GAIN_PCT
     reason="validation_mae_gain_meets_threshold" if chosen else "validation_gain_below_threshold"
     selection_rows.append({"horizon_days":h,"fold":fi,"segment":seg,"selected":bool(chosen),
       "validation_baseline_mae":bmae,"validation_specialist_mae":smae,"validation_gain_pct":gain,"reason":reason})
     current_fold_selected[seg]=bool(chosen)
     if chosen:
      specialist_val_predictions[seg]=sp_val
      specialist_test_predictions[seg]=sp_test
    # Prospective gate: two completed validation wins among the latest three.
    # No test outcome is used to enable, disable, or select a specialist.
    for seg,selected_now in current_fold_selected.items():
     history=selection_history[seg]
     history.append(bool(selected_now))
     gate_open=len(history)>=2 and sum(history[-3:])>=2
     if gate_open and seg in specialist_test_predictions:
      sp_test=specialist_test_predictions[seg]
      test_mask=ev.demand_segment.eq(seg).to_numpy()
      val_mask=va.demand_segment.eq(seg).to_numpy()
      if int(test_mask.sum())!=len(sp_test): raise RuntimeError("specialist test routing row mismatch")
      if int(val_mask.sum())!=len(specialist_val_predictions[seg]): raise RuntimeError("specialist validation routing row mismatch")
      routed[test_mask]=sp_test
      val_routed[val_mask]=specialist_val_predictions[seg]
    # Fit a segment-wise multiplicative correction on this fold's validation
    # predictions only. Shrink small segments toward 1 and clip extreme factors.
    calibrated=routed.copy()
    for seg in ["smooth","erratic","lumpy","intermittent"]:
     vm=va.demand_segment.eq(seg).to_numpy()
     tm=ev.demand_segment.eq(seg).to_numpy()
     n=int(vm.sum())
     if n:
      actual_sum=float(va.loc[vm,"target"].sum())
      pred_sum=float(val_routed[vm].sum())
      raw_factor=actual_sum/max(pred_sum,1e-9)
      shrink=n/(n+CALIBRATION_PRIOR_ROWS)
      factor=float(np.clip(1+(raw_factor-1)*shrink,CALIBRATION_MIN_FACTOR,CALIBRATION_MAX_FACTOR))
     else:
      actual_sum=0.0; pred_sum=0.0; raw_factor=1.0; shrink=0.0; factor=1.0
     calibrated[tm]=np.maximum(routed[tm]*factor,0)
     calibration_audit.append({"horizon_days":h,"fold":fi,"segment":seg,"validation_rows":n,
      "validation_actual_sum":actual_sum,"validation_prediction_sum":pred_sum,
      "raw_factor":raw_factor,"shrinkage_weight":shrink,"applied_factor":factor})
    fold_rows.extend([
     {"horizon_days":h,"fold":fi,"model":"current_lightgbm_baseline",**score(ev.target,pb)},
     {"horizon_days":h,"fold":fi,"model":"validation_gated_segment_specialists",**score(ev.target,routed)},
     {"horizon_days":h,"fold":fi,"model":"validation_gated_segment_specialists_calibrated",**score(ev.target,calibrated)}])
    for idx,(_,r) in enumerate(ev.iterrows()):
     prediction_rows.append({"horizon_days":h,"fold":fi,"target_date":r.target_date.date().isoformat(),
      "store_id":str(r.store_id),"product_id":str(r.product_id),"segment":str(r.demand_segment),
      "actual":float(r.target),"baseline_prediction":float(pb[idx]),"selected_prediction":float(routed[idx]),
      "calibrated_prediction":float(calibrated[idx])})
    audits.append({"horizon_days":h,"fold":fi,"train_dates":len(td),"validation_dates":len(vd),"test_dates":len(ed),
      "train_start":str(min(td).date()),"train_end":str(max(td).date()),
      "validation_start":str(min(vd).date()),"validation_end":str(max(vd).date()),
      "test_start":str(min(ed).date()),"test_end":str(max(ed).date()),
      "train_rows":len(tr),"validation_rows":len(va),"test_rows":len(ev)})
  fdf=pd.DataFrame(fold_rows); sdf=pd.DataFrame(selection_rows); pdf=pd.DataFrame(prediction_rows)
  # Refuse to publish partial or malformed prediction outputs.  The CSV stores
  # one row per entity/date with both model predictions, so group by horizon only.
  if pdf.empty:
   raise RuntimeError("No test predictions were generated; refusing to publish empty artifacts")
  expected_horizons=set(HORIZONS)
  observed_horizons=set(pdf["horizon_days"].astype(int).unique())
  if observed_horizons!=expected_horizons:
   raise RuntimeError(f"Prediction horizon mismatch: expected={sorted(expected_horizons)}, observed={sorted(observed_horizons)}")
  values=pdf[["actual","baseline_prediction","selected_prediction","calibrated_prediction"]].to_numpy(dtype=float)
  if not np.isfinite(values).all():
   raise RuntimeError("Non-finite actuals or predictions found; refusing to publish metrics")
  # Confidence intervals are calculated in a separate post-hoc job from this saved artifact.\n  uncertainty=[]
  pooled=[]
  for h,g in pdf.groupby("horizon_days",sort=True):
   observed_folds=set(g["fold"].astype(int).unique())
   expected_folds=set(range(1,FOLDS+1))
   if observed_folds!=expected_folds:
    raise RuntimeError(f"h={h}: incomplete test folds; expected={sorted(expected_folds)}, observed={sorted(observed_folds)}")
   for name,col in [("current_lightgbm_baseline","baseline_prediction"),("validation_gated_segment_specialists","selected_prediction"),("validation_gated_segment_specialists_calibrated","calibrated_prediction")]:
    pooled.append({"horizon_days":int(h),"model":name,"rows":int(len(g)),
                   "test_folds":int(g["fold"].nunique()),**score(g.actual,g[col])})
  stability=[]
  for (h,seg),g in sdf.groupby(["horizon_days","segment"],sort=True):
   wins=int(g.selected.sum()); folds=int(len(g))
   stable=wins>=max(2,int(np.ceil(2*folds/3)))
   ineligible=int(g.reason.astype(str).str.startswith("insufficient").sum())
   stability.append({"horizon_days":int(h),"segment":seg,"validation_wins":wins,"validation_folds":folds,
    "ineligible_folds":ineligible,"stable_specialist":bool(stable),
    "selection_frequency_pct":100*wins/max(folds,1)})
  segment_test_metrics=[]
  for (h,seg),g in pdf.groupby(["horizon_days","segment"],sort=True):
   base=score(g.actual,g.baseline_prediction)
   selected=score(g.actual,g.selected_prediction)
   calibrated_score=score(g.actual,g.calibrated_prediction)
   picked=sdf[(sdf.horizon_days==h)&(sdf.segment==seg)]
   segment_test_metrics.append({
    "horizon_days":int(h),"segment":str(seg),"rows":int(len(g)),
    "test_folds":int(g.fold.nunique()),"specialist_selected_folds":int(picked.selected.sum()),
    "baseline_mae":base["mae"],"selected_mae":selected["mae"],"calibrated_mae":calibrated_score["mae"],
    "calibrated_bias_pct":calibrated_score["bias_pct"],
    "mae_improvement_pct":100*(base["mae"]-selected["mae"])/max(base["mae"],1e-9),
    "calibrated_mae_improvement_pct":100*(base["mae"]-calibrated_score["mae"])/max(base["mae"],1e-9),
    "baseline_rmse":base["rmse"],"selected_rmse":selected["rmse"],
    "baseline_wape_pct":base["wape_pct"],"selected_wape_pct":selected["wape_pct"]
   })
  report={"experiment":"multi-window rolling backtest with validation-gated segment specialists",
   "fold_design":{"folds":FOLDS,"configured_test_block_days":BLOCK_DAYS,"validation_block_days":BLOCK_DAYS,
    "training":"expanding chronological history","test_windows_disjoint":True,"effective_test_block_days_by_horizon":{str(h):next(a["test_dates"] for a in audits if a["horizon_days"]==h and a["fold"]==1) for h in HORIZONS}},
   "selection_policy":{"metric":"segment validation MAE","minimum_gain_pct":MIN_VALIDATION_GAIN_PCT,
    "stability_rule":"prospective gate: at least two completed validation windows and at least two selections in the most recent three; applies only to the subsequent test block",
    "test_isolation":"test blocks never authorize specialists; only current and prior validation decisions control routing"},
   "pooled_test_metrics":pooled,"fold_test_metrics":fold_rows,"calibration_diagnostics":calibration_audit,
   "paired_cluster_bootstrap_uncertainty":uncertainty,
   "segment_test_metrics":segment_test_metrics,"selection_diagnostics":stability,
   "fold_audit":audits,"selection_details":selection_rows,
   "segment_strategy":{"smooth":"LightGBM regression","erratic":"Huber LightGBM","lumpy":"occurrence classifier x positive-size regressor","intermittent":"occurrence classifier x positive-size regressor"},
   "calendar_features":target_cal_cols,"calendar_policy":"Only explicitly named target-date calendar flags are specialist-only; original origin-date flags remain in the production baseline. Verify target flags are available before forecast time.",
   "calibration_policy":{"method":"segment-wise multiplicative ratio calibration","fit_data":"current fold validation block only, using the validation-routed model","shrinkage_prior_rows":CALIBRATION_PRIOR_ROWS,"factor_clip":[CALIBRATION_MIN_FACTOR,CALIBRATION_MAX_FACTOR],"test_isolation":"test targets are never used to estimate correction factors"},
   "caveats":"Calibration factors are estimated from each fold's validation predictions and shrunk toward 1. They are not fitted on test outcomes. Compare calibrated MAE, RMSE, WAPE and bias against the uncalibrated gated challenger; calibration may reduce bias while worsening MAE. Confirm on a later live period before production promotion.",
   "uncertainty_methodology":"Computed by the separate post-hoc paired cluster bootstrap workflow after predictions are saved; no intervals are computed in the model-training job.",
   "generated_at_utc":datetime.now(timezone.utc).isoformat()}
  # A compact paired comparison table makes the uncertainty and evaluation
  # sample sizes visible without requiring consumers to parse the full JSON.
  uncertainty_lookup={(int(x["horizon_days"]),x["estimand"]):x for x in uncertainty}
  comparison_rows=[]
  for h,g in pdf.groupby("horizon_days",sort=True):
   base=score(g.actual,g.baseline_prediction)
   challenger=score(g.actual,g.calibrated_prediction)
   n_clusters=int(g[GROUP].astype(str).agg("::".join,axis=1).nunique())
   metrics=[
    ("mae_improvement_pct",100*(base["mae"]-challenger["mae"])/max(base["mae"],1e-9)),
    ("rmse_improvement_pct",100*(base["rmse"]-challenger["rmse"])/max(base["rmse"],1e-9)),
    ("wape_improvement_pp",base["wape_pct"]-challenger["wape_pct"]),
    ("bias_change_pp",challenger["bias_pct"]-base["bias_pct"]),
   ]
   for metric_name,point in metrics:
    ci=uncertainty_lookup.get((int(h),metric_name))
    comparison_rows.append({"horizon_days":int(h),"comparison":"calibrated_specialists_vs_current_lightgbm",
     "metric":metric_name,"baseline_value":base[{"mae_improvement_pct":"mae","rmse_improvement_pct":"rmse",
       "wape_improvement_pp":"wape_pct","bias_change_pp":"bias_pct"}[metric_name]],
     "challenger_value":challenger[{"mae_improvement_pct":"mae","rmse_improvement_pct":"rmse",
       "wape_improvement_pp":"wape_pct","bias_change_pp":"bias_pct"}[metric_name]],
     "improvement_point_estimate":point,"ci_95_lower":ci["ci_95_lower"] if ci else None,
     "ci_95_upper":ci["ci_95_upper"] if ci else None,"evaluation_rows":int(len(g)),
     "store_product_clusters":n_clusters,"test_folds":int(g.fold.nunique()),
     "bootstrap_replicates":int(ci["bootstrap_replicates"]) if ci else 0,
     "uncertainty_scope":"pending post-hoc paired cluster bootstrap; no CI included in training job"})
  comparison_df=pd.DataFrame(comparison_rows)
  comparison_md=["# Rolling-backtest experiment comparison","",
   "Calibrated validation-gated segment specialists versus the current LightGBM baseline.",
   "Confidence intervals are added by the separate post-hoc bootstrap workflow; this file contains point estimates only until that step succeeds.","",
   "| Horizon | Metric | Baseline | Challenger | Improvement | 95% CI | Eval rows | Clusters | Folds |",
   "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
  for row in comparison_rows:
   comparison_md.append(f"| {row['horizon_days']}d | {row['metric']} | {row['baseline_value']:.4f} | {row['challenger_value']:.4f} | {row['improvement_point_estimate']:.4f} | [{row['ci_95_lower']:.4f}, {row['ci_95_upper']:.4f}] | {row['evaluation_rows']:,} | {row['store_product_clusters']:,} | {row['test_folds']} |")
  comparison_md += ["","**Interpretation:** positive MAE/RMSE/WAPE improvement favours the challenger; bias change is a signed percentage-point difference. Treat this as a research challenger until temporal validation is expanded."]
  report["comparison_sample_sizes"]=[{"horizon_days":int(h),"evaluation_rows":int(len(g)),
   "store_product_clusters":int(g[GROUP].astype(str).agg("::".join,axis=1).nunique()),
   "test_folds":int(g.fold.nunique()),"target_dates":int(g.target_date.nunique())}
   for h,g in pdf.groupby("horizon_days",sort=True)]
  report["comparison_report"]="segment_rolling_backtest_comparison.csv"
  outputs={"segment_rolling_backtest_report.json":report,
   "segment_rolling_backtest_comparison.csv":comparison_df,
   "segment_rolling_backtest_comparison.md":"\n".join(comparison_md)+"\n",
   "segment_rolling_backtest_folds.csv":fdf,
   "segment_rolling_backtest_selection.csv":sdf,"segment_rolling_backtest_predictions.csv":pdf,
   "segment_rolling_backtest_calibration.csv":pd.DataFrame(calibration_audit)}
  for name,obj in outputs.items():
   path=root/name
   if name.endswith(".csv"): obj.to_csv(path,index=False); ctype="text/csv"
   else: path.write_text(json.dumps(obj,indent=2,default=str)); ctype="application/json"
   key=f"{OUT}/{name}"; s3.upload_file(str(path),bucket,key,ExtraArgs={"ContentType":ctype})
   if s3.head_object(Bucket=bucket,Key=key)["ContentLength"]!=path.stat().st_size: raise RuntimeError(f"R2 verification failed: {key}")
   print(f"Verified s3://{bucket}/{key} ({path.stat().st_size:,} bytes)")
  print(json.dumps(report,indent=2,default=str))

if __name__=="__main__": main()
