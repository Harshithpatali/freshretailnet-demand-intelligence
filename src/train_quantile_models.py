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
        pl.col("sale_amount").shift(1).rolling_std(window_size=7,min_samples=2).over(grp).fill_null(0).alias("sales_roll_std_7"),
        pl.col("stock_hour6_22_cnt").shift(1).over(grp).alias("stockout_hours_lag_1"),
        pl.col("stock_hour6_22_cnt").shift(1).rolling_max(window_size=7,min_samples=1).over(grp).alias("stockout_hours_roll_max_7"),
    ])
    excluded={"dt","date","sale_amount","hours_sale","hours_stock_status","hourly_sales_sum",
      "hourly_sales_mean","hourly_sales_peak","hourly_slots","stock_status_sum","stock_status_slots","stock_hour6_22_cnt"}
    feats=[c for c in df.columns if c not in excluded]
    select_cols=list(dict.fromkeys(["date","store_id","product_id","sale_amount","stock_hour6_22_cnt"]+feats))
    return df.drop_nulls(subset=["sales_lag_1","sales_lag_7"]).select(select_cols)
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

EXPOSURE_GROUPS=("0_hours","1_2_hours","3_5_hours","6_9_hours","10_15_hours","16_hours")


def exposure_groups(values):
    """Bucket a prior-period exposure feature that is known at forecast issue time."""
    return pd.cut(
        pd.Series(np.asarray(values, dtype=float)),
        bins=[-np.inf, 0, 2, 5, 9, 15, np.inf],
        labels=list(EXPOSURE_GROUPS),
        include_lowest=True,
    ).astype(str).to_numpy()


def conditional_conformal_adjustments(y, lower, upper, context_exposure,
                                      global_qhat, coverage=.80,
                                      min_group_rows=1000,
                                      prior_strength_rows=2000):
    """Estimate group adjustments on validation residuals using only lagged exposure context."""
    y=np.asarray(y, dtype=float)
    lower=np.asarray(lower, dtype=float)
    upper=np.asarray(upper, dtype=float)
    groups=exposure_groups(context_exposure)
    adjustments={}
    audit=[]
    for label in EXPOSURE_GROUPS:
        mask=groups==label
        n=int(mask.sum())
        if n < min_group_rows:
            group_qhat=float(global_qhat)
            weight=0.0
            applied=float(global_qhat)
            method="global_fallback_insufficient_validation_rows"
        else:
            group_qhat=conformal_adjustment(y[mask],lower[mask],upper[mask],coverage)
            weight=float(n/(n+prior_strength_rows))
            applied=float(weight*group_qhat+(1.0-weight)*global_qhat)
            method="validation_group_quantile_shrunk_to_global"
        adjustments[label]=applied
        audit.append({
            "prior_exposure_group":label,
            "validation_rows":n,
            "group_adjustment":float(group_qhat),
            "global_adjustment":float(global_qhat),
            "shrinkage_weight":weight,
            "applied_adjustment":applied,
            "method":method,
        })
    return adjustments,audit


def calibrated_interval_by_group(y,p,context_exposure,adjustments):
    """Apply validation-fitted adjustments selected by prior seven-day exposure."""
    groups=exposure_groups(context_exposure)
    qhat=np.array([adjustments.get(group,0.0) for group in groups],dtype=float)
    lower=np.maximum(0.0,np.asarray(p[.10],dtype=float)-qhat)
    upper=np.maximum(lower,np.asarray(p[.90],dtype=float)+qhat)
    actual=np.asarray(y,dtype=float)
    summary={
        "coverage_pct":float(((actual>=lower)&(actual<=upper)).mean()*100),
        "mean_width":float((upper-lower).mean()),
        "mean_adjustment":float(qhat.mean()),
        "context_feature":"stockout_hours_roll_max_7",
    }
    return summary,lower,upper

RISK_FEATURES=(
    "stockout_hours_lag_1","stockout_hours_roll_max_7",
    "stock_status_lag_1","stock_status_lag_7",
    "sales_lag_1","sales_lag_7","sales_lag_14","sales_lag_28",
    "sales_roll_mean_7","sales_roll_mean_28","sales_roll_std_7",
    "day_of_week","day_of_month","month","day_of_year","week_of_year",
)


def fit_stockout_risk_model(train,validation,evaluation):
    """Predict >=16h stockout exposure from prior-only numeric history and calendar."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    features=[name for name in RISK_FEATURES if name in train.columns]
    if len(features)<5:
        raise ValueError(f"Too few prior-only stockout risk features: {features}")
    def matrix(frame):
        return frame[features].astype(float).replace([np.inf,-np.inf],np.nan).fillna(0.0)
    x_train,x_val,x_eval=matrix(train),matrix(validation),matrix(evaluation)
    y_train=(train["stock_hour6_22_cnt"].to_numpy(dtype=float)>=16).astype(int)
    y_val=(validation["stock_hour6_22_cnt"].to_numpy(dtype=float)>=16).astype(int)
    y_eval=(evaluation["stock_hour6_22_cnt"].to_numpy(dtype=float)>=16).astype(int)
    positives=int(y_train.sum())
    negatives=int(len(y_train)-positives)
    if positives<100 or negatives<100:
        raise ValueError(f"Insufficient high/low stockout examples: positives={positives:,}, negatives={negatives:,}")
    params={
        "objective":"binary","metric":"auc","learning_rate":.05,
        "num_leaves":31,"max_depth":-1,"min_data_in_leaf":200,
        "feature_fraction":.9,"bagging_fraction":.9,"bagging_freq":1,
        "lambda_l2":5.0,"scale_pos_weight":negatives/max(positives,1),
        "verbosity":-1,"seed":42,"num_threads":4,
    }
    train_set=lgb.Dataset(x_train,label=y_train,free_raw_data=False)
    valid_set=lgb.Dataset(x_val,label=y_val,reference=train_set,free_raw_data=False)
    model=lgb.train(params,train_set,num_boost_round=800,valid_sets=[valid_set],
                    callbacks=[lgb.early_stopping(50,verbose=False)])
    p_val=np.clip(model.predict(x_val,num_iteration=model.best_iteration),0.0,1.0)
    p_eval=np.clip(model.predict(x_eval,num_iteration=model.best_iteration),0.0,1.0)
    def ranking(labels,prob):
        if np.unique(labels).size<2:
            return {"roc_auc":None,"average_precision":None,"positive_rate_pct":float(labels.mean()*100)}
        return {"roc_auc":float(roc_auc_score(labels,prob)),
                "average_precision":float(average_precision_score(labels,prob)),
                "positive_rate_pct":float(labels.mean()*100)}
    cutoff=float(np.quantile(p_val,.90))
    top=p_eval>=cutoff
    total_positive=max(int(y_eval.sum()),1)
    audit={
        "validation":ranking(y_val,p_val),"evaluation":ranking(y_eval,p_eval),
        "validation_top_decile_threshold":cutoff,
        "evaluation_top_decile_rows":int(top.sum()),
        "evaluation_top_decile_precision_pct":float(y_eval[top].mean()*100) if top.any() else 0.0,
        "evaluation_top_decile_recall_pct":float(y_eval[top].sum()/total_positive*100),
        "prior_only_features":features,
        "positive_definition":"stock_hour6_22_cnt >= 16 on target day",
    }
    return model,p_val,p_eval,audit


def risk_score_edges(validation_prob,bins=10):
    values=np.asarray(validation_prob,dtype=float)
    edges=np.unique(np.quantile(values,np.linspace(0,1,bins+1)))
    if len(edges)<2:
        return np.array([-np.inf,np.inf],dtype=float)
    edges[0]=-np.inf
    edges[-1]=np.inf
    return edges


def risk_score_groups(probability,edges):
    return np.digitize(np.asarray(probability,dtype=float),edges[1:-1],right=True).astype(int)


def risk_conformal_adjustments(y,lower,upper,validation_risk_prob,global_qhat,
                               coverage=.80,min_group_rows=1000,prior_strength_rows=2000):
    edges=risk_score_edges(validation_risk_prob)
    groups=risk_score_groups(validation_risk_prob,edges)
    adjustments={}
    audit=[]
    for group in range(len(edges)-1):
        mask=groups==group
        n=int(mask.sum())
        if n<min_group_rows:
            group_qhat=float(global_qhat); weight=0.0; applied=float(global_qhat)
            method="global_fallback_insufficient_validation_rows"
        else:
            group_qhat=conformal_adjustment(np.asarray(y)[mask],np.asarray(lower)[mask],
                                            np.asarray(upper)[mask],coverage)
            weight=float(n/(n+prior_strength_rows))
            applied=float(weight*group_qhat+(1-weight)*global_qhat)
            method="risk_bin_quantile_shrunk_to_global"
        adjustments[group]=applied
        audit.append({"predicted_risk_decile":group+1,"validation_rows":n,
                      "group_adjustment":float(group_qhat),"global_adjustment":float(global_qhat),
                      "shrinkage_weight":weight,"applied_adjustment":applied,"method":method})
    return edges,adjustments,audit


def calibrated_interval_by_risk(y,p,evaluation_risk_prob,validation_edges,adjustments):
    groups=risk_score_groups(evaluation_risk_prob,validation_edges)
    qhat=np.array([adjustments.get(int(group),0.0) for group in groups],dtype=float)
    lower=np.maximum(0.0,np.asarray(p[.10],dtype=float)-qhat)
    upper=np.maximum(lower,np.asarray(p[.90],dtype=float)+qhat)
    actual=np.asarray(y,dtype=float)
    return {"coverage_pct":float(((actual>=lower)&(actual<=upper)).mean()*100),
            "mean_width":float((upper-lower).mean()),"mean_adjustment":float(qhat.mean()),
            "context_feature":"predicted_high_stockout_risk_decile"},lower,upper


def risk_decile_diagnostics(y,actual_exposure,probability,edges,lower,upper):
    groups=risk_score_groups(probability,edges)
    actual=np.asarray(y,dtype=float); exposure=np.asarray(actual_exposure,dtype=float)
    lower=np.asarray(lower,dtype=float); upper=np.asarray(upper,dtype=float)
    rows=[]
    for group in range(len(edges)-1):
        mask=groups==group
        if mask.any():
            rows.append({"predicted_risk_decile":group+1,"rows":int(mask.sum()),
                         "actual_16_plus_rate_pct":float((exposure[mask]>=16).mean()*100),
                         "coverage_pct":float(((actual[mask]>=lower[mask])&(actual[mask]<=upper[mask])).mean()*100),
                         "mean_interval_width":float((upper[mask]-lower[mask]).mean()),
                         "mean_predicted_high_exposure_probability":float(np.asarray(probability)[mask].mean())})
    return rows
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

        # Conditional conformal is stratified by the maximum stockout exposure
        # observed during the previous seven days for that series. It does not
        # use contemporaneous evaluation-day stock status to choose an interval.
        history_adjustments,history_calibration_audit=conditional_conformal_adjustments(
            yv,vp[.10],vp[.90],
            v["stockout_hours_roll_max_7"].to_numpy(dtype=float),
            qhat,coverage=.80,min_group_rows=1000,prior_strength_rows=2000,
        )
        val_cond_cal,val_cond_lo,val_cond_hi=calibrated_interval_by_group(
            yv,vp,v["stockout_hours_roll_max_7"].to_numpy(dtype=float),history_adjustments
        )
        eval_cond_cal,eval_cond_lo,eval_cond_hi=calibrated_interval_by_group(
            ye,ep,e["stockout_hours_roll_max_7"].to_numpy(dtype=float),history_adjustments
        )
        conditionally_calibrated_ep={.10:eval_cond_lo,.50:ep[.50],.90:eval_cond_hi}

        # Predict current-day high stockout exposure from prior-only numeric
        # history and known calendar features; fit risk bins and conformal
        # adjustments using validation data only.
        risk_model,risk_val,risk_eval,risk_model_metrics=fit_stockout_risk_model(a,v,e)
        risk_edges,risk_adjustments,risk_calibration_audit=risk_conformal_adjustments(
            yv,vp[.10],vp[.90],risk_val,qhat,coverage=.80,
            min_group_rows=1000,prior_strength_rows=2000,
        )
        risk_val_interval,risk_val_lo,risk_val_hi=calibrated_interval_by_risk(
            yv,vp,risk_val,risk_edges,risk_adjustments
        )
        risk_eval_interval,risk_eval_lo,risk_eval_hi=calibrated_interval_by_risk(
            ye,ep,risk_eval,risk_edges,risk_adjustments
        )
        risk_calibrated_ep={.10:risk_eval_lo,.50:ep[.50],.90:risk_eval_hi}
        risk_decile_audit=risk_decile_diagnostics(
            ye,e["stock_hour6_22_cnt"].to_numpy(dtype=float),
            risk_eval,risk_edges,risk_eval_lo,risk_eval_hi,
        )
        risk_path=root/"lightgbm_high_stockout_risk.txt"
        risk_model.save_model(str(risk_path))
        paths.append(risk_path)

        result={"validation":metrics(yv,vp),"evaluation":metrics(ye,ep),
          "conformal_calibration":{"method":"split conformalized quantile regression",
            "nominal_coverage_pct":80.0,"validation_adjustment":qhat,
            "validation_calibrated_interval":val_cal,
            "evaluation_calibrated_interval":eval_cal,
            "caveat":"Calibration is estimated on a prior time window; temporal distribution shift can invalidate nominal future coverage."},
          "evaluation_calibrated_stockout_exposure_strata":strata(e,calibrated_ep),
          "evaluation_conditional_calibrated_stockout_exposure_strata":strata(e,conditionally_calibrated_ep),
          "evaluation_risk_conditioned_stockout_exposure_strata":strata(e,risk_calibrated_ep),
          "evaluation_stockout_exposure_strata":strata(e,ep),
          "risk_conditioned_conformal_calibration":{
            "method":"prior-only high-stockout risk classifier with validation-fitted risk-decile conformal adjustments",
            "context_features":risk_model_metrics["prior_only_features"],
            "risk_model_metrics":risk_model_metrics,
            "risk_score_edges_validation":risk_edges.tolist(),
            "validation_group_adjustments":risk_calibration_audit,
            "validation_interval":risk_val_interval,
            "evaluation_interval":risk_eval_interval,
            "evaluation_metrics":metrics(ye,risk_calibrated_ep),
            "evaluation_risk_deciles":risk_decile_audit,
            "interpretation":"Candidate only. Evaluate actual 16+ hour coverage and interval width; do not promote unless high-exposure calibration improves with a reliable forecast-time risk ranking.",
          },
          "conditional_conformal_calibration":{
            "method":"lagged stockout-exposure-grouped split conformal with shrinkage to global adjustment",
            "context_feature":"stockout_hours_roll_max_7",
            "context_definition":"maximum stock_hour6_22_cnt over the seven daily observations preceding the forecast date",
            "context_available_at_forecast_time":True,
            "nominal_coverage_pct":80.0,
            "minimum_group_validation_rows":1000,
            "prior_strength_rows":2000,
            "global_validation_adjustment":float(qhat),
            "validation_group_adjustments":history_calibration_audit,
            "validation_interval":val_cond_cal,
            "evaluation_interval":eval_cond_cal,
            "evaluation_metrics":metrics(ye,conditionally_calibrated_ep),
            "interpretation":"Candidate calibration only. Compare held-out coverage by contemporaneous stockout exposure and interval width; do not promote unless the high-exposure coverage gain is useful without unacceptable width inflation.",
          },
          "validation_period":{"start":str(min(vd)),"end":str(max(vd))},
          "evaluation_period":{"start":str(ev.get_column("date").min()),"end":str(ev.get_column("date").max())},
          "rows":{"train":len(a),"validation":len(v),"evaluation":len(e)},
          "best_iterations":{f"p{int(q*100)}":int(models[q].best_iteration) for q in QUANTILES},
          "features":feats,"categorical_features":cats,"quantiles":list(QUANTILES),
          "nominal_interval_coverage_pct":80.0,"target":"sale_amount",
          "target_semantics":"Observed daily sales, not guaranteed uncensored demand.",
          "leakage_note":"Contemporaneous hourly sales and stock status are excluded from predictors. Historical stockout exposure is shifted by at least one day; current-day stock exposure is used only for post-hoc evaluation stratification.",
          "stockout_note":"Coverage by stockout exposure is descriptive, not causal. Confirm status encoding against dataset technical documentation.",
          "generated_at_utc":datetime.now(timezone.utc).isoformat()}
        mf=root/"quantile_metrics.json"; mf.write_text(json.dumps(result,indent=2,default=str))
        pf=root/"evaluation_quantile_predictions.csv"
        pd.DataFrame({"date":e["date"].astype(str),"store_id":e["store_id"],"product_id":e["product_id"],
                      "actual_sale_amount":ye.to_numpy(),"p10":ep[.10],"p50":ep[.50],"p90":ep[.90],
                      "p10_calibrated":eval_lo,"p90_calibrated":eval_hi,
                      "p10_conditional_calibrated":eval_cond_lo,"p90_conditional_calibrated":eval_cond_hi,
                      "p10_risk_conditioned":risk_eval_lo,"p90_risk_conditioned":risk_eval_hi,
                      "predicted_high_stockout_risk_probability":risk_eval,
                      "stockout_hours_lag_1":e["stockout_hours_lag_1"].to_numpy(dtype=float),
                      "stockout_hours_roll_max_7":e["stockout_hours_roll_max_7"].to_numpy(dtype=float),
                      "seasonal_naive_lag7":np.maximum(e["sales_lag_7"].to_numpy(dtype=float),0),
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
