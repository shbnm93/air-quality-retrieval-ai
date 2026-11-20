"""Runnable end-to-end forecast experiment. No held-out test model selection."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from .data_loader import load_air_quality_data, ROOT
from .windows import make_windows
from .evaluation import errors, seasonal_metrics, station_metrics
from .baselines import ridge_model, boosting_model
from .retrieval import HistoricalRetriever

TRAIN_END=np.datetime64("2015-03-01")
VAL_END=np.datetime64("2016-03-01")

def run(output=ROOT/"results", history=24, library=15000, eval_samples=2000, train_samples=100000):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    df=load_air_quality_data(verbose=True)
    subsets=[df[df.datetime < pd.Timestamp(TRAIN_END)],
             df[(df.datetime>=pd.Timestamp(TRAIN_END)) & (df.datetime<pd.Timestamp(VAL_END))],
             df[df.datetime>=pd.Timestamp(VAL_END)]]
    (Xt,yt,st,tt),(Xv,yv,sv,tv),(_,_,_,_)= [make_windows(d,history) for d in subsets]
    # Test windows are intentionally not evaluated during model selection.
    np.random.seed(42)
    print("Forecasting samples train/validation:",len(yt),len(yv))
    ridges={}
    for a in (0.1,10.0,1000.0):
        m=ridge_model(a).fit(Xt,yt)
        pred=m.predict(Xv)
        ridges[a]=errors(yv,pred)["MAE"]
    best_alpha=min(ridges,key=ridges.get)
    ridge=ridge_model(best_alpha).fit(Xt,yt)
    ridge_val=ridge.predict(Xv)
    persistence=Xv[:,-1]
    # Compare nonretrieval methods on exactly the same deterministic validation subset
    rng=np.random.default_rng(42)
    vidx=np.sort(rng.choice(len(yv), min(eval_samples,len(yv)), replace=False))
    tidx=np.sort(rng.choice(len(yt), min(train_samples,len(yt)), replace=False))
    boosted=boosting_model().fit(Xt[tidx],yt[tidx])
    gb_val=boosted.predict(Xv[vidx])
    # Retrieval stores only TRAIN references; no validation or test targets in the index.
    ret=HistoricalRetriever(n_neighbors=5).fit(Xt,yt,st,max_library=library,seed=42)
    knn_val,inds,distances=ret.predict(Xv[vidx])
    ridge_train=ridge.predict(Xt)
    hybrid_val,_,_=ret.predict(Xv[vidx],baseline=ridge_val[vidx],train_baseline=ridge_train)
    preds={"Persistence":persistence[vidx],"Ridge":ridge_val[vidx],
           "GradientBoosting":gb_val,"Historical_kNN":knn_val,
           "Ridge_plus_retrieved_residuals":hybrid_val}
    rows=[]
    for name,p in preds.items():
        rows.append({"model":name,"split":"validation_subset","n":len(vidx),**errors(yv[vidx],p)})
    report=pd.DataFrame(rows).sort_values("MAE")
    report.to_csv(output/"validation_model_comparison.csv",index=False)
    seasonal=pd.concat([seasonal_metrics(yv[vidx],p,tv[vidx]).assign(model=name)
                        for name,p in preds.items()],ignore_index=True)
    seasonal.to_csv(output/"validation_by_season.csv",index=False)
    station=pd.concat([station_metrics(yv[vidx],p,sv[vidx]).assign(model=name)
                       for name,p in preds.items()],ignore_index=True)
    station.to_csv(output/"validation_by_station.csv",index=False)
    pd.DataFrame({"station":sv[vidx],"target_time":tv[vidx],"true_pm25":yv[vidx],
                  **{name: p for name,p in preds.items()}}).to_csv(output/"validation_predictions.csv",index=False)
    meta={"dataset":"UCI Beijing Multi-Site Air Quality", "history_hours":history,
          "forecast_horizon_hours":1,"train_end_exclusive":str(TRAIN_END),
          "validation_end_exclusive":str(VAL_END),"seed":42,
          "training_windows":len(yt),"validation_windows":len(yv),"new_experiment_uses_test_for_selection":False,
          "ridge_alpha_selected_by_validation_mae":best_alpha,"ridge_alpha_maes":ridges,
          "gradient_boosting_training_samples":len(tidx),"retrieval_library_samples":len(ret.y_library),
          "validation_subset_samples":len(vidx),
          "retrieval_library_source":"training_only",
          "limitations":"Single validation period; random deterministic subset for expensive comparisons; no final test model selection."}
    (output/"experiment_metadata.json").write_text(json.dumps(meta,indent=2,default=float))
    print("\nVALIDATION RESULTS — all on the same held-out validation subset\n")
    print(report.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print(f"\nSaved files under {output}")
    return report

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default=str(ROOT/"results"))
    parser.add_argument("--library",type=int,default=15000)
    parser.add_argument("--eval-samples",type=int,default=2000)
    parser.add_argument("--train-samples",type=int,default=100000)
    args=parser.parse_args()
    run(output=Path(args.output),library=args.library,eval_samples=args.eval_samples,train_samples=args.train_samples)
