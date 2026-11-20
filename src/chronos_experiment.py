"""Optional zero-shot Chronos-Bolt baseline; separate run to keep core dependencies light.

Install: python -m pip install chronos-forecasting
Model downloads weights from Hugging Face on first execution.
We deliberately evaluate only a fixed validation subset (no test access).
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .data_loader import load_air_quality_data, ROOT
from .windows import make_windows
from .evaluation import errors

def main(samples=256,batch_size=32):
    try:
        from chronos import BaseChronosPipeline
    except ImportError as exc:
        raise SystemExit("First install: python -m pip install chronos-forecasting") from exc
    df=load_air_quality_data()
    val=df[(df.datetime >= "2015-03-01") & (df.datetime < "2016-03-01")]
    X,y,stations,times=make_windows(val,history=24)
    rng=np.random.default_rng(42)
    chosen=np.sort(rng.choice(len(y),min(samples,len(y)),replace=False))
    device="mps" if torch.backends.mps.is_available() else "cpu"
    print("Loading amazon/chronos-bolt-tiny using",device)
    try:
        pipeline=BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-tiny",device_map=device)
    except (RuntimeError,ValueError) as e:
        print("Device initialization failed, retrying on CPU:",e)
        pipeline=BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-tiny",device_map="cpu")
        device="cpu"
    predictions=[]
    for start in range(0,len(chosen),batch_size):
        context=torch.from_numpy(X[chosen[start:start+batch_size]]).float()
        forecast=pipeline.predict(context=context,prediction_length=1)
        # Chronos-Bolt quantiles: [batch, quantiles, horizon]. Median is center quantile.
        if isinstance(forecast,(tuple,list)):
            forecast=forecast[0]
        if isinstance(forecast,torch.Tensor):
            forecast=forecast.detach().cpu().numpy()
        forecast=np.asarray(forecast)
        if forecast.ndim!=3:
            raise RuntimeError(f"Unexpected Chronos forecast shape: {forecast.shape}")
        median=forecast[:,forecast.shape[1]//2,0]
        predictions.extend(median)
    preds=np.array(predictions)
    if len(preds)!=len(chosen):
        raise RuntimeError("Incomplete Chronos prediction batch")
    out=ROOT/"results"
    out.mkdir(exist_ok=True,parents=True)
    pd.DataFrame({"station":stations[chosen],"target_time":times[chosen],
                  "actual":y[chosen],"chronos_bolt_tiny":preds,
                  "persistence":X[chosen,-1]}).to_csv(out/"chronos_validation_predictions.csv",index=False)
    print("Chronos-Bolt Tiny zero-shot validation",errors(y[chosen],preds))
    print("Persistence, SAME subset",errors(y[chosen],X[chosen,-1]))
    print("Saved results/chronos_validation_predictions.csv")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--samples",type=int,default=256)
    parser.add_argument("--batch-size",type=int,default=32)
    a=parser.parse_args()
    main(samples=a.samples,batch_size=a.batch_size)
