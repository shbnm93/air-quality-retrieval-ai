import numpy as np
import pandas as pd
import pytest
from src.windows import make_windows
from src.evaluation import errors
from src.retrieval import HistoricalRetriever

def synthetic():
    start=pd.Timestamp("2020-01-01")
    rows=[]
    for station,offset in (("A",0.0),("B",1000.0)):
        for t in range(30):
            if station=="B" and t==10:
                continue
            val=offset+t
            if station=="A" and t==12:
                val=np.nan
            rows.append({"station":station,"datetime":start+pd.Timedelta(hours=t),"PM2.5":val})
    return pd.DataFrame(rows)

def test_consecutive_complete_station_windows():
    raw=synthetic()
    X,y,stations,times=make_windows(raw,history=3)
    assert X.shape[1]==3
    assert len(X)==len(y)==len(stations)==len(times)
    assert np.isfinite(X).all() and np.isfinite(y).all()
    for x,target,station,t in zip(X,y,stations,times):
        g=raw[raw.station==station].set_index('datetime')
        time=pd.Timestamp(t)
        assert target==g.loc[time,'PM2.5']
        for lag,z in enumerate(x[::-1],1):
            assert z==g.loc[time-pd.Timedelta(hours=lag),'PM2.5']
        assert np.all(x<1000) if station=='A' else np.all(x>=1000)

def test_bad_history():
    with pytest.raises(ValueError):
        make_windows(synthetic(),history=0)

def test_duplicate_timestamps_fail():
    raw=synthetic()
    duplicate=pd.concat([raw,raw.iloc[[0]]],ignore_index=True)
    with pytest.raises(ValueError,match='Duplicate'):
        make_windows(duplicate,history=3)

def test_error_metrics():
    m=errors([1.,2.],[1.,4.])
    assert m['MAE']==1.0
    assert abs(m['RMSE']-np.sqrt(2))<1e-9

def test_retrieval_training_only_and_reproducible():
    X=np.arange(100*24,dtype=float).reshape(100,24)
    y=np.arange(100,dtype=float)
    ret=HistoricalRetriever(n_neighbors=3).fit(X,y,max_library=20,seed=42)
    p,ind,_=ret.predict(X[:5])
    assert len(p)==5 and ind.max()<20
    assert np.allclose(p, ret.y_library[ind].mean(axis=1))
