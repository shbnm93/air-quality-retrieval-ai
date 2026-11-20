import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def errors(y, prediction):
    y = np.asarray(y, dtype=float)
    p = np.asarray(prediction, dtype=float)
    return {"MAE": float(mean_absolute_error(y, p)),
            "RMSE": float(np.sqrt(mean_squared_error(y, p)))}

def seasonal_metrics(y, prediction, target_times):
    months = pd.DatetimeIndex(target_times).month
    seasons = np.asarray([("Winter" if m in (12,1,2) else
                            "Spring" if m in (3,4,5) else
                            "Summer" if m in (6,7,8) else "Autumn") for m in months])
    rows=[]
    for season in ["Winter", "Spring", "Summer", "Autumn"]:
        mask=seasons == season
        if mask.any():
            rows.append({"season":season,"n":int(mask.sum()), **errors(y[mask], prediction[mask])})
    return pd.DataFrame(rows)

def station_metrics(y, prediction, stations):
    rows=[]
    for station in sorted(set(stations)):
        mask=stations==station
        rows.append({"station":station,"n":int(mask.sum()),**errors(y[mask],prediction[mask])})
    return pd.DataFrame(rows)
