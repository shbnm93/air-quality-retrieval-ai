"""Feature engineering in later iterations must use only known-at-forecast-time values."""
import numpy as np

def lag_change_features(X):
    X=np.asarray(X, dtype=float)
    return np.column_stack([X, X[:,-1]-X[:,-2], X[:,-1]-X.mean(axis=1)])
