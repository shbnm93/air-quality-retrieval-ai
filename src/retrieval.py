"""Time-ordered historical pattern retrieval, restricted to training references."""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

class HistoricalRetriever:
    def __init__(self, n_neighbors=5):
        self.n_neighbors=n_neighbors
    def fit(self, X_train, y_train, stations_train=None, max_library=15000, seed=42):
        X_train=np.asarray(X_train, dtype=float)
        y_train=np.asarray(y_train,dtype=float)
        rng=np.random.default_rng(seed)
        choices=np.arange(len(X_train))
        if len(choices)>max_library:
            choices=np.sort(rng.choice(choices,max_library,replace=False))
        self.scaler=StandardScaler().fit(X_train)
        self.X_library=X_train[choices]
        self.y_library=y_train[choices]
        self.stations_library=None if stations_train is None else np.asarray(stations_train)[choices]
        self.knn=NearestNeighbors(n_neighbors=min(self.n_neighbors,len(choices)),algorithm="auto",n_jobs=-1)
        self.knn.fit(self.scaler.transform(self.X_library))
        self.library_indices=choices
        return self
    def predict(self, X, baseline=None, train_baseline=None):
        """Return retrieval targets, or ridge+average retrieved *training residual*."""
        distances,inds=self.knn.kneighbors(self.scaler.transform(X))
        neighbors=self.y_library[inds]
        kpred=neighbors.mean(axis=1)
        if baseline is None:
            return kpred, inds, distances
        if train_baseline is None:
            raise ValueError("Supply training predictions for retrieved training examples")
        residuals=self.y_library - np.asarray(train_baseline)[self.library_indices]
        hybrid=np.asarray(baseline) + residuals[inds].mean(axis=1)
        return hybrid,inds,distances
