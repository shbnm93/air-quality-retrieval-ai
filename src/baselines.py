from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

def ridge_model(alpha=10.0):
    return Ridge(alpha=alpha)

def boosting_model():
    return HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=15,
                                         learning_rate=0.06, l2_regularization=5.0,
                                         random_state=42)