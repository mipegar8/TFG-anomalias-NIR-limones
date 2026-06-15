import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from typing import Dict


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """Calcula MAE, RMSE y R² para predicciones de regresión."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred) 
    return {"mae": float(mae), "rmse": rmse, "r2": float(r2)}
