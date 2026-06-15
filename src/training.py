from typing import Tuple, Dict
import joblib
from sklearn.model_selection import GridSearchCV

from .metrics import regression_metrics


def evaluate_model(estimator, X_train, y_train, X_eval, y_eval) -> Tuple[object, Dict]:
    """Entrena el estimador y evalúa en el conjunto de evaluación. Devuelve (estimador_entrenado, métricas)."""
    estimator.fit(X_train, y_train)
    preds = estimator.predict(X_eval)
    metrics = regression_metrics(y_eval, preds)
    return estimator, metrics


def grid_search_tune(estimator, param_grid: dict, X, y, cv=5, scoring=None):
    """Ajusta hiperparámetros mediante GridSearchCV. Devuelve el objeto GridSearch ya entrenado."""
    gs = GridSearchCV(estimator, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    gs.fit(X, y)
    return gs


def save_model(estimator, path):
    """Guarda el estimador en disco con joblib."""
    joblib.dump(estimator, path)


def load_model(path):
    """Carga un estimador guardado con joblib."""
    return joblib.load(path)
