from typing import Dict
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV



def get_baseline_models(pls_components: int = 2) -> Dict[str, object]:
    """Devuelve un diccionario nombre → estimador con los modelos de referencia."""
    models = {
        "DummyMean": DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "RandomForest": RandomForestRegressor(n_jobs=-1, random_state=42),
        "PLS": PLSRegression(n_components=max(1, pls_components)),
    }
    return models


def get_pls_estimator(n_components: int):
    """Devuelve un estimador PLS con el número de componentes indicado."""
    return PLSRegression(n_components=n_components)


def tune_pls_components(X, y, cv=5, n_components_list=None, scoring="neg_root_mean_squared_error"):
    """Busca el número óptimo de componentes PLS mediante GridSearchCV."""
    if n_components_list is None:
        n_components_list = list(range(1, min(50, X.shape[1]) + 1))
    param_grid = {"n_components": n_components_list}
    pls = PLSRegression()
    gs = GridSearchCV(pls, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    gs.fit(X, y)
    return gs
