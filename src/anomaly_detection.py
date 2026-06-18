"""
anomaly_detection.py  –  Detección de anomalías avanzada con PyOD.

Este módulo es el núcleo metodológico del TFG: aplica varios algoritmos
modernos de detección de anomalías (PyOD) sobre los datos espectrales y
morfológicos, y compara sus resultados entre sí y frente a los métodos
univariantes clásicos del notebook 1.

Requiere: pip install pyod
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def get_anomaly_detectors(
    contamination: float = 0.05,
    random_state: int = 42,
    n_neighbors: int = 20,
) -> Dict[str, object]:
    """Devuelve un diccionario nombre -> detector de anomalías de PyOD.

    contamination: proporción esperada de anomalías en los datos (0.05 = 5%).
    """
    from pyod.models.iforest import IForest
    from pyod.models.lof import LOF
    from pyod.models.ocsvm import OCSVM
    from pyod.models.pca import PCA as PyOD_PCA
    from pyod.models.copod import COPOD

    detectors = {
        "IsolationForest": IForest(contamination=contamination, random_state=random_state),
        "LOF": LOF(contamination=contamination, n_neighbors=n_neighbors),
        "OCSVM": OCSVM(contamination=contamination),
        "PCA": PyOD_PCA(contamination=contamination, random_state=random_state),
        "COPOD": COPOD(contamination=contamination),
    }
    return detectors


def run_anomaly_detection(
    X: pd.DataFrame,
    detectors: Optional[Dict[str, object]] = None,
    contamination: float = 0.05,
) -> pd.DataFrame:
    """Ejecuta todos los detectores sobre X y devuelve un DataFrame con
    una columna booleana por modelo (True = anomalía) más el score crudo.

    Devuelve
    --------
    DataFrame indexado igual que X, con columnas:
        <modelo>_anomaly  (bool)
        <modelo>_score    (float, mayor = más anómalo)
    """
    if detectors is None:
        detectors = get_anomaly_detectors(contamination=contamination)

    X_values = X.values if hasattr(X, "values") else np.asarray(X)
    results = pd.DataFrame(index=X.index if hasattr(X, "index") else range(len(X)))

    for name, model in detectors.items():
        model.fit(X_values)
        labels = model.labels_  # 0 = normal, 1 = anomalía
        scores = model.decision_scores_  # mayor score = más anómalo

        results[f"{name}_anomaly"] = labels.astype(bool)
        results[f"{name}_score"] = scores

    return results


def consensus_anomalies(
    results: pd.DataFrame,
    min_models: int = 3,
) -> pd.DataFrame:
    """Calcula el consenso entre modelos: cuántos detectores marcan cada
    muestra como anómala, y si supera el umbral min_models se considera
    una anomalía "fuerte" (consensuada).

    Parámetros
    ----------
    results    : salida de run_anomaly_detection().
    min_models : número mínimo de modelos que deben coincidir para
                 considerar la muestra una anomalía de consenso.

    Devuelve
    --------
    DataFrame con columnas: n_models_flagged, is_consensus_anomaly
    """
    anomaly_cols = [c for c in results.columns if c.endswith("_anomaly")]
    n_flagged = results[anomaly_cols].sum(axis=1)

    consensus = pd.DataFrame(index=results.index)
    consensus["n_models_flagged"] = n_flagged
    consensus["total_models"] = len(anomaly_cols)
    consensus["is_consensus_anomaly"] = n_flagged >= min_models

    return consensus


def summarize_detection(results: pd.DataFrame) -> pd.DataFrame:
    """Genera una tabla resumen: nº y % de anomalías detectadas por cada modelo.

    Útil para la comparativa entre métodos que pide el tutor.
    """
    anomaly_cols = [c for c in results.columns if c.endswith("_anomaly")]
    n_total = len(results)

    rows = []
    for col in anomaly_cols:
        model_name = col.replace("_anomaly", "")
        n_anom = int(results[col].sum())
        rows.append({
            "Modelo": model_name,
            "Anomalías detectadas": n_anom,
            "% del total": round(100 * n_anom / n_total, 2),
        })

    return pd.DataFrame(rows).sort_values("Anomalías detectadas", ascending=False).reset_index(drop=True)


def overlap_matrix(results: pd.DataFrame) -> pd.DataFrame:
    """Calcula el solapamiento (Jaccard) entre cada par de modelos: de las
    muestras que un modelo marca como anómalas, qué porcentaje también
    marca el otro modelo. Ayuda a ver si los métodos coinciden o son
    complementarios.
    """
    anomaly_cols = [c for c in results.columns if c.endswith("_anomaly")]
    names = [c.replace("_anomaly", "") for c in anomaly_cols]

    matrix = pd.DataFrame(index=names, columns=names, dtype=float)
    for i, col_i in enumerate(anomaly_cols):
        set_i = set(results.index[results[col_i]])
        for j, col_j in enumerate(anomaly_cols):
            set_j = set(results.index[results[col_j]])
            union = set_i | set_j
            inter = set_i & set_j
            jaccard = len(inter) / len(union) if union else 0.0
            matrix.iloc[i, j] = round(jaccard, 3)

    return matrix


def crosscheck_with_target(
    df: pd.DataFrame,
    consensus: pd.DataFrame,
    target_cols: List[str],
) -> pd.DataFrame:
    """Compara los valores de los targets químicos entre muestras
    consensuadas como anómalas y muestras normales (test de Mann-Whitney).

    Responde a la pregunta: ¿las anomalías detectadas tienen una calidad
    química distinta del resto?
    """
    from scipy.stats import mannwhitneyu

    common_idx = df.index.intersection(consensus.index)
    is_anomaly = consensus.loc[common_idx, "is_consensus_anomaly"]

    rows = []
    for target in target_cols:
        if target not in df.columns:
            continue
        serie = pd.to_numeric(df.loc[common_idx, target], errors="coerce")
        group_anom = serie[is_anomaly].dropna()
        group_normal = serie[~is_anomaly].dropna()

        if len(group_anom) < 3 or len(group_normal) < 3:
            continue

        stat, pvalue = mannwhitneyu(group_anom, group_normal, alternative="two-sided")

        rows.append({
            "target": target,
            "media_anomalias": round(float(group_anom.mean()), 3),
            "media_normales": round(float(group_normal.mean()), 3),
            "n_anomalias": len(group_anom),
            "n_normales": len(group_normal),
            "p_value": round(float(pvalue), 4),
            "diferencia_significativa": pvalue < 0.05,
        })

    return pd.DataFrame(rows)
