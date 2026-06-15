import numpy as np
import pandas as pd
from typing import List, Optional


def univariate_outlier_table(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    exclude_cols: Optional[List[str]] = None,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> pd.DataFrame:
    """Genera la tabla comparativa de outliers univariantes por los 3 métodos.

    Parámetros
    ----------
    df          : DataFrame original (antes de escalar/imputar).
    columns     : columnas a analizar. Si es None, usa todas las numéricas.
    exclude_cols: columnas a excluir (p.ej. los targets químicos).
    p_low       : percentil inferior para la regla de percentiles (defecto 1).
    p_high      : percentil superior para la regla de percentiles (defecto 99).

    Devuelve
    --------
    DataFrame con una fila por variable y columnas:
        Variable | Muestras Válidas | Outliers Percentil | Outliers 3-Sigma | Outliers IQR | % Outliers IQR
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if exclude_cols:
        columns = [c for c in columns if c not in exclude_cols]

    rows = []
    for var in columns:
        if var not in df.columns:
            continue
        serie = pd.to_numeric(df[var], errors="coerce").dropna()
        if serie.empty:
            continue

        n = len(serie)

        # 1: Percentiles (P1–P99)
        lim_pct_inf = float(np.percentile(serie, p_low))
        lim_pct_sup = float(np.percentile(serie, p_high))
        n_pct = int(((serie < lim_pct_inf) | (serie > lim_pct_sup)).sum())

        # 2: 3-Sigma (media ± 3·std) 
        media = float(serie.mean())
        std = float(serie.std())
        n_sigma = int(((serie < media - 3 * std) | (serie > media + 3 * std)).sum())

        # 3: IQR / Tukey (Q1 - 1.5·IQR,  Q3 + 1.5·IQR)
        q1 = float(np.percentile(serie, 25))
        q3 = float(np.percentile(serie, 75))
        iqr = q3 - q1
        n_iqr = int(((serie < q1 - 1.5 * iqr) | (serie > q3 + 1.5 * iqr)).sum())

        rows.append({
            "Variable": var,
            "Muestras Válidas": n,
            f"Outliers Percentil (P{int(p_low)}-P{int(p_high)})": n_pct,
            "Outliers 3-Sigma": n_sigma,
            "Outliers IQR (Tukey)": n_iqr,
            "% Outliers IQR": round(100.0 * n_iqr / n, 2),
        })

    return pd.DataFrame(rows)
