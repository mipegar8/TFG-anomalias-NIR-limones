from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA


def _infer_spectral_cols(df: pd.DataFrame, prefix: str = "B") -> List[str]:
    # Fallback: infiere espectrales por prefijo si no vienen del meta
    cols = [c for c in df.columns if c.startswith(prefix) or c.startswith(prefix.lower())]
    return cols


def _series_mode(series: pd.Series):
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    modes = non_null.mode()
    if modes.empty:
        return np.nan
    return modes.iloc[0]


def build_dataset_for_target(
    df: pd.DataFrame,
    meta: Optional[Dict] = None,
    target: str = "tss",
    spectral_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
    morf_cols: Optional[List[str]] = None,
    group_col: Optional[str] = None,
    random_state: int = 42,
    val_size: float = 0.2,
    test_size: float = 0.2,
) -> Dict:
    """Create train/val/test splits and spectral-only subsets.

    Returns a dict with keys: X_train, X_val, X_test, X_train_spec, X_val_spec, X_test_spec,
    y_train, y_val, y_test, scalers and encoders used.
    """
    from sklearn.model_selection import train_test_split

    df = df.copy()
    if target not in df.columns:
        raise ValueError(f"Target {target} not found in dataframe columns")

    df = df[df[target].notna()].copy()

    # Inferir grupos de columnas desde el meta del notebook 1
    if meta is None:
        meta = {}

    if spectral_cols is None:
        spectral_cols = meta.get("spectral_cols") or meta.get("spec_cols") or meta.get("spectral_bands") or _infer_spectral_cols(df)

    if morf_cols is None:
        morf_cols = meta.get("morpho_cols") or meta.get("morf_cols") or []

    if cat_cols is None:
        cat_cols = meta.get("categorical_cols") or meta.get("cat_cols") or list(df.select_dtypes(include=["object", "category"]).columns)
        
    if group_col is None:
        group_col = meta.get("group_col")

    origin_col = next((c for c in df.columns if c.lower() == "origin"), None)
    if origin_col is None:
        raise ValueError("Column 'origin' not found in dataframe")

    # Separación de features: numéricas estrictas vs categóricas (incluye fechas y texto)
    feature_cols = [c for c in df.columns if c != target]
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
    extra_non_numeric = [c for c in feature_cols if c not in numeric_cols and c not in cat_cols]
    cat_cols = list(dict.fromkeys(cat_cols + extra_non_numeric))
    cat_cols = [c for c in cat_cols if c != origin_col
            and not str(c).lower().startswith("unnamed")
            and df[c].nunique(dropna=False) < len(df)]

    X = df[numeric_cols + cat_cols]
    y = df[target]

    origin_counts = df[origin_col].dropna().value_counts()
    if origin_counts.empty:
        raise ValueError("No origin values available for block validation split")

    test_origin = "C" if "C" in origin_counts.index else origin_counts.idxmin()
    test_mask = df[origin_col].eq(test_origin)
    if not test_mask.any() or test_mask.all():
        raise ValueError("Unable to build a valid test block from origin")

    df_test = df.loc[test_mask].copy()
    df_remaining = df.loc[~test_mask].copy()

    if df_remaining.empty:
        raise ValueError("Empty training pool after selecting the test origin")

    remaining_origin_series = df_remaining[origin_col]
    remaining_origin_counts = remaining_origin_series.dropna().value_counts()
    if len(remaining_origin_counts) >= 2:
        val_origin = remaining_origin_counts.idxmin()
        val_mask = df_remaining[origin_col].eq(val_origin)
        if val_mask.any() and not val_mask.all():
            df_val = df_remaining.loc[val_mask].copy()
            df_train = df_remaining.loc[~val_mask].copy()
        else:
            val_fraction = val_size / max(1.0 - (len(df_test) / len(df)), 1e-12)
            stratify_remaining = remaining_origin_series if remaining_origin_series.notna().all() and remaining_origin_counts.min() >= 2 else None
            X_train, X_val, y_train, y_val = train_test_split(
                df_remaining[numeric_cols + cat_cols],
                df_remaining[target],
                test_size=val_fraction,
                random_state=random_state,
                stratify=stratify_remaining,
            )
            df_train = X_train.join(df_remaining[[origin_col]], how="left")
            df_val = X_val.join(df_remaining[[origin_col]], how="left")
    else:
        val_fraction = val_size / max(1.0 - (len(df_test) / len(df)), 1e-12)
        stratify_remaining = remaining_origin_series if remaining_origin_series.notna().all() and remaining_origin_counts.min() >= 2 else None
        X_train, X_val, y_train, y_val = train_test_split(
            df_remaining[numeric_cols + cat_cols],
            df_remaining[target],
            test_size=val_fraction,
            random_state=random_state,
            stratify=stratify_remaining,
        )
        df_train = X_train.join(df_remaining[[origin_col]], how="left")
        df_val = X_val.join(df_remaining[[origin_col]], how="left")

    X_train = df_train[numeric_cols + cat_cols].copy()
    X_val = df_val[numeric_cols + cat_cols].copy()
    X_test = df_test[numeric_cols + cat_cols].copy()
    y_train = df[target].reindex(X_train.index) if target not in df_train.columns else df_train[target]
    y_val   = df[target].reindex(X_val.index)   if target not in df_val.columns   else df_val[target]
    y_test  = df[target].reindex(X_test.index)  if target not in df_test.columns  else df_test[target]

    variety_col = "variety" if "variety" in df.columns else group_col

    # Imputación por variedad: mediana/moda del grupo, fallback a estadístico global del train    
    X_train_num = X_train[numeric_cols].copy()
    X_val_num = X_val[numeric_cols].copy()
    X_test_num = X_test[numeric_cols].copy()

    X_train_cat = X_train[cat_cols].copy() if cat_cols else pd.DataFrame(index=X_train.index)
    X_val_cat = X_val[cat_cols].copy() if cat_cols else pd.DataFrame(index=X_val.index)
    X_test_cat = X_test[cat_cols].copy() if cat_cols else pd.DataFrame(index=X_test.index)

    if variety_col and variety_col in X_train.columns:
        train_groups = X_train.groupby(variety_col, dropna=False)
        numeric_group_medians = train_groups[numeric_cols].median(numeric_only=True) if numeric_cols else pd.DataFrame()
        categorical_group_modes = train_groups[cat_cols].agg(_series_mode) if cat_cols else pd.DataFrame()
    else:
        numeric_group_medians = pd.DataFrame()
        categorical_group_modes = pd.DataFrame()

    global_numeric_medians = X_train_num.median(numeric_only=True) if numeric_cols else pd.Series(dtype=float)
    global_categorical_modes = X_train_cat.agg(_series_mode) if cat_cols else pd.Series(dtype=object)

    def _impute_by_variety(frame: pd.DataFrame) -> pd.DataFrame:
        imputed = frame.copy()

        if variety_col and variety_col in imputed.columns and not numeric_group_medians.empty:
            variety_values = imputed[variety_col]
            for col in numeric_cols:
                if col not in imputed.columns:
                    continue
                group_fills = variety_values.map(numeric_group_medians[col]) if col in numeric_group_medians.columns else pd.Series(index=imputed.index, dtype=float)
                imputed[col] = imputed[col].fillna(group_fills).fillna(global_numeric_medians.get(col, np.nan))
        else:
            for col in numeric_cols:
                if col in imputed.columns:
                    imputed[col] = imputed[col].fillna(global_numeric_medians.get(col, np.nan))

        if variety_col and variety_col in imputed.columns and not categorical_group_modes.empty:
            variety_values = imputed[variety_col]
            for col in cat_cols:
                if col not in imputed.columns:
                    continue
                group_fills = variety_values.map(categorical_group_modes[col]) if col in categorical_group_modes.columns else pd.Series(index=imputed.index, dtype=object)
                imputed[col] = imputed[col].fillna(group_fills).fillna(global_categorical_modes.get(col, np.nan))
        else:
            for col in cat_cols:
                if col in imputed.columns:
                    imputed[col] = imputed[col].fillna(global_categorical_modes.get(col, np.nan))

        return imputed

    X_train_num_imp = _impute_by_variety(X_train_num)
    X_val_num_imp = _impute_by_variety(X_val_num)
    X_test_num_imp = _impute_by_variety(X_test_num)

    X_train_cat = _impute_by_variety(X_train_cat)
    X_val_cat = _impute_by_variety(X_val_cat)
    X_test_cat = _impute_by_variety(X_test_cat)

    # Codificación one-hot de variables categóricas (solo variety, clon_type y similares)
    encoder = None
    if cat_cols:
        try:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        except TypeError:
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoder.fit(X_train_cat)
        X_train_cat = pd.DataFrame(encoder.transform(X_train_cat), index=X_train.index, columns=encoder.get_feature_names_out(cat_cols))
        X_val_cat = pd.DataFrame(encoder.transform(X_val_cat), index=X_val.index, columns=encoder.get_feature_names_out(cat_cols))
        X_test_cat = pd.DataFrame(encoder.transform(X_test_cat), index=X_test.index, columns=encoder.get_feature_names_out(cat_cols))

    X_train_proc = pd.concat([X_train_num_imp, X_train_cat], axis=1)
    X_val_proc = pd.concat([X_val_num_imp, X_val_cat], axis=1)
    X_test_proc = pd.concat([X_test_num_imp, X_test_cat], axis=1)

    # Escalado de espectrales y morfológicas + PCA al 95% de varianza
    spec_cols_present = [c for c in spectral_cols if c in X_train_proc.columns]
    scaler_spec = StandardScaler()
    scaler_morf = StandardScaler()

    if spec_cols_present:
        X_train_spec = pd.DataFrame(scaler_spec.fit_transform(X_train_proc[spec_cols_present]), columns=spec_cols_present, index=X_train_proc.index)
        X_val_spec = pd.DataFrame(scaler_spec.transform(X_val_proc[spec_cols_present]), columns=spec_cols_present, index=X_val_proc.index)
        X_test_spec = pd.DataFrame(scaler_spec.transform(X_test_proc[spec_cols_present]), columns=spec_cols_present, index=X_test_proc.index)
        
        pca_spec = PCA(n_components=0.95, random_state=random_state)
        X_train_spec_pca_values = pca_spec.fit_transform(X_train_spec)
        X_val_spec_pca_values = pca_spec.transform(X_val_spec)
        X_test_spec_pca_values = pca_spec.transform(X_test_spec)
        
        pca_cols = [f"PC{i+1}" for i in range(pca_spec.n_components_)]
        X_train_spec_pca = pd.DataFrame(X_train_spec_pca_values, columns=pca_cols, index=X_train_proc.index)
        X_val_spec_pca = pd.DataFrame(X_val_spec_pca_values, columns=pca_cols, index=X_val_proc.index)
        X_test_spec_pca = pd.DataFrame(X_test_spec_pca_values, columns=pca_cols, index=X_test_proc.index)
    else:
        X_train_spec = pd.DataFrame(index=X_train_proc.index)
        X_val_spec = pd.DataFrame(index=X_val_proc.index)
        X_test_spec = pd.DataFrame(index=X_test_proc.index)
        pca_spec = None
        X_train_spec_pca = pd.DataFrame(index=X_train_proc.index)
        X_val_spec_pca = pd.DataFrame(index=X_val_proc.index)
        X_test_spec_pca = pd.DataFrame(index=X_test_proc.index)

    morf_cols_present = [c for c in numeric_cols if c in X_train_proc.columns and c not in spec_cols_present]
    if morf_cols_present:
        X_train_morf = pd.DataFrame(scaler_morf.fit_transform(X_train_proc[morf_cols_present]), columns=morf_cols_present, index=X_train_proc.index)
        X_val_morf = pd.DataFrame(scaler_morf.transform(X_val_proc[morf_cols_present]), columns=morf_cols_present, index=X_val_proc.index)
        X_test_morf = pd.DataFrame(scaler_morf.transform(X_test_proc[morf_cols_present]), columns=morf_cols_present, index=X_test_proc.index)
    else:
        X_train_morf = pd.DataFrame(index=X_train_proc.index)
        X_val_morf = pd.DataFrame(index=X_val_proc.index)
        X_test_morf = pd.DataFrame(index=X_test_proc.index)

    # Ensamblado final: espectrales escaladas + morfológicas escaladas + categóricas codificadas
    X_train_full = pd.concat([X_train_spec, X_train_morf, X_train_cat], axis=1).reindex(index=X_train.index)
    X_val_full = pd.concat([X_val_spec, X_val_morf, X_val_cat], axis=1).reindex(index=X_val.index)
    X_test_full = pd.concat([X_test_spec, X_test_morf, X_test_cat], axis=1).reindex(index=X_test.index)

    imputer = {
        "type": "groupby_variety",
        "group_col": variety_col,
        "numeric_group_medians": numeric_group_medians.to_dict() if not numeric_group_medians.empty else {},
        "categorical_group_modes": categorical_group_modes.to_dict() if not categorical_group_modes.empty else {},
        "global_numeric_medians": global_numeric_medians.to_dict() if not global_numeric_medians.empty else {},
        "global_categorical_modes": global_categorical_modes.to_dict() if not global_categorical_modes.empty else {},
    }

    return {
        "X_train": X_train_full,
        "X_val": X_val_full,
        "X_test": X_test_full,
        "X_train_spec": X_train_spec,
        "X_val_spec": X_val_spec,
        "X_test_spec": X_test_spec,
        "X_train_spec_pca": X_train_spec_pca,
        "X_val_spec_pca": X_val_spec_pca,
        "X_test_spec_pca": X_test_spec_pca,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "imputer": imputer,
        "encoder": encoder,
        "scaler_spec": scaler_spec,
        "scaler_morf": scaler_morf,
        "pca_spec": pca_spec,
        "spec_cols": spec_cols_present,
        "morf_cols": morf_cols_present,
        "target_name": target,
    }
