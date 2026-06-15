from pathlib import Path
import json
import pandas as pd
from typing import Tuple, Dict


def load_clean_df(data_path: Path = Path("results/df_clean_initial.csv"),
                  meta_path: Path = Path("results/df_meta_initial.json")) -> Tuple[pd.DataFrame, Dict]:
    """Load cleaned dataframe and metadata JSON produced by the prep notebook."""
    data_path = Path(data_path)
    meta_path = Path(meta_path)
    df = pd.read_csv(data_path)
    with open(meta_path, "r", encoding="utf8") as fh:
        meta = json.load(fh)
    return df, meta


def read_prepared_splits(prepared_dir: Path = Path("results/prepared")) -> Dict[str, pd.DataFrame]:
    """Read prepared X/y splits saved as CSVs in `results/prepared/`.

    Returns a dict with keys: X_train, X_val, X_test, y_train, y_val, y_test
    """
    p = Path(prepared_dir)
    out = {}
    for name in ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]:
        fp = p / f"{name}.csv"
        if fp.exists():
            out[name] = pd.read_csv(fp, index_col=0)
        else:
            out[name] = None
    return out


def ensure_model_dir(root: Path = Path("results/modelado"), target: str = "default") -> Path:
    """Create and return a model directory for a given target."""
    root = Path(root)
    target_dir = root / target
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir
