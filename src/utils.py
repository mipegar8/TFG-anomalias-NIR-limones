from pathlib import Path
import joblib


def save_joblib(obj, path: Path):
    """Guarda un objeto en disco con joblib, creando el directorio si no existe."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_joblib(path: Path):
    """Carga un objeto guardado con joblib."""
    return joblib.load(path)
