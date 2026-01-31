# src/models_loader.py
import pickle
from pathlib import Path

def load_models(models_dir="models"):
    models_path = Path(models_dir)
    models = {}
    for pkl_file in models_path.glob("*.pkl"):
        with open(pkl_file, "rb") as f:
            models[pkl_file.stem] = pickle.load(f)
    return models
