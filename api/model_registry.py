"""
Loads the trained model + metadata once at startup and hands it to routes via
FastAPI's dependency injection, rather than re-reading files from disk on
every request.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch

from src.data.loaders import impute_and_scale
from src.models.mlp import MECFSClassifier

MODEL_DIR = Path(__file__).parent.parent / "data" / "processed" / "model"
DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "dataset.csv"


class ModelBundle:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.feature_names: list[str] = json.loads(
            (model_dir / "feature_names.json").read_text()
        )
        self.metrics: dict = json.loads((model_dir / "metrics.json").read_text())

        self.model = MECFSClassifier(n_features=len(self.feature_names))
        self.model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
        self.model.eval()

        # Fit the imputer/scaler + a small SHAP background set from the same
        # processed dataset the model was trained on, so /predict inputs get
        # transformed consistently with training.
        df = pd.read_csv(DATA_PATH)
        X = df[self.feature_names]
        self._X_scaled, _, self.imputer, self.scaler = impute_and_scale(X)
        background = shap.sample(self._X_scaled, min(50, len(self._X_scaled)))
        self.explainer = shap.KernelExplainer(self._predict_proba, background)

    def _predict_proba(self, data: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits = self.model(torch.tensor(data, dtype=torch.float32))
            return torch.sigmoid(logits).numpy()

    def predict(self, features: dict[str, float]) -> dict:
        missing = set(self.feature_names) - set(features)
        if missing:
            raise ValueError(f"Missing required features: {sorted(missing)}")
        extra = set(features) - set(self.feature_names)
        if extra:
            raise ValueError(f"Unknown features (not used by this model): {sorted(extra)}")

        x = np.array([[features[f] for f in self.feature_names]])
        x_imputed = self.imputer.transform(x)
        x_scaled = self.scaler.transform(x_imputed)

        prob = float(self._predict_proba(x_scaled)[0])
        shap_vals = self.explainer.shap_values(x_scaled, nsamples=100)[0]

        contributions = sorted(
            zip(self.feature_names, x[0], shap_vals),
            key=lambda t: -abs(t[2]),
        )[:5]

        return {
            "predicted_class": int(prob > 0.5),
            "probability": prob,
            "top_contributing_features": [
                {"feature": f, "value": float(v), "contribution": float(c)}
                for f, v, c in contributions
            ],
        }


@lru_cache(maxsize=1)
def get_model_bundle() -> ModelBundle:
    return ModelBundle()
