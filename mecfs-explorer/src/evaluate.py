"""
Feature-importance / explainability helper. Run after src/train.py to check
whether the model's top features are biologically plausible (e.g. do they
line up with plasmalogens, phosphatidylcholines, sphingomyelins — the lipids
flagged across multiple independent ME/CFS metabolomics studies?) rather than
trusting the model's accuracy number alone.

Usage:
    python src/evaluate.py --data data/processed/dataset.csv --target label \
        --model-dir data/processed/model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import shap
import torch

from src.data.loaders import load_tabular_dataset, clean_features, impute_and_scale
from src.models.mlp import MECFSClassifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--model-dir", default="data/processed/model")
    parser.add_argument("--top-n", type=int, default=15)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    feature_names = json.loads((model_dir / "feature_names.json").read_text())

    X, y = load_tabular_dataset(args.data, args.target)
    X = clean_features(X)[feature_names]  # keep only features the model was trained on
    X_scaled, _, _, _ = impute_and_scale(X)

    model = MECFSClassifier(n_features=len(feature_names))
    model.load_state_dict(torch.load(model_dir / "model.pt"))
    model.eval()

    def predict_fn(data: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits = model(torch.tensor(data, dtype=torch.float32))
            return torch.sigmoid(logits).numpy()

    # KernelExplainer works with any black-box function and is the simplest
    # correct choice for a small MLP on tabular data. It's slow-ish — that's
    # fine at this N, and a legitimate reason to subsample the background set
    # on a bigger dataset.
    background = shap.sample(X_scaled, min(50, len(X_scaled)))
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X_scaled, nsamples=100)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    ranking = sorted(zip(feature_names, mean_abs_shap), key=lambda t: -t[1])

    print(f"Top {args.top_n} features by mean |SHAP value|:")
    for name, val in ranking[: args.top_n]:
        print(f"  {name:<30} {val:.4f}")

    out_path = model_dir / "feature_importance.json"
    out_path.write_text(
        json.dumps([{"feature": n, "mean_abs_shap": float(v)} for n, v in ranking], indent=2)
    )
    print(f"\nFull ranking saved to {out_path}")
    print(
        "\nNext step: compare this list against the metabolites/genes flagged "
        "in the papers cited in PROJECT_PLAN.md. Overlap is a good sign; a "
        "totally unrelated top feature is worth investigating before trusting "
        "it (data leakage and batch effects are common culprits)."
    )


if __name__ == "__main__":
    main()
