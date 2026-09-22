"""
K-fold training loop for the small-N ME/CFS classification task.

Why k-fold (and not one train/test split): with ~100-200 subjects, a single
80/20 split leaves a test set of ~20-40 people — small enough that the
reported accuracy can swing by 10-20 points depending on which 20% you
happened to hold out. Repeated stratified k-fold gives you a distribution of
scores instead of one noisy point estimate, which is the honest way to report
results on a dataset this size.

Usage:
    python src/train.py --data data/processed/dataset.csv --target label --epochs 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

from src.data.loaders import load_tabular_dataset, clean_features, impute_and_scale
from src.models.mlp import MECFSClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    patience: int = 15,
    lr: float = 1e-3,
) -> tuple[MECFSClassifier, dict]:
    model = MECFSClassifier(n_features=X_train.shape[1]).to(DEVICE)

    # Class weighting matters here: ME/CFS case/control counts are rarely
    # perfectly balanced, and with small N even a mild imbalance can make a
    # model lazily predict the majority class.
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(model(X_val_t)).cpu().numpy()

    metrics = {
        "val_loss": best_val_loss,
        "val_auc": roc_auc_score(y_val, val_probs) if len(set(y_val)) > 1 else float("nan"),
        "val_accuracy": accuracy_score(y_val, val_probs > 0.5),
        "epochs_trained": epoch + 1,
    }
    return model, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to processed CSV/parquet")
    parser.add_argument("--target", required=True, help="Name of the binary label column")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--out-dir", default="data/processed/model")
    args = parser.parse_args()

    X, y = load_tabular_dataset(args.data, args.target)
    X = clean_features(X)
    feature_names = list(X.columns)
    print(f"{len(feature_names)} features, {len(y)} subjects, "
          f"{y.sum()} cases / {len(y) - y.sum()} controls")

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    best_fold_model = None
    best_fold_auc = -1.0

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_scaled, X_val_scaled, imputer, scaler = impute_and_scale(
            X.iloc[train_idx], X.iloc[val_idx]
        )
        model, metrics = train_one_fold(
            X_train_scaled, y.iloc[train_idx].values,
            X_val_scaled, y.iloc[val_idx].values,
            epochs=args.epochs,
        )
        metrics["fold"] = fold_idx
        fold_metrics.append(metrics)
        print(f"Fold {fold_idx}: AUC={metrics['val_auc']:.3f} "
              f"acc={metrics['val_accuracy']:.3f} epochs={metrics['epochs_trained']}")

        if metrics["val_auc"] > best_fold_auc:
            best_fold_auc = metrics["val_auc"]
            best_fold_model = model

    aucs = [m["val_auc"] for m in fold_metrics if not np.isnan(m["val_auc"])]
    summary = {
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
        "mean_accuracy": float(np.mean([m["val_accuracy"] for m in fold_metrics])),
        "n_features": len(feature_names),
        "n_subjects": len(y),
        "fold_metrics": fold_metrics,
    }
    print(f"\nCross-validated AUC: {summary['mean_auc']:.3f} +/- {summary['std_auc']:.3f}")
    print(
        "Report that range, not just the best fold's number — a wide spread "
        "across folds is itself an important finding on a dataset this size."
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_fold_model.state_dict(), out_dir / "model.pt")
    (out_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved best-fold model + metrics to {out_dir}/")


if __name__ == "__main__":
    main()
