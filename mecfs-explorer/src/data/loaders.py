"""
Shared loading / cleaning utilities. The notebook and src/train.py both import
from here so your preprocessing logic lives in exactly one place.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def load_tabular_dataset(path: str | Path, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load a processed CSV/parquet file with one row per subject, one column
    per feature, plus a label column.

    Expects the label column to be binary (e.g. 0 = control, 1 = case).
    Raises a clear error if it isn't, since a silently-multiclass or
    silently-continuous target is a common source of confusing bugs.
    """
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' not found in columns: {list(df.columns)}")

    y = df[target_col]
    unique_vals = sorted(y.dropna().unique())
    if unique_vals not in ([0, 1], [False, True]):
        raise ValueError(
            f"Expected a binary target in {{0, 1}}, got values: {unique_vals}. "
            "Recode your label column before loading."
        )

    X = df.drop(columns=[target_col])
    return X, y.astype(int)


def clean_features(
    X: pd.DataFrame,
    max_missing_frac: float = 0.4,
) -> pd.DataFrame:
    """
    Drop features missing in more than `max_missing_frac` of subjects, then
    keep only numeric columns. Omics datasets routinely have batches of
    features that are mostly missing (below detection limit, dropout, etc.)
    — dropping them explicitly, and logging what you dropped, beats silently
    imputing garbage.
    """
    numeric = X.select_dtypes(include=[np.number]).copy()
    dropped_non_numeric = set(X.columns) - set(numeric.columns)
    if dropped_non_numeric:
        print(f"Dropped non-numeric columns: {sorted(dropped_non_numeric)}")

    missing_frac = numeric.isna().mean()
    keep_cols = missing_frac[missing_frac <= max_missing_frac].index
    dropped_missing = set(numeric.columns) - set(keep_cols)
    if dropped_missing:
        print(
            f"Dropped {len(dropped_missing)} columns with >"
            f"{max_missing_frac:.0%} missing values"
        )
    return numeric[keep_cols]


def impute_and_scale(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray | None, SimpleImputer, StandardScaler]:
    """
    Fit imputer + scaler on the TRAINING fold only, then apply to both train
    and test. Fitting these on the full dataset before splitting is a classic
    small-N leakage bug — it lets information about the test fold's
    distribution bleed into training, and it inflates cross-validated scores
    without you noticing.
    """
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_imp = imputer.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train_imp)

    X_test_scaled = None
    if X_test is not None:
        X_test_imp = imputer.transform(X_test)
        X_test_scaled = scaler.transform(X_test_imp)

    return X_train_scaled, X_test_scaled, imputer, scaler
