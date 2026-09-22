from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas import DatasetSummary

router = APIRouter(tags=["eda"])

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "dataset.csv"
TARGET_COL = "label"  # adjust to match your processed dataset


@router.get("/dataset/summary", response_model=DatasetSummary)
def dataset_summary() -> DatasetSummary:
    if not DATA_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No processed dataset found at {DATA_PATH}. Run your "
            "data prep notebook/script first.",
        )
    df = pd.read_csv(DATA_PATH)
    if TARGET_COL not in df.columns:
        raise HTTPException(
            status_code=500,
            detail=f"Expected label column '{TARGET_COL}' not found in dataset.",
        )

    features = df.drop(columns=[TARGET_COL])
    return DatasetSummary(
        n_subjects=len(df),
        n_features=features.shape[1],
        n_cases=int(df[TARGET_COL].sum()),
        n_controls=int((df[TARGET_COL] == 0).sum()),
        feature_missingness={
            col: float(features[col].isna().mean()) for col in features.columns
        },
    )
