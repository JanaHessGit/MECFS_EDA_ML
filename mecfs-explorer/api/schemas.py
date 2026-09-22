from __future__ import annotations

from pydantic import BaseModel, Field, ConfigDict


class PredictRequest(BaseModel):
    """
    Feature values keyed by name, matching data/processed/model/feature_names.json.
    Using a dict (rather than positional floats) means the API validates that
    every expected feature is present and rejects typos in feature names
    loudly, instead of silently misaligning columns.
    """

    model_config = ConfigDict(json_schema_extra={
        "example": {"features": {"feature_1": 0.42, "feature_2": -1.1}}
    })

    features: dict[str, float] = Field(
        ..., description="Feature name -> value, matching the model's trained feature set"
    )


class FeatureContribution(BaseModel):
    feature: str
    value: float
    contribution: float = Field(..., description="Signed contribution to the prediction")


class PredictResponse(BaseModel):
    predicted_class: int = Field(..., description="0 = control-like, 1 = case-like")
    probability: float = Field(..., ge=0.0, le=1.0)
    top_contributing_features: list[FeatureContribution]
    disclaimer: str = (
        "Research/learning artifact only. Not a diagnostic tool. Trained on a "
        "small, specific study cohort and should not be interpreted as a "
        "general ME/CFS classifier."
    )


class DatasetSummary(BaseModel):
    n_subjects: int
    n_features: int
    n_cases: int
    n_controls: int
    feature_missingness: dict[str, float]


class ModelMetadata(BaseModel):
    n_features: int
    n_subjects_trained_on: int
    mean_cv_auc: float
    std_cv_auc: float
    mean_cv_accuracy: float
    disclaimer: str = (
        "Cross-validated performance on one small cohort. Do not extrapolate "
        "to other populations or treat as clinically validated."
    )
