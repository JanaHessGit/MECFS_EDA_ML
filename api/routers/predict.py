from fastapi import APIRouter, Depends, HTTPException

from api.model_registry import ModelBundle, get_model_bundle
from api.schemas import ModelMetadata, PredictRequest, PredictResponse

router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictResponse)
def predict(
    request: PredictRequest,
    bundle: ModelBundle = Depends(get_model_bundle),
) -> PredictResponse:
    try:
        result = bundle.predict(request.features)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return PredictResponse(**result)


@router.get("/model/metadata", response_model=ModelMetadata)
def model_metadata(bundle: ModelBundle = Depends(get_model_bundle)) -> ModelMetadata:
    m = bundle.metrics
    return ModelMetadata(
        n_features=m["n_features"],
        n_subjects_trained_on=m["n_subjects"],
        mean_cv_auc=m["mean_auc"],
        std_cv_auc=m["std_auc"],
        mean_cv_accuracy=m["mean_accuracy"],
    )
