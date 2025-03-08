from fastapi import APIRouter, HTTPException
from ..schemas.input_schemas import PredictionInput
from ..models.ml_models import predict


router = APIRouter()


@router.post("/predict/")
def make_prediction(input_data: PredictionInput):
    try:
        result = predict(input_data.features)
        return {"predicition": int(result)}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )
