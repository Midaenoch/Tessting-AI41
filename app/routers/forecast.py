from fastapi import APIRouter, HTTPException
from app.schemas.forecast import ForecastResponse, ForecastFeatures
from app.services import forecast_service

router = APIRouter(prefix="/forecast", tags=["forecast"])

@router.get("/latest", response_model=ForecastResponse)
def latest(decision_threshold: float = 0.5):
    try:
        return forecast_service.predict_latest(decision_threshold)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

@router.post("/predict", response_model=ForecastResponse)
def predict(features: ForecastFeatures):
    try:
        data = features.model_dump()
        return forecast_service.predict_features(data, data.get("decision_threshold"))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))