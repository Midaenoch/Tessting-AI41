from fastapi import APIRouter, HTTPException
from app.schemas.patient import PatientInput, PatientPredictionResponse
from app.services import patient_service

router = APIRouter(prefix="/patient", tags=["patient"])


@router.post("/predict", response_model=PatientPredictionResponse)
def predict(patient: PatientInput):
    data = patient.model_dump()
    threshold = data.pop("decision_threshold", 0.5) or 0.5
    try:
        return patient_service.predict(data, decision_threshold=threshold)
    except patient_service.InvalidPatientInput as e:
        raise HTTPException(status_code=400, detail=e.message)
    except patient_service.PredictionError as e:
        raise HTTPException(status_code=503, detail=e.user_message)
