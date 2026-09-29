from typing import List, Optional
from pydantic import BaseModel, Field


class PatientInput(BaseModel):
    age: float = Field(..., ge=0, le=110, description="Age in years")
    sex: str = Field(..., description="'M'/'male' or 'F'/'female'")

    general_weakness: bool = False
    fever: bool = False
    cough: bool = False
    abdominal_pain: bool = False
    headache: bool = False
    chest_pain: bool = False
    nausea: bool = False
    vomiting: bool = False
    diarrhoea: bool = False
    difficulty_breathing: bool = False
    catarrh: bool = False
    sore_throat: bool = False
    joint_pain: bool = False
    red_eyes: bool = False
    jaundice: bool = False
    oedema: bool = False
    bleeding_injection_site: bool = False
    bleeding_nose: bool = False
    bleeding_gums: bool = False
    unexplained_bleeding: bool = False
    coke_coloured_urine: bool = False
    blood_in_faeces: bool = False
    seizure: bool = False
    deafness: bool = False
    photophobia: bool = False
    hiccups: bool = False
    skin_rash: bool = False
    coma: bool = False
    confused: bool = False
    chills: bool = False

    decision_threshold: Optional[float] = Field(0.5, ge=0, le=1)


class PatientPredictionResponse(BaseModel):
    mortality_risk_probability: float
    risk_level: str
    decision_threshold_used: float
    top_contributing_factors: List[str]
    explanation_available: bool
    model_trained_at: Optional[str] = None
    model_n_samples: Optional[int] = None
    note: str
