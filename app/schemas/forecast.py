from typing import List, Optional
from pydantic import BaseModel, Field

class ForecastResponse(BaseModel):
    based_on_month: str
    forecast_next_month_cases: int
    outbreak_probability_provisional: float
    risk_level_provisional: str
    decision_threshold_used: float
    top_contributing_factors: List[str]
    note: str

class ForecastFeatures(BaseModel):
    case_count: float
    case_count_lag1: float
    case_count_lag2: float
    case_count_lag3: float
    case_count_lag6: float
    case_count_lag12: float
    case_count_roll3_mean: float
    case_count_roll6_mean: float
    case_count_roll3_max: float
    case_growth_lag1: float
    positivity_rate_lag1: float = Field(..., ge=0, le=1)
    month_sin: float
    month_cos: float
    year: int
    decision_threshold: Optional[float] = Field(0.5, ge=0, le=1)