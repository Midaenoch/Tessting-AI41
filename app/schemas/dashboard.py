from typing import List, Optional, Dict
from pydantic import BaseModel

class KPIResponse(BaseModel):
    confirmed_cases: int
    recoveries: int
    deaths: int
    fatality_rate_percent: float
    recovery_rate_percent: float
    total_outbreaks: int
    states_affected: int
    lgas_affected: int
    total_records: int
    years_covered: List[int]
    generated_at: str
    data_sources: Dict[str, str]

class DemographicItem(BaseModel):
    category: str
    count: int
    percentage: float

class DemographicsResponse(BaseModel):
    age_groups: List[DemographicItem]
    gender: List[DemographicItem]

class LGAItem(BaseModel):
    lga: str
    state: str
    cases: int
    deaths: int
    recovery_rate_percent: float
    last_update: Optional[str] = None

class LGABreakdownResponse(BaseModel):
    results: List[LGAItem]
    total_matching_lgas: int
    data_source: str

class RecentAlert(BaseModel):
    lga: str
    state: str
    reported_date: str
    risk_level: str
    risk_color: str
    advice: str
    cases: int
    deaths: int