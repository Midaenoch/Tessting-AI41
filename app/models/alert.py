from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

class AlertInDB(BaseModel):
    lga: str
    state: str
    cases: int
    deaths: int
    risk_level: str
    advice: str
    reported_date: datetime = Field(default_factory=datetime.utcnow)