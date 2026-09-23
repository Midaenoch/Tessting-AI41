from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.database import get_db
from app.services import dashboard_service

router = APIRouter(prefix="/alerts", tags=["alerts"])


class AlertIn(BaseModel):
    lga: str
    state: str
    reported_date: Optional[str] = None
    cases: int = 0
    deaths: int = 0


@router.get("/recent")
async def recent(limit: int = Query(3, ge=1, le=50)):
    try:
        return await dashboard_service.get_recent_alerts(limit)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("")
async def create_alert(payload: AlertIn):
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected")

    rd = payload.reported_date
    if rd:
        try:
            rd_dt = datetime.fromisoformat(rd.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid reported_date format")
    else:
        rd_dt = datetime.utcnow()

    doc = {
        "lga": payload.lga,
        "state": payload.state,
        "reported_date": rd_dt,
        "cases": payload.cases,
        "deaths": payload.deaths,
    }
    result = await db.alerts.insert_one(doc)
    return {"id": str(result.inserted_id), "status": "created"}