from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from pymongo.errors import PyMongoError

from app.services import dashboard_service, outbreak_service

router = APIRouter(prefix="/api/v1", tags=["dashboard"])


@router.get("/kpi")
async def kpi():
    try:
        return await dashboard_service.get_kpi()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except PyMongoError as e:
        print(f"[db] WARNING: {e}")
        raise HTTPException(status_code=503, detail="Database temporarily unavailable. Please try again shortly.")


@router.get("/outbreaks/definition")
async def outbreaks_definition():
    return {
        "field": "total_outbreaks",
        "definition": (
            "A month is declared an outbreak if its confirmed case count is at "
            "or above that calendar month's own historical baseline "
            "(mean + k×SD, k=1.5 by default), NOT a flat year-round threshold — "
            "Lassa fever is strongly seasonal (January/February average "
            "~440-500 cases/month vs ~170-210 in May-September), so a single "
            "threshold would falsely flag most Januaries and could miss a real "
            "outbreak mid-year."
        ),
        "method": "Historical per-calendar-month statistical threshold (mean + k*SD), not a trained ML model — see GET /api/v1/outbreaks/baseline for the full reference table.",
        "status": "ACTIVE",
    }


@router.get("/outbreaks/baseline")
async def outbreaks_baseline(k: float = Query(1.5, description="Threshold multiplier: mean + k*SD")):
    """The full, auditable per-calendar-month baseline/threshold table."""
    try:
        return outbreak_service.compute_baselines(k=k).to_dict(orient="records")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/outbreaks/current")
async def outbreaks_current(k: float = Query(1.5, description="Threshold multiplier: mean + k*SD")):
    """Outbreak declaration for the most recent month in the dataset."""
    try:
        return outbreak_service.declare_latest(k=k)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/outbreaks/history")
async def outbreaks_history(k: float = Query(1.5, description="Threshold multiplier: mean + k*SD")):
    """The declaration rule applied retroactively to every month on record."""
    try:
        return outbreak_service.declaration_history(k=k)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/trends/yearly")
async def yearly():
    try:
        return await dashboard_service.get_yearly_trends()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except PyMongoError as e:
        print(f"[db] WARNING: {e}")
        raise HTTPException(status_code=503, detail="Database temporarily unavailable. Please try again shortly.")


@router.get("/trends/weekly")
async def weekly():
    try:
        return await dashboard_service.get_weekly_trends()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except PyMongoError as e:
        print(f"[db] WARNING: {e}")
        raise HTTPException(status_code=503, detail="Database temporarily unavailable. Please try again shortly.")


@router.get("/demographics")
async def demographics():
    try:
        return await dashboard_service.get_demographics()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except PyMongoError as e:
        print(f"[db] WARNING: {e}")
        raise HTTPException(status_code=503, detail="Database temporarily unavailable. Please try again shortly.")


@router.get("/lga-breakdown")
async def lga_breakdown(
    state: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    min_cases: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
):
    try:
        return await dashboard_service.get_lga_breakdown(state, search, min_cases, limit)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except PyMongoError as e:
        print(f"[db] WARNING: {e}")
        raise HTTPException(status_code=503, detail="Database temporarily unavailable. Please try again shortly.")
