from typing import Optional
from fastapi import APIRouter, HTTPException, Query

from app.services import dashboard_service

router = APIRouter(prefix="/api/v1", tags=["dashboard"])


@router.get("/kpi")
async def kpi():
    try:
        return await dashboard_service.get_kpi()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/outbreaks/definition")
async def outbreaks_definition():
    return {
        "field": "total_outbreaks",
        "definition": "Count of distinct (year, month) pairs with >= 1 confirmed case.",
        "status": "PROVISIONAL",
    }


@router.get("/trends/yearly")
async def yearly():
    try:
        return await dashboard_service.get_yearly_trends()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/trends/weekly")
async def weekly():
    try:
        return await dashboard_service.get_weekly_trends()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/demographics")
async def demographics():
    try:
        return await dashboard_service.get_demographics()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


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