"""Admin endpoints: dataset upload, dataset export."""
from __future__ import annotations

import csv
import io
import json
import os
from datetime import datetime
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pymongo.errors import PyMongoError

from app.auth.deps import require_roles
from app.database import get_db
from app.services import forecast_service, patient_service

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LINE_LIST_REQUIRED = ["YEAR", "MONTH"]
LGA_REQUIRED = ["LGA", "State", "Case_Status", "Outcome"]

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def _read_upload_to_dataframe(filename: str, raw: bytes) -> pd.DataFrame:
    name = filename.lower()

    if name.endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(raw))
        except Exception as e:
            raise HTTPException(400, f"Failed to parse CSV: {e}")

    elif name.endswith(".json"):
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception as e:
            raise HTTPException(400, f"Failed to parse JSON: {e}")

        if isinstance(data, dict) and "records" in data:
            data = data["records"]
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            raise HTTPException(400, "JSON must be an array of records")
        df = pd.DataFrame(data)

    elif name.endswith((".xlsx", ".xls")):
        try:
            df = pd.read_excel(io.BytesIO(raw), engine="openpyxl")
        except Exception as e:
            raise HTTPException(400, f"Failed to parse Excel: {e}")

    else:
        raise HTTPException(400, "Unsupported file type. Use CSV, JSON, or XLSX.")

    if df.empty:
        raise HTTPException(400, "Uploaded file contains no rows")

    df.columns = [str(c).strip() for c in df.columns]
    return df


def _get_col(df: pd.DataFrame, name: str):
    for c in df.columns:
        if c.lower() == name.lower():
            return df[c]
    return None


def _ensure_columns(df: pd.DataFrame, required: list[str]) -> None:
    lower = {c.lower() for c in df.columns}
    missing = [r for r in required if r.lower() not in lower]
    if missing:
        raise HTTPException(
            400,
            f"Missing required column(s): {missing}. Found: {list(df.columns)}",
        )


def _to_int(v):
    try:
        if pd.isna(v):
            return None
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _to_float(v):
    try:
        if pd.isna(v):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _str_or_none(v):
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip()
        return v or None
    return str(v)


def _row_from(df: pd.DataFrame, idx: int, mapping: dict[str, str]) -> dict:
    out = {}
    for target, source in mapping.items():
        col = _get_col(df, source)
        out[target] = col.iloc[idx] if col is not None else None
    return out


# ---------------------------------------------------------------------------
# Row mappers
# ---------------------------------------------------------------------------
def _line_list_row(df: pd.DataFrame, idx: int) -> dict:
    r = _row_from(df, idx, {
        "year": "YEAR", "month": "MONTH", "age": "AGE in Years",
        "sex": "SEX", "result": "RESULTS", "condition": "CONDITION",
        "state": "State", "lga": "LGA",
    })
    doc = {
        "year": _to_int(r["year"]),
        "month": r["month"] if not pd.isna(r["month"]) else None,
        "age": _to_float(r["age"]),
        "sex": _str_or_none(r["sex"]),
        "result": _str_or_none(r["result"]),
        "condition": _str_or_none(r["condition"]),
        "state": _str_or_none(r["state"]),
        "lga": _str_or_none(r["lga"]),
        "source": "upload",
        "uploaded_at": datetime.utcnow(),
    }
    # Presenting-symptom columns (optional — only present if the uploaded
    # file has them; needed for app.services.patient_service.retrain()).
    symptom_mapping = {feat: raw_col for raw_col, feat in patient_service.SYMPTOM_COLUMNS.items()}
    symptom_r = _row_from(df, idx, symptom_mapping)
    for feat, val in symptom_r.items():
        doc[feat] = patient_service._clean_yes_no(val) if val is not None and not pd.isna(val) else 0
    return doc


def _lga_synthetic_row(df: pd.DataFrame, idx: int) -> dict:
    r = _row_from(df, idx, {
        "lga": "LGA", "state": "State", "case_status": "Case_Status",
        "outcome": "Outcome", "last_update": "Last_Update",
        "patient_id": "Patient_ID",
    })
    return {
        "lga": _str_or_none(r["lga"]),
        "state": _str_or_none(r["state"]),
        "case_status": _str_or_none(r["case_status"]),
        "outcome": _str_or_none(r["outcome"]),
        "last_update": _str_or_none(r["last_update"]),
        "patient_id": _str_or_none(r["patient_id"]),
        "source": "upload",
        "uploaded_at": datetime.utcnow(),
    }


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------
@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    upload_type: str = Form(...),
    replace_existing: bool = Form(False),
    retrain_model: bool = Form(False),
    _user: dict = Depends(require_roles("admin", "health_officer")),
):
    db = get_db()
    if db is None:
        raise HTTPException(503, "Database not connected")

    if upload_type not in ("outbreak_data", "lga_breakdown"):
        raise HTTPException(400, f"Unknown upload_type: {upload_type}")

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413, f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit"
        )

    df = _read_upload_to_dataframe(file.filename or "", raw)

    if upload_type == "outbreak_data":
        _ensure_columns(df, LINE_LIST_REQUIRED)
        target = db.cases
        docs = [_line_list_row(df, i) for i in range(len(df))]
        docs = [d for d in docs if d["year"] is not None]
    else:
        _ensure_columns(df, LGA_REQUIRED)
        target = db.lga_synthetic
        docs = [_lga_synthetic_row(df, i) for i in range(len(df))]
        docs = [d for d in docs if d["lga"] and d["state"]]

    if not docs:
        raise HTTPException(400, "No valid rows after parsing")

    deleted_count = 0
    if replace_existing:
        del_result = await target.delete_many({})
        deleted_count = del_result.deleted_count

    inserted = 0
    CHUNK = 5000
    for i in range(0, len(docs), CHUNK):
        chunk = docs[i : i + CHUNK]
        res = await target.insert_many(chunk, ordered=False)
        inserted += len(res.inserted_ids)

    retrain_report: dict[str, Any] = {"attempted": False, "success": False}
    patient_retrain_report: dict[str, Any] = {"attempted": False, "success": False}
    if retrain_model and upload_type == "outbreak_data":
        retrain_report["attempted"] = True
        try:
            result = await forecast_service.retrain(db, triggered_by=_user)
            retrain_report["success"] = True
            retrain_report["metrics"] = result
        except Exception as e:
            retrain_report["success"] = False
            retrain_report["error"] = str(e)

        patient_retrain_report["attempted"] = True
        try:
            p_result = await patient_service.retrain(db, triggered_by=_user)
            patient_retrain_report["success"] = True
            patient_retrain_report["metrics"] = p_result
        except Exception as e:
            patient_retrain_report["success"] = False
            patient_retrain_report["error"] = str(e)

    return {
        "status": "ok",
        "upload_type": upload_type,
        "filename": file.filename,
        "rows_parsed": len(df),
        "rows_inserted": inserted,
        "rows_rejected": len(df) - len(docs),
        "replaced_existing": replace_existing,
        "deleted_count": deleted_count,
        "retrain": retrain_report,
        "patient_retrain": patient_retrain_report,
        "uploaded_at": datetime.utcnow().isoformat() + "Z",
    }


# ---------------------------------------------------------------------------
# GET /export/{kind}  — open to admin AND health_officer
# ---------------------------------------------------------------------------
@router.get("/export/{kind}")
async def export_dataset(
    kind: str,
    _user: dict = Depends(require_roles("admin", "health_officer")),
):
    db = get_db()
    if db is None:
        raise HTTPException(503, "Database not connected")
    try:
        # Cheap upfront check: once the StreamingResponse below starts, HTTP
        # headers are already sent and the status code can no longer change,
        # so a Mongo failure would otherwise look like a silently truncated
        # download instead of a clean error.
        await db.command("ping")
    except PyMongoError as e:
        print(f"[db] WARNING: {e}")
        raise HTTPException(503, "Database temporarily unavailable. Please try again shortly.")

    if kind == "public_reports":
        pipeline = [
            {"$match": {"case_status": {"$regex": r"^\s*confirmed\s*$", "$options": "i"}}},
            {"$group": {
                "_id": {"state": "$state", "lga": "$lga"},
                "cases": {"$sum": 1},
                "deaths": {"$sum": {"$cond": [
                    {"$regexMatch": {
                        "input": {"$ifNull": ["$outcome", ""]},
                        "regex": r"^\s*deceased\s*$",
                        "options": "i",
                    }}, 1, 0,
                ]}},
            }},
            {"$sort": {"cases": -1}},
        ]
        cursor = db.lga_synthetic.aggregate(pipeline)
        headers = ["state", "lga", "cases", "deaths"]

        async def iter_rows():
            async for doc in cursor:
                yield [
                    doc["_id"].get("state", ""),
                    doc["_id"].get("lga", ""),
                    doc.get("cases", 0),
                    doc.get("deaths", 0),
                ]
    else:
        filt: dict = {}
        projection: dict = {"_id": 0}
        if kind == "filtered_data":
            filt["result"] = {"$regex": "positive", "$options": "i"}
            projection.update({
                "year": 1, "month": 1, "age": 1, "sex": 1,
                "result": 1, "condition": 1, "state": 1, "lga": 1,
            })
        elif kind == "research_data":
            projection.update({
                "year": 1, "month": 1, "result": 1, "condition": 1,
                "state": 1, "lga": 1,
            })
        elif kind == "full_dataset":
            projection.update({
                "year": 1, "month": 1, "age": 1, "sex": 1,
                "result": 1, "condition": 1, "state": 1, "lga": 1,
            })
        else:
            raise HTTPException(400, f"Unknown export kind: {kind}")

        cursor = db.cases.find(filt, projection)
        headers = [k for k in projection.keys() if k != "_id"]

        async def iter_rows():
            async for doc in cursor:
                yield [doc.get(h, "") for h in headers]

    filename = f"ai4lassa_{kind}_{datetime.utcnow().strftime('%Y%m%d')}.csv"

    async def stream():
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(headers)
        yield buf.getvalue()
        buf.seek(0); buf.truncate(0)

        async for row in iter_rows():
            writer.writerow(row)
            yield buf.getvalue()
            buf.seek(0); buf.truncate(0)

    return StreamingResponse(
        stream(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# GET /model-versions  — list archived model generations
# ---------------------------------------------------------------------------
@router.get("/model-versions")
async def list_model_versions(
    _user: dict = Depends(require_roles("admin", "health_officer")),
):
    versions = forecast_service.list_archived_versions(BASE_DIR)
    return {
        "current_config": forecast_service.get_config_summary(),
        "archived_versions": versions,
    }


# ---------------------------------------------------------------------------
# GET /model-history  — audit log of retrain/rollback events
# ---------------------------------------------------------------------------
@router.get("/model-history")
async def model_history(
    limit: int = 50,
    _user: dict = Depends(require_roles("admin", "health_officer")),
):
    db = get_db()
    events = await forecast_service.get_model_history(db, limit=limit)
    return {"events": events}


# ---------------------------------------------------------------------------
# POST /rollback  — restore an archived model generation as the live one
# ---------------------------------------------------------------------------
@router.post("/rollback")
async def rollback_model(
    version: str | None = Form(None),
    _user: dict = Depends(require_roles("admin")),
):
    """Roll back to a previous model generation.

    Body form field `version` is optional — omit it to roll back to the most
    recently archived generation (i.e. undo the last retrain/promotion).
    """
    db = get_db()
    try:
        result = await forecast_service.rollback_to_version(
            BASE_DIR, version, db=db, triggered_by=_user
        )
    except RuntimeError as e:
        raise HTTPException(400, str(e))

    return {
        "status": "ok",
        "rolled_back_to": result["rolled_back_to"],
        "available_versions": result["available_versions"],
        "rolled_back_at": datetime.utcnow().isoformat() + "Z",
    }


# ---------------------------------------------------------------------------
# GET /patient/model-versions  — list archived PATIENT model generations
# ---------------------------------------------------------------------------
@router.get("/patient/model-versions")
async def list_patient_model_versions(
    _user: dict = Depends(require_roles("admin", "health_officer")),
):
    versions = patient_service.list_archived_versions(BASE_DIR)
    return {
        "current_config": patient_service.get_config_summary(),
        "archived_versions": versions,
    }


# ---------------------------------------------------------------------------
# POST /patient/rollback  — restore an archived PATIENT model generation
# ---------------------------------------------------------------------------
@router.post("/patient/rollback")
async def rollback_patient_model(
    version: str | None = Form(None),
    _user: dict = Depends(require_roles("admin")),
):
    db = get_db()
    try:
        result = await patient_service.rollback_to_version(
            BASE_DIR, version, db=db, triggered_by=_user
        )
    except RuntimeError as e:
        raise HTTPException(400, str(e))

    return {
        "status": "ok",
        "rolled_back_to": result["rolled_back_to"],
        "available_versions": result["available_versions"],
        "rolled_back_at": datetime.utcnow().isoformat() + "Z",
    }
