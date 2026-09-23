"""Admin endpoints: dataset upload, dataset export, model retraining."""
from __future__ import annotations

import io
import csv
import json
from datetime import datetime
from typing import Any, Optional

import pandas as pd
from bson import ObjectId
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.auth.deps import require_roles
from app.database import get_db
from app.services import forecast_service

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
LINE_LIST_REQUIRED = ["YEAR", "MONTH"]  # minimal validation
LINE_LIST_OPTIONAL = [
    "AGE in Years",
    "SEX",
    "RESULTS",
    "CONDITION",
    "State",
    "LGA",
]

LGA_REQUIRED = ["LGA", "State", "Case_Status", "Outcome"]
LGA_OPTIONAL = ["Last_Update", "Patient_ID"]

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def _read_upload_to_dataframe(filename: str, raw: bytes) -> pd.DataFrame:
    """Parse CSV / JSON / XLSX into a DataFrame, case-insensitively."""
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

        # Accept either [{...}, ...] or {"records": [...]} or a single dict
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

    # Normalize column names for case-insensitive matching
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _ensure_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Case-insensitive column check. Adds missing-optional as NaN if absent."""
    lower_map = {c.lower(): c for c in df.columns}
    missing = [r for r in required if r.lower() not in lower_map]
    if missing:
        raise HTTPException(
            400,
            f"Missing required column(s): {missing}. "
            f"Found: {list(df.columns)}",
        )


def _get_col(df: pd.DataFrame, name: str) -> Optional[pd.Series]:
    """Case-insensitive column lookup."""
    for c in df.columns:
        if c.lower() == name.lower():
            return df[c]
    return None


def _to_int(v) -> Optional[int]:
    try:
        if pd.isna(v):
            return None
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _to_float(v) -> Optional[float]:
    try:
        if pd.isna(v):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _str_or_none(v) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip()
        return v or None
    return str(v)


# ---------------------------------------------------------------------------
# Row mappers
# ---------------------------------------------------------------------------
def _line_list_row(df: pd.DataFrame, idx: int) -> dict:
    return {
        "year": _to_int(_get_col(df, "YEAR").iloc[idx]) if _get_col(df, "YEAR") is not None else None,
        "month": (
            _to_int(_get_col(df, "MONTH").iloc[idx])
            if _get_col(df, "MONTH") is not None
            else (_get_col(df, "MONTH").iloc[idx] if _get_col(df, "MONTH") is not None else None)
        ),
        "age": _to_float(_get_col(df, "AGE in Years").iloc[idx])
        if _get_col(df, "AGE in Years") is not None
        else None,
        "sex": _str_or_none(_get_col(df, "SEX").iloc[idx])
        if _get_col(df, "SEX") is not None
        else None,
        "result": _str_or_none(_get_col(df, "RESULTS").iloc[idx])
        if _get_col(df, "RESULTS") is not None
        else None,
        "condition": _str_or_none(_get_col(df, "CONDITION").iloc[idx])
        if _get_col(df, "CONDITION") is not None
        else None,
        "state": _str_or_none(_get_col(df, "State").iloc[idx])
        if _get_col(df, "State") is not None
        else None,
        "lga": _str_or_none(_get_col(df, "LGA").iloc[idx])
        if _get_col(df, "LGA") is not None
        else None,
        "source": "upload",
        "uploaded_at": datetime.utcnow(),
    }


def _lga_synthetic_row(df: pd.DataFrame, idx: int) -> dict:
    def get(name: str):
        col = _get_col(df, name)
        return col.iloc[idx] if col is not None else None

    return {
        "lga": _str_or_none(get("LGA")),
        "state": _str_or_none(get("State")),
        "case_status": _str_or_none(get("Case_Status")),
        "outcome": _str_or_none(get("Outcome")),
        "last_update": _str_or_none(get("Last_Update")),
        "patient_id": _str_or_none(get("Patient_ID")),
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
    """Upload a CSV/JSON/XLSX dataset.

    upload_type:
      - "outbreak_data"  -> inserts into `cases` collection
      - "lga_breakdown"  -> inserts into `lga_synthetic` collection

    If replace_existing=True, the target collection is cleared first.
    If retrain_model=True and upload_type=="outbreak_data", the forecast
    model is retrained after insertion.
    """
    db = get_db()
    if db is None:
        raise HTTPException(503, "Database not connected")

    if upload_type not in ("outbreak_data", "lga_breakdown"):
        raise HTTPException(400, f"Unknown upload_type: {upload_type}")

    # --- Read and size-check ------------------------------------------------
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413,
            f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit",
        )

    df = _read_upload_to_dataframe(file.filename or "", raw)

    # --- Map rows -----------------------------------------------------------
    if upload_type == "outbreak_data":
        _ensure_columns(df, LINE_LIST_REQUIRED)
        target = db.cases
        docs = [_line_list_row(df, i) for i in range(len(df))]
    else:
        _ensure_columns(df, LGA_REQUIRED)
        target = db.lga_synthetic
        docs = [_lga_synthetic_row(df, i) for i in range(len(df))]

    # Drop rows missing essential fields
    if upload_type == "outbreak_data":
        docs = [d for d in docs if d["year"] is not None]
    else:
        docs = [d for d in docs if d["lga"] and d["state"]]
    if not docs:
        raise HTTPException(400, "No valid rows after parsing")

    # --- Write to Mongo -----------------------------------------------------
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

    # --- Optional: retrain forecast ----------------------------------------
    retrain_report: dict[str, Any] = {"attempted": False, "success": False}
    if retrain_model and upload_type == "outbreak_data":
        retrain_report["attempted"] = True
        try:
            # forecast_service.retrain is defined in the next section
            result = await forecast_service.retrain(db)
            retrain_report["success"] = True
            retrain_report["metrics"] = result
        except Exception as e:
            retrain_report["success"] = False
            retrain_report["error"] = str(e)

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
        "uploaded_at": datetime.utcnow().isoformat() + "Z",
    }


# ---------------------------------------------------------------------------
# GET /export/{kind}  — CSV streaming (used by the download cards)
# ---------------------------------------------------------------------------
@router.get("/export/{kind}")
async def export_dataset(
    kind: str,
    _user: dict = Depends(require_roles("admin")),
):
    """Stream a CSV export. Admin only.

    kind:
      - "full_dataset"   -> all case records
      - "filtered_data"  -> confirmed-only case records
      - "research_data"  -> anonymized (no age/sex) case records
      - "public_reports" -> LGA-level summary
    """
    db = get_db()
    if db is None:
        raise HTTPException(503, "Database not connected")

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
        headers = list(projection.keys())
        headers.remove("_id")

        async def iter_rows():
            async for doc in cursor:
                yield [doc.get(h, "") for h in headers]

    filename = f"ai4lassa_{kind}_{datetime.utcnow().strftime('%Y%m%d')}.csv"

    async def stream():
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(headers)
        yield buf.getvalue()
        buf.seek(0)
        buf.truncate(0)

        async for row in iter_rows():
            writer.writerow(row)
            yield buf.getvalue()
            buf.seek(0)
            buf.truncate(0)

    return StreamingResponse(
        stream(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )