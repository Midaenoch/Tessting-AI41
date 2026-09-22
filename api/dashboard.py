"""
AI4Lassa dashboard endpoints: KPI strip, yearly + weekly trend charts,
demographics, and LGA breakdown — as specified in AI4Lassa_Dashboard_API_Spec.docx.

Two source files, loaded once at startup and cached in memory:
  - data/raw/2015-2025__3_.xlsx                       (real national line-list)
  - data/raw/lga_breakdown_PLACEHOLDER_synthetic.csv   (synthetic — State/LGA
    is not present anywhere in the real dataset, so /lga-breakdown runs off
    this placeholder until real geo-tagged data is available. Swap the file
    referenced in `load_data()` for a real one the moment it exists — no
    other code needs to change.)

Run locally:
    uvicorn api.main:app --reload --port 8000
    curl http://localhost:8000/api/v1/kpi
"""
import os
import re
from datetime import datetime, date
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api/v1", tags=["dashboard"])

_data = {}  # populated by load_data() at app startup

LINE_LIST_COLUMNS = ["YEAR", "MONTH", "AGE in Years", "SEX", "RESULTS", "CONDITION"]

# The MONTH column in the source spreadsheet has been observed containing
# full month names ("January", "february", etc), not just numbers — this
# maps either case to a 1-12 int. See _parse_month().
_MONTH_NAME_TO_NUM = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_data(base_dir: str):
    """Reads both source files once and caches derived, cleaned columns.
    Called from api/main.py's startup event. Raises on failure so a bad
    deploy fails loudly at boot instead of 500ing on first request."""
    line_list_path = os.path.join(base_dir, "data", "raw", "2015-2025__3_.xlsx")
    lga_path = os.path.join(
        base_dir, "data", "raw", "lga_breakdown_PLACEHOLDER_synthetic.csv"
    )

    if not os.path.exists(line_list_path):
        _data["error"] = f"Line-list file not found: {line_list_path}"
        return
    if not os.path.exists(lga_path):
        _data["error"] = f"LGA placeholder file not found: {lga_path}"
        return

    try:
        df = pd.read_excel(
            line_list_path, sheet_name=0, usecols=LINE_LIST_COLUMNS, engine="openpyxl"
        )
        df["confirmed"] = df["RESULTS"].map(_is_confirmed)
        df["condition_clean"] = df["CONDITION"].map(_clean_condition)
        df["gender_clean"] = df["SEX"].map(_clean_sex)
        df["age_bucket"] = df["AGE in Years"].map(_age_bucket)
        _data["line_list"] = df

        lga_df = pd.read_csv(lga_path)
        _data["lga"] = lga_df

        _data["error"] = None
    except Exception as e:
        _data["error"] = f"Failed to load/process source data: {e}"


def _require_data():
    if _data.get("error"):
        raise HTTPException(status_code=503, detail=_data["error"])
    if "line_list" not in _data:
        raise HTTPException(
            status_code=503,
            detail="Dashboard data not loaded yet — check server startup logs.",
        )


# ---------------------------------------------------------------------------
# Cleaning helpers — the source spreadsheet has inconsistent casing/spacing
# ---------------------------------------------------------------------------
def _is_confirmed(val) -> bool:
    if not isinstance(val, str):
        return False
    return "positive" in val.strip().lower()


def _clean_condition(val) -> Optional[str]:
    if not isinstance(val, str):
        return None
    v = val.strip().lower()
    if v.startswith("dead"):
        return "dead"
    if v.startswith("alive"):
        return "alive"
    return None


def _clean_sex(val) -> str:
    if not isinstance(val, str):
        return "Other/Unknown"
    v = val.strip().lower()
    if v == "m":
        return "Male"
    if v == "f":
        return "Female"
    return "Other/Unknown"


def _age_bucket(age) -> Optional[str]:
    if pd.isna(age):
        return None
    try:
        age = float(age)
    except (TypeError, ValueError):
        return None
    if age < 0 or age > 120:
        return None  # guard against bad data entry
    if age <= 14:
        return "0-14"
    if age <= 49:
        return "15-49"
    return "50+"


def _parse_month(val) -> Optional[int]:
    """The MONTH column has been observed holding both numeric values
    (1-12, or '1'/'01' as strings) and full month-name strings
    ('January', 'february', etc). Normalizes either form to a 1-12 int,
    or returns None if the value can't be parsed — rows with an
    unparseable month are dropped by _month_to_ts() rather than crashing
    the endpoint."""
    if pd.isna(val):
        return None
    if isinstance(val, str):
        key = val.strip().lower()
        if key in _MONTH_NAME_TO_NUM:
            return _MONTH_NAME_TO_NUM[key]
        try:
            val = float(key)
        except ValueError:
            return None
    try:
        month = int(val)
    except (TypeError, ValueError):
        return None
    return month if 1 <= month <= 12 else None


def _pct(part: int, whole: int) -> float:
    return round((part / whole) * 100, 1) if whole else 0.0


def _count_total_outbreaks(confirmed_df: pd.DataFrame) -> int:
    """See /api/v1/outbreaks/definition for what this counts and why."""
    pairs = confirmed_df.dropna(subset=["YEAR", "MONTH"])[["YEAR", "MONTH"]]
    return int(pairs.drop_duplicates().shape[0])


def _month_to_ts(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a `month_ts` column derived from YEAR+MONTH. Drops rows where
    either is missing or unparseable (including MONTH values that are
    neither a valid number nor a recognized month name — see
    _parse_month()). Returns a new DataFrame (no mutation)."""
    out = df.dropna(subset=["YEAR"]).copy()
    out["month_num"] = out["MONTH"].map(_parse_month)
    out = out.dropna(subset=["month_num"])
    out["month_ts"] = pd.to_datetime(
        dict(
            year=out["YEAR"].astype(int),
            month=out["month_num"].astype(int),
            day=1,
        ),
        errors="coerce",
    )
    return out.dropna(subset=["month_ts"]).drop(columns=["month_num"])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.get("/kpi")
def get_kpi():
    """Powers the 8 KPI cards. See spec doc for exact field definitions.

    Two fields — states_affected and lgas_affected — are computed from the
    PLACEHOLDER LGA dataset (see /lga-breakdown), not the real line-list,
    because the real dataset has no location field. They will not add up
    against confirmed_cases/deaths, which ARE from real data. Check
    `data_sources` in the response before wiring any field to the UI.
    """
    _require_data()
    df = _data["line_list"]
    confirmed_df = df[df["confirmed"]]

    confirmed_cases = len(confirmed_df)
    deaths = int((confirmed_df["condition_clean"] == "dead").sum())
    recoveries = int((confirmed_df["condition_clean"] == "alive").sum())
    total_outbreaks = _count_total_outbreaks(confirmed_df)

    lga_df = _data["lga"]
    lga_confirmed = lga_df[lga_df["Case_Status"] == "Confirmed"]
    states_affected = int(lga_confirmed["State"].nunique())
    lgas_affected = int(lga_confirmed["LGA"].nunique())

    return {
        "confirmed_cases": confirmed_cases,
        "recoveries": recoveries,
        "deaths": deaths,
        "fatality_rate_percent": _pct(deaths, confirmed_cases),
        "recovery_rate_percent": _pct(recoveries, confirmed_cases),
        "total_outbreaks": total_outbreaks,
        "states_affected": states_affected,
        "lgas_affected": lgas_affected,
        "total_records": len(df),
        "years_covered": sorted(int(y) for y in df["YEAR"].dropna().unique()),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "data_sources": {
            "confirmed_cases": "real (2015-2025__3_.xlsx)",
            "recoveries": "real (2015-2025__3_.xlsx)",
            "deaths": "real (2015-2025__3_.xlsx)",
            "total_outbreaks": "derived from real data — see /api/v1/outbreaks/definition",
            "states_affected": "PLACEHOLDER synthetic data — not real, see /api/v1/lga-breakdown",
            "lgas_affected": "PLACEHOLDER synthetic data — not real, see /api/v1/lga-breakdown",
        },
    }


@router.get("/outbreaks/definition")
def get_outbreaks_definition():
    """Explains what total_outbreaks in /kpi actually counts, since a raw
    line-list of individual cases doesn't have a built-in definition of
    'one outbreak'. Swap this definition (and _count_total_outbreaks below)
    for an NCDC/official one the moment it's provided — nothing else in
    the API depends on how this number is computed."""
    return {
        "field": "total_outbreaks",
        "definition": (
            "Count of distinct (year, month) pairs that contain at least one "
            "confirmed case. E.g. confirmed cases in both March 2021 and "
            "March 2022 count as 2 separate outbreak-months, not 1."
        ),
        "status": "PROVISIONAL — no official NCDC/proposal definition was supplied. "
        "Replace with the real definition as soon as one exists.",
    }


@router.get("/trends/yearly")
def get_yearly_trends():
    """Powers the yearly outbreak trend line chart."""
    _require_data()
    df = _data["line_list"]
    confirmed_df = df[df["confirmed"]]

    counts = (
        confirmed_df.groupby("YEAR").size().reindex(sorted(df["YEAR"].dropna().unique()), fill_value=0)
    )
    return [
        {"year": int(year), "confirmed_cases": int(count)}
        for year, count in counts.items()
    ]


@router.get("/trends/weekly")
def get_weekly_trends():
    """Powers the weekly outbreak trend line chart.

    NOTE: The raw line-list only has YEAR + MONTH (no week column), so this
    aggregates confirmed cases by ISO year-week approximated from year+month
    (day=1 of each month). Every confirmed case in a given month lands in the
    ISO week that contains that month's 1st. MONTH values are normalized via
    _parse_month() since the source data mixes numeric months and month
    names. Replace with real weekly data if/when it exists — the response
    shape is what the frontend expects.

    Response items: { year_week, total_cases, confirmed_cases, deaths, recoveries }
    """
    _require_data()
    df = _data["line_list"]
    confirmed_df = df[df["confirmed"]]
    confirmed_df = _month_to_ts(confirmed_df)

    if confirmed_df.empty:
        return []

    iso = confirmed_df["month_ts"].dt.isocalendar()
    confirmed_df = confirmed_df.assign(
        year_week=iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    )

    grouped = (
        confirmed_df.groupby("year_week")
        .agg(
            confirmed_cases=("confirmed", "size"),
            deaths=("condition_clean", lambda s: int((s == "dead").sum())),
            recoveries=("condition_clean", lambda s: int((s == "alive").sum())),
        )
        .reset_index()
        .sort_values("year_week")
    )
    grouped["total_cases"] = grouped["confirmed_cases"]

    return grouped[
        ["year_week", "total_cases", "confirmed_cases", "deaths", "recoveries"]
    ].to_dict(orient="records")


@router.get("/demographics")
def get_demographics():
    """Powers the age-group and gender cards. Percentages are of confirmed
    cases only (matches the KPI card denominators)."""
    _require_data()
    df = _data["line_list"]
    confirmed_df = df[df["confirmed"]]
    total = len(confirmed_df)

    age_order = ["0-14", "15-49", "50+"]
    age_counts = confirmed_df["age_bucket"].value_counts()
    age_groups = [
        {
            "category": cat,
            "count": int(age_counts.get(cat, 0)),
            "percentage": _pct(int(age_counts.get(cat, 0)), total),
        }
        for cat in age_order
    ]

    gender_order = ["Male", "Female", "Other/Unknown"]
    gender_counts = confirmed_df["gender_clean"].value_counts()
    gender = [
        {
            "category": cat,
            "count": int(gender_counts.get(cat, 0)),
            "percentage": _pct(int(gender_counts.get(cat, 0)), total),
        }
        for cat in gender_order
    ]

    return {"age_groups": age_groups, "gender": gender}


@router.get("/lga-breakdown")
def get_lga_breakdown(
    state: Optional[str] = Query(None, description="Filter to one state (case-insensitive)"),
    search: Optional[str] = Query(None, description="Substring match on LGA name (case-insensitive)"),
    min_cases: int = Query(0, ge=0, description="Only return LGAs with at least this many cases"),
    limit: int = Query(50, ge=1, le=500, description="Max rows to return, ranked by case count"),
):
    """Powers the LGA Breakdown table.

    PLACEHOLDER DATA — runs off a synthetic CSV because the real
    2015-2025 line-list has no State/LGA column. Replace the file this
    reads (see load_data()) the moment real geo-tagged data exists.
    """
    _require_data()
    lga_df = _data["lga"]

    if state:
        lga_df = lga_df[lga_df["State"].str.lower() == state.lower()]
        if lga_df.empty:
            raise HTTPException(status_code=404, detail=f"No records for state '{state}'")

    if search:
        lga_df = lga_df[lga_df["LGA"].str.lower().str.contains(search.lower())]

    confirmed = lga_df[lga_df["Case_Status"] == "Confirmed"]

    rows = []
    for (lga, state_name), group in confirmed.groupby(["LGA", "State"]):
        cases = len(group)
        if cases < min_cases:
            continue
        deaths = int((group["Outcome"] == "Deceased").sum())
        recoveries = int((group["Outcome"] == "Discharged").sum())
        last_update = pd.to_datetime(group["Last_Update"]).max()
        rows.append(
            {
                "lga": lga,
                "state": state_name,
                "cases": cases,
                "deaths": deaths,
                "recovery_rate_percent": _pct(recoveries, cases),
                "last_update": last_update.strftime("%Y-%m-%d")
                if pd.notna(last_update)
                else None,
            }
        )

    rows.sort(key=lambda r: r["cases"], reverse=True)
    return {
        "results": rows[:limit],
        "total_matching_lgas": len(rows),
        "data_source": "PLACEHOLDER synthetic data — see api/dashboard.py load_data()",
    }




# Add this to the bottom of api/dashboard.py

@router.get("/recent-alerts")
def get_recent_alerts(limit: int = Query(3, ge=1, le=10)):
    """
    Powers the 'Recent Alerts' section on the Live Alerts page.
    Returns the most recent confirmed cases from the placeholder LGA dataset.
    """
    _require_data()
    lga_df = _data["lga"]
    
    # Filter for confirmed cases only
    confirmed = lga_df[lga_df["Case_Status"] == "Confirmed"].copy()
    
    if confirmed.empty:
        return []
        
    # Ensure Last_Update is datetime
    confirmed["Last_Update"] = pd.to_datetime(confirmed["Last_Update"])
    
    # Sort by most recent update
    recent = confirmed.sort_values("Last_Update", ascending=False)
    
    # Group by LGA and State to get unique locations, then take top N
    # We aggregate to get the latest status per LGA
    grouped = recent.groupby(["LGA", "State"]).agg({
        "Case_Status": "count", # Count of cases in that LGA
        "Outcome": lambda x: (x == "Deceased").sum(), # Count deaths
        "Last_Update": "max"
    }).reset_index()
    
    # Sort again by date after grouping and take top N
    grouped = grouped.sort_values("Last_Update", ascending=False).head(limit)
    
    results = []
    for _, row in grouped.iterrows():
        cases = int(row["Case_Status"])
        deaths = int(row["Outcome"])
        
        # Determine Risk Level based on case count (logic from UI)
        if cases > 50:
            risk_level = "High"
            risk_color = "text-lassa-red"
            advice = "Avoid affected zones. Practice hygiene."
        elif cases > 20:
            risk_level = "Moderate"
            risk_color = "text-orange-600"
            advice = "Monitor symptoms. Avoid contact with rodents."
        else:
            risk_level = "Low"
            risk_color = "text-green-600"
            advice = "No active threat. Stay informed."
            
        results.append({
            "lga": row["LGA"],
            "state": row["State"],
            "reported_date": row["Last_Update"].strftime("%Y-%m-%d"),
            "risk_level": risk_level,
            "risk_color": risk_color,
            "advice": advice,
            "cases": cases,
            "deaths": deaths
        })
        
    return results