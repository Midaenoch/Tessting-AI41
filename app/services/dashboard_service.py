"""Dashboard service backed entirely by MongoDB.

Collections used:
  - cases          : real line-list records (year, month, age, sex, result, condition)
  - lga_synthetic  : placeholder synthetic geo data (state, lga, case_status, outcome)
  - alerts         : stored alert documents

No runtime file reads.
"""
from datetime import datetime
from typing import Optional

from app.database import get_db


_MONTH_NAME_TO_NUM = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _require_db():
    db = get_db()
    if db is None:
        raise RuntimeError("Database not connected")
    return db


def _parse_month(val) -> Optional[int]:
    if val is None:
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
        m = int(val)
    except (TypeError, ValueError):
        return None
    return m if 1 <= m <= 12 else None


def _clean_condition(condition: Optional[str]) -> Optional[str]:
    if not condition:
        return None
    v = condition.strip().lower()
    if v.startswith("dead"):
        return "dead"
    if v.startswith("alive"):
        return "alive"
    return None


def _clean_sex(sex: Optional[str]) -> str:
    if not sex:
        return "Other/Unknown"
    v = sex.strip().lower()
    if v == "m":
        return "Male"
    if v == "f":
        return "Female"
    return "Other/Unknown"


def _age_bucket(age) -> Optional[str]:
    if age is None:
        return None
    try:
        age = float(age)
    except (TypeError, ValueError):
        return None
    if age < 0 or age > 120:
        return None
    if age <= 14:
        return "0-14"
    if age <= 49:
        return "15-49"
    return "50+"


def _pct(part: int, whole: int) -> float:
    return round((part / whole) * 100, 1) if whole else 0.0


def _confirmed_filter() -> dict:
    """Case-insensitive match for 'positive' anywhere in `result`."""
    return {"result": {"$regex": "positive", "$options": "i"}}


def _confirmed_status_filter() -> dict:
    """Case-insensitive, whitespace-tolerant match for 'confirmed' on lga_synthetic."""
    # ^\\s*confirmed\\s*$ matches "Confirmed", "confirmed", " confirmed ", "CONFIRMED"
    return {"case_status": {"$regex": r"^\s*confirmed\s*$", "$options": "i"}}


def _outcome_match(value: str) -> dict:
    """Regex that matches a specific outcome word, case-insensitive + trimmed."""
    return {"outcome": {"$regex": rf"^\s*{value}\s*$", "$options": "i"}}


# ---------------------------------------------------------------------------
# Public API — all read from MongoDB
# ---------------------------------------------------------------------------
async def get_kpi() -> dict:
    db = _require_db()
    cases = db.cases
    cf = _confirmed_filter()

    # Real line-list numbers
    confirmed_cases = await cases.count_documents(cf)
    deaths = await cases.count_documents(
        {**cf, "condition": {"$regex": r"^\s*dead", "$options": "i"}}
    )
    recoveries = await cases.count_documents(
        {**cf, "condition": {"$regex": r"^\s*alive", "$options": "i"}}
    )

    pipeline_outbreaks = [
        {"$match": {**cf, "year": {"$ne": None}, "month": {"$ne": None}}},
        {"$group": {"_id": {"year": "$year", "month": "$month"}}},
        {"$count": "total"},
    ]
    outbreaks_res = await cases.aggregate(pipeline_outbreaks).to_list(length=1)
    total_outbreaks = outbreaks_res[0]["total"] if outbreaks_res else 0

    years = await cases.distinct("year", {"year": {"$ne": None}})
    total_records = await cases.count_documents({})

    # Placeholder geo numbers — from lga_synthetic, matched robustly
    lga = db.lga_synthetic
    cs_filter = _confirmed_status_filter()
    state_values = await lga.distinct("state", cs_filter)
    lga_values = await lga.distinct("lga", cs_filter)

    states_affected = len([s for s in state_values if isinstance(s, str) and s.strip()])
    lgas_affected = len([l for l in lga_values if isinstance(l, str) and l.strip()])

    return {
        "confirmed_cases": confirmed_cases,
        "recoveries": recoveries,
        "deaths": deaths,
        "fatality_rate_percent": _pct(deaths, confirmed_cases),
        "recovery_rate_percent": _pct(recoveries, confirmed_cases),
        "total_outbreaks": total_outbreaks,
        "states_affected": states_affected,
        "lgas_affected": lgas_affected,
        "total_records": total_records,
        "years_covered": sorted(int(y) for y in years),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "data_sources": {
            "confirmed_cases": "mongodb.cases (real)",
            "recoveries": "mongodb.cases (real)",
            "deaths": "mongodb.cases (real)",
            "total_outbreaks": "mongodb.cases (real, derived)",
            "states_affected": "mongodb.lga_synthetic (PLACEHOLDER)",
            "lgas_affected": "mongodb.lga_synthetic (PLACEHOLDER)",
        },
    }


async def get_yearly_trends() -> list:
    db = _require_db()
    pipeline = [
        {"$match": {**_confirmed_filter(), "year": {"$ne": None}}},
        {"$group": {"_id": "$year", "confirmed_cases": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
    ]
    rows = await db.cases.aggregate(pipeline).to_list(length=None)
    return [
        {"year": int(r["_id"]), "confirmed_cases": int(r["confirmed_cases"])}
        for r in rows
    ]


async def get_weekly_trends() -> list:
    """ISO-week buckets derived from year+month (day = 1 of each month)."""
    db = _require_db()
    rows = await db.cases.find(
        {**_confirmed_filter(), "year": {"$ne": None}, "month": {"$ne": None}},
        {"_id": 0, "year": 1, "month": 1, "condition": 1},
    ).to_list(length=None)

    buckets: dict[str, dict] = {}
    for r in rows:
        m = _parse_month(r.get("month"))
        if m is None:
            continue
        try:
            ts = datetime(int(r["year"]), m, 1)
        except (TypeError, ValueError):
            continue
        iso_year, iso_week, _ = ts.isocalendar()
        key = f"{iso_year}-W{iso_week:02d}"
        b = buckets.setdefault(key, {
            "year_week": key,
            "total_cases": 0,
            "confirmed_cases": 0,
            "deaths": 0,
            "recoveries": 0,
        })
        b["confirmed_cases"] += 1
        b["total_cases"] += 1
        cond = _clean_condition(r.get("condition"))
        if cond == "dead":
            b["deaths"] += 1
        elif cond == "alive":
            b["recoveries"] += 1

    return sorted(buckets.values(), key=lambda x: x["year_week"])


async def get_demographics() -> dict:
    db = _require_db()
    rows = await db.cases.find(
        _confirmed_filter(),
        {"_id": 0, "age": 1, "sex": 1},
    ).to_list(length=None)

    total = len(rows)
    age_counts = {"0-14": 0, "15-49": 0, "50+": 0}
    gender_counts = {"Male": 0, "Female": 0, "Other/Unknown": 0}

    for r in rows:
        bucket = _age_bucket(r.get("age"))
        if bucket:
            age_counts[bucket] += 1
        gender_counts[_clean_sex(r.get("sex"))] += 1

    return {
        "age_groups": [
            {"category": c, "count": n, "percentage": _pct(n, total)}
            for c, n in age_counts.items()
        ],
        "gender": [
            {"category": c, "count": n, "percentage": _pct(n, total)}
            for c, n in gender_counts.items()
        ],
    }


async def get_lga_breakdown(
    state: Optional[str],
    search: Optional[str],
    min_cases: int,
    limit: int,
) -> dict:
    db = _require_db()
    coll = db.lga_synthetic

    match: dict = _confirmed_status_filter()
    if state:
        match["state"] = {"$regex": rf"^\s*{state}\s*$", "$options": "i"}
    if search:
        match["lga"] = {"$regex": search, "$options": "i"}

    pipeline = [
        {"$match": match},
        {"$group": {
            "_id": {"lga": "$lga", "state": "$state"},
            "cases": {"$sum": 1},
            "deaths": {"$sum": {"$cond": [
                {"$regexMatch": {
                    "input": {"$ifNull": ["$outcome", ""]},
                    "regex": r"^\s*deceased\s*$",
                    "options": "i",
                }}, 1, 0,
            ]}},
            "recoveries": {"$sum": {"$cond": [
                {"$regexMatch": {
                    "input": {"$ifNull": ["$outcome", ""]},
                    "regex": r"^\s*discharged\s*$",
                    "options": "i",
                }}, 1, 0,
            ]}},
            "last_update": {"$max": "$last_update"},
        }},
        {"$match": {"cases": {"$gte": min_cases}}},
        {"$sort": {"cases": -1}},
        {"$limit": limit},
    ]

    rows = await coll.aggregate(pipeline).to_list(length=None)
    results = [
        {
            "lga": r["_id"]["lga"],
            "state": r["_id"]["state"],
            "cases": int(r["cases"]),
            "deaths": int(r["deaths"]),
            "recovery_rate_percent": _pct(int(r["recoveries"]), int(r["cases"])),
            "last_update": r.get("last_update"),
        }
        for r in rows
    ]

    return {
        "results": results,
        "total_matching_lgas": len(results),
        "data_source": "PLACEHOLDER synthetic data (mongodb.lga_synthetic)",
    }


async def get_recent_alerts(limit: int = 3) -> list:
    """Alerts read from the `alerts` collection."""
    db = _require_db()
    cursor = db.alerts.find({}).sort("reported_date", -1).limit(limit)
    docs = await cursor.to_list(length=limit)

    results = []
    for d in docs:
        cases = int(d.get("cases", 0))
        if cases > 50:
            level, color, advice = "High", "text-red-600", "Avoid affected zones."
        elif cases > 20:
            level, color, advice = "Moderate", "text-orange-600", "Monitor symptoms."
        else:
            level, color, advice = "Low", "text-green-600", "No active threat."

        rd = d.get("reported_date")
        rd_str = rd.strftime("%Y-%m-%d") if isinstance(rd, datetime) else (str(rd) if rd else None)

        results.append({
            "lga": d.get("lga"),
            "state": d.get("state"),
            "reported_date": rd_str,
            "risk_level": level,
            "risk_color": color,
            "advice": advice,
            "cases": cases,
            "deaths": int(d.get("deaths", 0)),
        })
    return results