"""
app/services/map_service.py

Aggregates the `alerts` collection into one record per Nigerian state for the
homepage map. Keys match the frontend's keyOf(): lowercase, letters only
("akwaibom", "fct", ...).

Status rules (adjust the constants to match how your team defines them):
  active : at least one alert with cases > 0 reported in the last ACTIVE_DAYS
  watch  : alerts exist in the last WATCH_DAYS but none qualify as active
  none   : no alerts in the last WATCH_DAYS (state is simply absent from the
           response, the frontend treats a missing key as "none")
"""
import re
from datetime import datetime, timedelta, timezone

from app.database import get_db

ACTIVE_DAYS = 30
WATCH_DAYS = 90


def state_key(name: str) -> str:
    """Python twin of keyOf() in NigeriaAlertMap.tsx. Keep the two in sync."""
    k = re.sub(r"\bstate\b", "", str(name).lower())
    k = re.sub(r"[^a-z]", "", k)
    if "federalcapital" in k or k in ("fct", "abuja"):
        return "fct"
    if k == "nassarawa":
        return "nasarawa"
    return k


def _naive_utc(dt: datetime) -> datetime:
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


async def get_state_alerts() -> dict:
    db = get_db()
    if db is None:
        raise RuntimeError("Database not connected")

    now = datetime.utcnow()
    watch_since = now - timedelta(days=WATCH_DAYS)
    active_since = now - timedelta(days=ACTIVE_DAYS)

    cursor = db.alerts.find(
        {"reported_date": {"$gte": watch_since}},
        {"_id": 0, "state": 1, "lga": 1, "cases": 1, "deaths": 1, "reported_date": 1},
    )
    docs = await cursor.to_list(length=None)

    states: dict[str, dict] = {}
    for d in docs:
        state, reported = d.get("state"), d.get("reported_date")
        if not state or not isinstance(reported, datetime):
            continue
        reported = _naive_utc(reported)
        key = state_key(state)
        if not key:
            continue

        cases = int(d.get("cases") or 0)
        deaths = int(d.get("deaths") or 0)

        s = states.setdefault(key, {
            "cases": 0, "deaths": 0, "lgas": set(),
            "latest": None, "latest_lga": None, "active": False,
        })
        s["cases"] += cases
        s["deaths"] += deaths
        if d.get("lga"):
            s["lgas"].add(str(d["lga"]).strip().lower())
        if cases > 0 and reported >= active_since:
            s["active"] = True
        if s["latest"] is None or reported > s["latest"]:
            s["latest"] = reported
            s["latest_lga"] = d.get("lga")

    return {
        key: {
            "status": "active" if s["active"] else "watch",
            "cases": s["cases"],
            "deaths": s["deaths"],
            "lgas": len(s["lgas"]),
            "latest_lga": s["latest_lga"],
            "updated_at": s["latest"].isoformat() + "Z",
        }
        for key, s in states.items()
    }