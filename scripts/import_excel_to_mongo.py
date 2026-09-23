"""One-time migration: load the Excel line-list into MongoDB.

Run from the project root:
    python -m backend.scripts.import_excel_to_mongo
"""
import os
import sys
import asyncio
import pandas as pd
from motor.motor_asyncio import AsyncIOMotorClient

# --- Make `app.config` importable -----------------------------------------
# This file: backend/scripts/import_excel_to_mongo.py
# We want `backend/` on sys.path so `from app.config import settings` works.
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.config import settings  # noqa: E402


# Columns we care about. We'll only request the ones that actually exist.
LINE_LIST_COLUMNS = [
    "YEAR",
    "MONTH",
    "AGE in Years",
    "SEX",
    "RESULTS",
    "CONDITION",
    "State",
    "LGA",
]

EXCEL_FILENAME = "2015-2025__3_.xlsx"


def normalize_row(row: pd.Series) -> dict:
    """Convert a pandas row into a clean MongoDB document."""
    def _str_or_none(v):
        return v if isinstance(v, str) and v.strip() else None

    year = row.get("YEAR")
    try:
        year = int(year) if pd.notna(year) else None
    except (TypeError, ValueError):
        year = None

    age = row.get("AGE in Years")
    try:
        age = float(age) if pd.notna(age) else None
    except (TypeError, ValueError):
        age = None

    return {
        "year": year,
        "month": row.get("MONTH") if pd.notna(row.get("MONTH")) else None,
        "age": age,
        "sex": _str_or_none(row.get("SEX")),
        "result": _str_or_none(row.get("RESULTS")),
        "condition": _str_or_none(row.get("CONDITION")),
        "state": _str_or_none(row.get("State")),
        "lga": _str_or_none(row.get("LGA")),
    }


async def main():
    # backend/data/raw/2015-2025__3_.xlsx
    path = os.path.join(BACKEND_DIR, "data", "raw", EXCEL_FILENAME)

    print(f"Looking for file at: {path}")
    print(f"Current working dir: {os.getcwd()}")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Excel file not found.\n"
            f"  Expected: {path}\n"
            f"  Check backend/data/raw/ contains {EXCEL_FILENAME}"
        )

    # Read header row to discover which of our target columns exist
    header_df = pd.read_excel(path, nrows=0, engine="openpyxl")
    available = set(header_df.columns)
    cols = [c for c in LINE_LIST_COLUMNS if c in available]
    missing = [c for c in LINE_LIST_COLUMNS if c not in available]

    print(f"Columns present:  {cols}")
    if missing:
        print(f"Columns missing (will be null on insert): {missing}")

    if not cols:
        raise RuntimeError(
            f"None of the expected columns were found in {path}.\n"
            f"  Expected any of: {LINE_LIST_COLUMNS}\n"
            f"  Found: {sorted(available)}"
        )

    df = pd.read_excel(path, sheet_name=0, usecols=cols, engine="openpyxl")
    print(f"Read {len(df)} rows from Excel.")

    docs = [normalize_row(r) for _, r in df.iterrows()]
    docs = [d for d in docs if d["year"] is not None]
    print(f"Prepared {len(docs)} documents (rows without YEAR dropped).")

    client = AsyncIOMotorClient(settings.MONGO_URI)
    db = client[settings.MONGO_DB]

    deleted = await db.cases.delete_many({})
    print(f"Cleared {deleted.deleted_count} existing docs from {settings.MONGO_DB}.cases")

    if docs:
        # inserted in chunks to keep memory / BSON size sane
        CHUNK = 5000
        total = 0
        for i in range(0, len(docs), CHUNK):
            chunk = docs[i : i + CHUNK]
            result = await db.cases.insert_many(chunk, ordered=False)
            total += len(result.inserted_ids)
        print(f"Inserted {total} case records into {settings.MONGO_DB}.cases")
    else:
        print("No documents to insert.")

    # Ensure indexes (safe to call repeatedly)
    await db.cases.create_index([("year", 1), ("month", 1)])
    await db.cases.create_index([("state", 1), ("lga", 1)])
    await db.cases.create_index([("result", 1)])
    print("Indexes ensured on cases: (year,month), (state,lga), (result).")

    client.close()


if __name__ == "__main__":
    asyncio.run(main())