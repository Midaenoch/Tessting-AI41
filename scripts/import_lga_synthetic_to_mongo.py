"""One-time migration: load the synthetic LGA CSV into MongoDB.

Run from project root:
    python -m backend.scripts.import_lga_synthetic_to_mongo
"""
import os
import sys
import asyncio
from datetime import datetime

import pandas as pd
from motor.motor_asyncio import AsyncIOMotorClient

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.config import settings  # noqa: E402


CSV_FILENAME = "lga_breakdown_PLACEHOLDER_synthetic.csv"
CHUNK_SIZE = 5000


def _to_iso_date(val):
    if pd.isna(val):
        return None
    try:
        ts = pd.to_datetime(val)
        return ts.strftime("%Y-%m-%d")
    except Exception:
        return None


async def main():
    path = os.path.join(BACKEND_DIR, "data", "raw", CSV_FILENAME)
    print(f"[lga-import] Looking for CSV: {path}")
    print(f"[lga-import] Mongo DB: {settings.MONGO_DB}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    print(f"[lga-import] Read {len(df)} rows. Columns: {list(df.columns)}")

    # Tolerant column mapping — adjust if your CSV header differs.
    # Expected columns based on the original code:
    #   LGA, State, Case_Status, Outcome, Last_Update
    required = {"LGA", "State", "Case_Status", "Outcome", "Last_Update"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"CSV missing required columns: {missing}\n"
            f"  Found: {list(df.columns)}"
        )

    docs = []
    for _, r in df.iterrows():
        docs.append({
            "lga": str(r["LGA"]).strip() if pd.notna(r["LGA"]) else None,
            "state": str(r["State"]).strip() if pd.notna(r["State"]) else None,
            "case_status": str(r["Case_Status"]).strip() if pd.notna(r["Case_Status"]) else None,
            "outcome": str(r["Outcome"]).strip() if pd.notna(r["Outcome"]) else None,
            "last_update": _to_iso_date(r["Last_Update"]),
        })

    client = AsyncIOMotorClient(settings.MONGO_URI, serverSelectionTimeoutMS=8000)
    db = client[settings.MONGO_DB]

    try:
        await client.admin.command("ping")
        print("[lga-import] Connected to MongoDB Atlas.")
    except Exception as e:
        raise RuntimeError(f"Could not connect: {e}")

    deleted = await db.lga_synthetic.delete_many({})
    print(f"[lga-import] Cleared {deleted.deleted_count} old docs from lga_synthetic.")

    total = 0
    for i in range(0, len(docs), CHUNK_SIZE):
        chunk = docs[i : i + CHUNK_SIZE]
        res = await db.lga_synthetic.insert_many(chunk, ordered=False)
        total += len(res.inserted_ids)
    print(f"[lga-import] Inserted {total} rows into {settings.MONGO_DB}.lga_synthetic")

    await db.lga_synthetic.create_index([("state", 1), ("lga", 1)])
    await db.lga_synthetic.create_index([("case_status", 1)])
    print("[lga-import] Indexes ensured on lga_synthetic.")

    client.close()
    print("[lga-import] Done.")


if __name__ == "__main__":
    asyncio.run(main())