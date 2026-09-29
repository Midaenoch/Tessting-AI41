"""
Local, no-Mongo-required test harness for app.services.forecast_service.retrain().

Builds a fake `db.cases` collection backed by the real historical line-list
(data/raw/2015-2025__3_.xlsx), so retrain() runs against real data without
needing network access to MongoDB Atlas.

Run: python3 test_retrain_local.py
"""
import asyncio
import os
import shutil
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from app.services import forecast_service  # noqa: E402


class FakeCursor:
    def __init__(self, docs):
        self._docs = docs

    def sort(self, field, direction):
        self._docs = sorted(self._docs, key=lambda d: d[field], reverse=(direction < 0))
        return self

    def limit(self, n):
        self._docs = self._docs[:n]
        return self

    async def to_list(self, length=None):
        return self._docs


class FakeCasesCollection:
    def __init__(self, docs):
        self._docs = docs

    def find(self, filt, projection):
        # Good enough for retrain()'s query shape: result~=positive, year/month not null.
        out = [d for d in self._docs if d.get("year") is not None and d.get("month") is not None]
        return FakeCursor(out)


class FakeModelEventsCollection:
    def __init__(self):
        self.docs = []

    async def insert_one(self, doc):
        self.docs.append(doc)
        return type("Result", (), {"inserted_id": len(self.docs)})()

    def find(self, filt, projection):
        return FakeCursor(list(self.docs))


class FakeDB:
    def __init__(self, docs):
        self.cases = FakeCasesCollection(docs)
        self.model_events = FakeModelEventsCollection()


def load_real_docs_from_excel(base_dir: str) -> list[dict]:
    path = os.path.join(base_dir, "data", "raw", "2015-2025__3_.xlsx")
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    docs = []
    for _, row in df.iterrows():
        year = row.get("YEAR")
        month = row.get("MONTH")
        result = row.get("RESULTS", "")
        if pd.isna(year) or pd.isna(month):
            continue
        docs.append({
            "year": int(year),
            "month": month,
            "result": str(result) if not pd.isna(result) else "",
        })
    return docs


FAKE_ADMIN = {"id": "user_1", "email": "admin@ai4lassa.org", "role": "admin"}
FAKE_OFFICER = {"id": "user_2", "email": "officer@ai4lassa.org", "role": "health_officer"}


async def main():
    print(f"Loading real historical line-list from data/raw/2015-2025__3_.xlsx ...")
    docs = load_real_docs_from_excel(BASE_DIR)
    positive_docs = [d for d in docs if "positive" in d["result"].lower()]
    print(f"  {len(docs)} total rows, {len(positive_docs)} confirmed-positive rows")

    print("\nLoading existing model artifacts (the 'currently live' model) ...")
    forecast_service.load_artifacts(BASE_DIR)
    print(f"  loaded: {forecast_service.is_loaded()}, error: {forecast_service.get_error()}")

    print("\n=== Test 1: retrain on the full real dataset (should promote) ===")
    fake_db = FakeDB(docs)
    result1 = await forecast_service.retrain(fake_db, triggered_by=FAKE_ADMIN)
    for k, v in result1.items():
        print(f"  {k}: {v}")

    versions = forecast_service.list_archived_versions(BASE_DIR)
    print(f"\nArchived versions after retrain #1: {versions}")

    print("\n=== Test 2: retrain again immediately on the SAME data (should also promote — same data, same skill) ===")
    result2 = await forecast_service.retrain(fake_db, triggered_by=FAKE_OFFICER)
    for k, v in result2.items():
        print(f"  {k}: {v}")

    versions = forecast_service.list_archived_versions(BASE_DIR)
    print(f"\nArchived versions after retrain #2: {versions}")

    print("\n=== Test 3: rollback to the previous version ===")
    rollback_result = await forecast_service.rollback_to_version(BASE_DIR, db=fake_db, triggered_by=FAKE_ADMIN)
    print(f"  {rollback_result}")
    print(f"  model reloaded, is_loaded={forecast_service.is_loaded()}, error={forecast_service.get_error()}")

    print("\n=== Test 4: artificially bad candidate should be BLOCKED ===")
    # Corrupt a copy of the docs so the resulting monthly series is nonsense —
    # a good implementation should refuse to promote this.
    import random
    bad_docs = [dict(d) for d in docs]
    random.seed(0)
    for d in bad_docs:
        if "positive" in d["result"].lower():
            d["year"] = random.choice([2015, 2016, 2017])  # scramble time ordering
    fake_bad_db = FakeDB(bad_docs)
    result4 = await forecast_service.retrain(fake_bad_db, triggered_by=FAKE_OFFICER)
    for k, v in result4.items():
        print(f"  {k}: {v}")
    print(f"  promoted={result4['promoted']} (expect this to plausibly be False, or at least run without crashing)")

    print("\n=== Test 5: audit log — GET /model-history equivalent ===")
    history = await forecast_service.get_model_history(fake_db, limit=50)
    print(f"  {len(history)} events logged for fake_db (retrain #1, #2, rollback):")
    for ev in history:
        who = (ev.get("triggered_by") or {}).get("email", "unknown")
        print(f"    [{ev['timestamp']}] {ev['event_type']:8s} by {who:28s} "
              f"promoted={ev.get('promoted')} rolled_back_to={ev.get('rolled_back_to')}")

    print("\nAll tests completed without unhandled exceptions.")


if __name__ == "__main__":
    asyncio.run(main())
