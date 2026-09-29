"""
End-to-end smoke test against a RUNNING AI4Lassa server backed by REAL MongoDB Atlas.

Unlike test_retrain_local.py (which fakes the database and calls Python
functions directly), this talks to the actual HTTP API — so it's the real
integration test to run once your server is up and Atlas is reachable.

Usage:
    pip install requests
    python3 test_live_server.py [base_url]

    base_url defaults to http://localhost:8000
"""
import sys
import time
import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
TEST_EMAIL = f"smoketest+{int(time.time())}@example.com"
TEST_PASSWORD = "SmokeTest123!"


def check(label, condition, extra=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" — {extra}" if extra else ""))
    return condition


def main():
    ok = True
    print(f"Target: {BASE}\n")

    # 1. Health check — confirms Mongo is actually reachable from the server
    print("1. GET /health")
    r = requests.get(f"{BASE}/health", timeout=10)
    data = r.json()
    print(f"   {data}")
    ok &= check("server responded", r.status_code == 200)
    ok &= check(
        "MongoDB connected",
        data.get("mongo_connected") is True,
        "if False: check Atlas Network Access allows this machine's IP",
    )

    # 2. Register a throwaway admin account
    print("\n2. POST /auth/register")
    r = requests.post(f"{BASE}/auth/register", json={
        "email": TEST_EMAIL,
        "password": TEST_PASSWORD,
        "full_name": "Smoke Test",
        "role": "admin",
    }, timeout=10)
    ok &= check("registered", r.status_code in (200, 201), f"status={r.status_code} body={r.text[:200]}")

    # 3. Log in
    print("\n3. POST /auth/login")
    r = requests.post(f"{BASE}/auth/login", data={
        "username": TEST_EMAIL,
        "password": TEST_PASSWORD,
    }, timeout=10)
    ok &= check("logged in", r.status_code == 200, f"body={r.text[:200]}")
    token = r.json().get("access_token") if r.status_code == 200 else None
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    ok &= check("got access token", bool(token))

    if not token:
        print("\nCannot continue without a token. Stopping here.")
        sys.exit(1)

    # 4. Confirm forecast endpoints work against real data already in Mongo
    print("\n4. GET /forecast/latest")
    r = requests.get(f"{BASE}/forecast/latest", timeout=15)
    print(f"   {r.json() if r.status_code == 200 else r.text[:200]}")
    ok &= check("forecast returned", r.status_code == 200)

    # 5. Check model-versions / model-history endpoints exist and respond
    print("\n5. GET /api/v1/admin/model-versions")
    r = requests.get(f"{BASE}/api/v1/admin/model-versions", headers=headers, timeout=10)
    print(f"   status={r.status_code} body={r.text[:300]}")
    ok &= check("model-versions responded", r.status_code == 200)

    print("\n6. GET /api/v1/admin/model-history")
    r = requests.get(f"{BASE}/api/v1/admin/model-history", headers=headers, timeout=10)
    print(f"   status={r.status_code} body={r.text[:300]}")
    ok &= check("model-history responded", r.status_code == 200)
    events_before = len(r.json().get("events", [])) if r.status_code == 200 else 0

    # 7. Trigger a real retrain via upload (requires a small real CSV — see note below)
    print("\n7. POST /api/v1/admin/upload (retrain_model=true)")
    print("   NOTE: this step needs a real CSV with YEAR, MONTH, RESULTS columns.")
    print("   Uncomment and point CSV_PATH at one to actually exercise retrain end-to-end.")
    CSV_PATH = None  # e.g. "sample_upload.csv"
    if CSV_PATH:
        with open(CSV_PATH, "rb") as f:
            r = requests.post(
                f"{BASE}/api/v1/admin/upload",
                headers=headers,
                files={"file": f},
                data={"upload_type": "outbreak_data", "retrain_model": "true"},
                timeout=120,
            )
        print(f"   status={r.status_code} body={r.text[:500]}")
        ok &= check("upload+retrain responded", r.status_code == 200)

        # 8. Confirm the audit log actually grew by one event
        print("\n8. GET /api/v1/admin/model-history (after retrain)")
        r = requests.get(f"{BASE}/api/v1/admin/model-history", headers=headers, timeout=10)
        events_after = len(r.json().get("events", [])) if r.status_code == 200 else 0
        ok &= check(
            "audit log grew by exactly 1 event",
            events_after == events_before + 1,
            f"before={events_before} after={events_after}",
        )
    else:
        print("   SKIPPED (no CSV_PATH set)")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED — see [FAIL] lines above"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
