import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.config import settings
from app.database import connect_db, close_db
from app.services import forecast_service
from app.routers import auth, forecast, dashboard, alerts, admin


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_db()
    try:
        forecast_service.load_artifacts(BASE_DIR)
    except Exception as e:
        print(f"[startup] WARNING: forecast model failed to load: {e}")
    yield
    await close_db()


app = FastAPI(
    title="AI4Lassa API",
    version="1.4.0",
    lifespan=lifespan,
)

origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
print(f"[startup] CORS allow_origins = {origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(forecast.router)
app.include_router(dashboard.router)
app.include_router(alerts.router)
app.include_router(admin.router)


@app.get("/health")
async def health():
    from app.database import get_db
    db = get_db()
    mongo_ok = False
    if db is not None:
        try:
            await db.command("ping")
            mongo_ok = True
        except Exception:
            mongo_ok = False

    return {
        "status": "ok" if mongo_ok else "degraded",
        "mongo_connected": mongo_ok,
        "forecast_model_loaded": forecast_service.is_loaded(),
        "load_error": forecast_service.get_error(),
    }


@app.get("/")
async def root():
    return {
        "service": "AI4Lassa API",
        "docs": "/docs",
        "endpoints": [
            "POST /auth/register",
            "POST /auth/login",
            "GET  /auth/me",
            "GET  /auth/users",
            "GET  /forecast/latest",
            "POST /forecast/predict",
            "GET  /api/v1/kpi",
            "GET  /api/v1/trends/yearly",
            "GET  /api/v1/trends/weekly",
            "GET  /api/v1/demographics",
            "GET  /api/v1/lga-breakdown",
            "GET  /alerts/recent",
            "POST /alerts",
            "POST /api/v1/admin/upload",
            "GET  /api/v1/admin/export/{kind}",
        ],
    }