"""
AI4Lassa API — serves the national early-warning forecast model as JSON
endpoints for integration into other applications/services.

Run locally:
    uvicorn api.main:app --reload --port 8000

Docs (auto-generated):
    http://localhost:8000/docs
"""
import os
import sys
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import ForecastResponse, ForecastFeatures

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)  # so `from src.explain import ...` resolves regardless of CWD

from src.explain import ExplanationError

app = FastAPI(
    title="AI4Lassa API",
    description="Programmatic access to the national case-count early-warning model.",
    version="1.1.0",
)

# Allow calls from any origin by default — tighten this to your actual
# frontend's domain before exposing this publicly in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FORECAST_FEATURES = [
    "case_count", "case_count_lag1", "case_count_lag2", "case_count_lag3",
    "case_count_lag6", "case_count_lag12",
    "case_count_roll3_mean", "case_count_roll6_mean", "case_count_roll3_max",
    "case_growth_lag1", "positivity_rate_lag1",
    "month_sin", "month_cos", "year",
]

_artifacts = {}


@app.on_event("startup")
def load_artifacts():
    """Loads the final trained models once at process startup."""
    model_dir = os.path.join(BASE_DIR, "models")
    data_path = os.path.join(BASE_DIR, "data", "processed", "monthly_features.csv")

    required = {
        "final_rf_regressor.pkl": os.path.join(model_dir, "final_rf_regressor.pkl"),
        "final_logit_riskflag.pkl": os.path.join(model_dir, "final_logit_riskflag.pkl"),
        "final_riskflag_scaler.pkl": os.path.join(model_dir, "final_riskflag_scaler.pkl"),
        "final_config.pkl": os.path.join(model_dir, "final_config.pkl"),
        "monthly_features.csv": data_path,
    }
    missing = [name for name, path in required.items() if not os.path.exists(path)]
    if missing:
        _artifacts["forecast_error"] = (
            f"Required file(s) not found: {', '.join(missing)}. This usually means "
            "the models/ folder and data/processed/monthly_features.csv weren't "
            "included when this service was deployed — check the deployment's "
            "file/build configuration."
        )
        return

    try:
        _artifacts["rf"] = joblib.load(required["final_rf_regressor.pkl"])
        _artifacts["logit"] = joblib.load(required["final_logit_riskflag.pkl"])
        _artifacts["fc_scaler"] = joblib.load(required["final_riskflag_scaler.pkl"])
        _artifacts["config"] = joblib.load(required["final_config.pkl"])
        _artifacts["monthly_df"] = pd.read_csv(data_path).sort_values("month_ts")
        _artifacts["background_X_scaled"] = _artifacts["fc_scaler"].transform(
            _artifacts["monthly_df"][FORECAST_FEATURES].iloc[:80]
        )
    except Exception as e:
        _artifacts["forecast_error"] = (
            f"Model files were found but failed to load: {e}. This usually means "
            "a version mismatch (e.g. a different scikit-learn version than "
            "what the models were trained with) or a corrupted/incomplete file."
        )


def _run_forecast(row: pd.Series, threshold: float):
    from src.explain import explain_forecast_plain, explain_risk_flag_plain

    fc = explain_forecast_plain(row, _artifacts["rf"])
    rk = explain_risk_flag_plain(row, _artifacts["logit"], _artifacts["fc_scaler"], _artifacts["background_X_scaled"])

    forecast = fc["forecast_cases"]
    proba = rk["probability_percent"] / 100
    risk_level = "HIGH" if proba >= threshold else ("ELEVATED" if proba >= threshold * 0.5 else "LOW")

    factors = fc["explanation_sentences"] + rk["explanation_sentences"]

    return forecast, proba, risk_level, factors


@app.get("/health")
def health():
    return {
        "status": "ok",
        "forecast_model_loaded": "rf" in _artifacts,
        "load_error": _artifacts.get("forecast_error"),  # None when everything loaded fine
    }


@app.get("/")
def root():
    """Landing response for anyone hitting the bare domain — points them
    somewhere useful instead of a bare 404."""
    return {
        "service": "AI4Lassa API",
        "status": "ok" if "rf" in _artifacts else "model not loaded — see /health",
        "docs": "/docs",
        "endpoints": {
            "GET /health": "service + model status",
            "GET /forecast/latest": "forecast using the most recent month in the dataset",
            "POST /forecast/predict": "forecast from a caller-supplied feature vector",
        },
    }


@app.get("/forecast/latest", response_model=ForecastResponse, tags=["forecast"])
def forecast_latest(decision_threshold: float = 0.5):
    """
    Forecasts next month's national case count using the most recent month
    in the processed dataset. No request body needed — mirrors what the
    Streamlit app does.
    """
    if "rf" not in _artifacts:
        raise HTTPException(
            status_code=503,
            detail=_artifacts.get(
                "forecast_error",
                "The forecast model isn't loaded, and no specific reason was recorded. "
                "Check the server's startup logs for details.",
            ),
        )

    row = _artifacts["monthly_df"].iloc[-1]
    try:
        forecast, proba, risk_level, factors = _run_forecast(row, decision_threshold)
    except ExplanationError as e:
        raise HTTPException(status_code=500, detail=e.user_message)

    return ForecastResponse(
        based_on_month=str(row["month"]),
        forecast_next_month_cases=round(forecast),
        outbreak_probability_provisional=round(proba, 3),
        risk_level_provisional=risk_level,
        decision_threshold_used=decision_threshold,
        top_contributing_factors=factors,
        note="Decision-support signal only — not a diagnosis or confirmed outbreak "
             "declaration. Risk level is provisional; see project report for "
             "validation limitations.",
    )


@app.post("/forecast/predict", response_model=ForecastResponse, tags=["forecast"])
def forecast_predict(features: ForecastFeatures):
    """
    Forecasts next month's case count from an explicit, caller-supplied
    feature vector — for integrators who maintain their own monthly series
    and compute lags/rolling stats themselves.
    """
    if "rf" not in _artifacts:
        raise HTTPException(
            status_code=503,
            detail=_artifacts.get(
                "forecast_error",
                "The forecast model isn't loaded, and no specific reason was recorded. "
                "Check the server's startup logs for details.",
            ),
        )

    row = pd.Series(features.model_dump(exclude={"decision_threshold"}))
    threshold = features.decision_threshold or 0.5
    try:
        forecast, proba, risk_level, factors = _run_forecast(row, threshold)
    except ExplanationError as e:
        raise HTTPException(status_code=500, detail=e.user_message)

    return ForecastResponse(
        based_on_month="caller-supplied features",
        forecast_next_month_cases=round(forecast),
        outbreak_probability_provisional=round(proba, 3),
        risk_level_provisional=risk_level,
        decision_threshold_used=threshold,
        top_contributing_factors=factors,
        note="Decision-support signal only — not a diagnosis or confirmed outbreak "
             "declaration. Risk level is provisional; see project report for "
             "validation limitations.",
    )
