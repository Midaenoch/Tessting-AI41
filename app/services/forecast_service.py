import os
from datetime import datetime
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Feature contract — must match the trained artifacts.
# ---------------------------------------------------------------------------
FORECAST_FEATURES = [
    "case_count", "case_count_lag1", "case_count_lag2", "case_count_lag3",
    "case_count_lag6", "case_count_lag12",
    "case_count_roll3_mean", "case_count_roll6_mean", "case_count_roll3_max",
    "case_growth_lag1", "positivity_rate_lag1",
    "month_sin", "month_cos", "year",
]

RISK_FLAG_THRESHOLD_QUANTILE = 0.75
MIN_MONTHS_FOR_TRAIN = 18
BACKGROUND_SAMPLE_SIZE = 80

_MONTH_LOOKUP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_artifacts: dict = {}


# ===========================================================================
# Loading existing artifacts
# ===========================================================================
def load_artifacts(base_dir: str):
    _artifacts["_base_dir"] = base_dir

    model_dir = os.path.join(base_dir, "models")
    data_path = os.path.join(base_dir, "data", "processed", "monthly_features.csv")

    required = {
        "rf": os.path.join(model_dir, "final_rf_regressor.pkl"),
        "logit": os.path.join(model_dir, "final_logit_riskflag.pkl"),
        "scaler": os.path.join(model_dir, "final_riskflag_scaler.pkl"),
        "config": os.path.join(model_dir, "final_config.pkl"),
        "monthly": data_path,
    }
    missing = [k for k, p in required.items() if not os.path.exists(p)]
    if missing:
        _artifacts["error"] = f"Missing files: {missing}"
        return

    try:
        _artifacts["rf"] = joblib.load(required["rf"])
        _artifacts["logit"] = joblib.load(required["logit"])
        _artifacts["scaler"] = joblib.load(required["scaler"])
        _artifacts["config"] = joblib.load(required["config"])
        _artifacts["monthly_df"] = pd.read_csv(data_path).sort_values("month_ts")
        _artifacts["background_X"] = _artifacts["scaler"].transform(
            _artifacts["monthly_df"][FORECAST_FEATURES].iloc[:BACKGROUND_SAMPLE_SIZE]
        )
        _artifacts["error"] = None
    except Exception as e:
        _artifacts["error"] = str(e)


def is_loaded() -> bool:
    return "rf" in _artifacts


def get_error() -> Optional[str]:
    return _artifacts.get("error")


# ===========================================================================
# Inference
# ===========================================================================
def _run(row: pd.Series, threshold: float):
    from src.explain import explain_forecast_plain, explain_risk_flag_plain

    fc = explain_forecast_plain(row, _artifacts["rf"])
    rk = explain_risk_flag_plain(
        row, _artifacts["logit"], _artifacts["scaler"], _artifacts["background_X"]
    )

    forecast = fc["forecast_cases"]
    proba = rk["probability_percent"] / 100
    risk_level = (
        "HIGH" if proba >= threshold
        else "ELEVATED" if proba >= threshold * 0.5
        else "LOW"
    )
    return (
        forecast,
        proba,
        risk_level,
        fc["explanation_sentences"] + rk["explanation_sentences"],
    )


def predict_latest(threshold: float = 0.5) -> dict:
    if not is_loaded():
        raise RuntimeError(get_error() or "Model not loaded")
    row = _artifacts["monthly_df"].iloc[-1]
    forecast, proba, risk, factors = _run(row, threshold)
    return {
        "based_on_month": str(row["month"]),
        "forecast_next_month_cases": round(forecast),
        "outbreak_probability_provisional": round(proba, 3),
        "risk_level_provisional": risk,
        "decision_threshold_used": threshold,
        "top_contributing_factors": factors,
        "note": "Decision-support signal only — not a diagnosis.",
    }


def predict_features(features: dict, threshold: Optional[float] = None) -> dict:
    if not is_loaded():
        raise RuntimeError(get_error() or "Model not loaded")
    th = threshold if threshold is not None else features.pop("decision_threshold", 0.5)
    features.pop("decision_threshold", None)
    row = pd.Series(features)
    forecast, proba, risk, factors = _run(row, th)
    return {
        "based_on_month": "caller-supplied features",
        "forecast_next_month_cases": round(forecast),
        "outbreak_probability_provisional": round(proba, 3),
        "risk_level_provisional": risk,
        "decision_threshold_used": th,
        "top_contributing_factors": factors,
        "note": "Decision-support signal only — not a diagnosis.",
    }


# ===========================================================================
# Retraining from MongoDB
# ===========================================================================
def _to_month_num(v) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, str):
        key = v.strip().lower()
        if key in _MONTH_LOOKUP:
            return _MONTH_LOOKUP[key]
        try:
            v = float(key)
        except ValueError:
            return None
    try:
        m = int(v)
        return m if 1 <= m <= 12 else None
    except (TypeError, ValueError):
        return None


def _build_monthly_features(rows: list[dict]) -> pd.DataFrame:
    """Turn raw case docs into the monthly feature frame the model expects."""
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No confirmed cases found in the `cases` collection")

    df["month_num"] = df["month"].apply(_to_month_num)
    df = df.dropna(subset=["year", "month_num"])
    if df.empty:
        raise RuntimeError("No rows have both a valid year and month")

    df["year"] = df["year"].astype(int)
    df["month_num"] = df["month_num"].astype(int)

    # Aggregate to monthly counts
    monthly = (
        df.groupby(["year", "month_num"])
        .size()
        .reset_index(name="case_count")
        .sort_values(["year", "month_num"])
        .reset_index(drop=True)
    )

    # Fill gaps so lags are time-contiguous
    monthly["month_ts"] = pd.to_datetime(
        dict(year=monthly["year"], month=monthly["month_num"], day=1)
    )
    full_range = pd.date_range(
        monthly["month_ts"].min(), monthly["month_ts"].max(), freq="MS"
    )
    monthly = (
        monthly.set_index("month_ts")
        .reindex(full_range, fill_value=0)
        .rename_axis("month_ts")
        .reset_index()
    )
    monthly["year"] = monthly["month_ts"].dt.year
    monthly["month_num"] = monthly["month_ts"].dt.month

    # Lag features
    cc = monthly["case_count"]
    monthly["case_count_lag1"] = cc.shift(1)
    monthly["case_count_lag2"] = cc.shift(2)
    monthly["case_count_lag3"] = cc.shift(3)
    monthly["case_count_lag6"] = cc.shift(6)
    monthly["case_count_lag12"] = cc.shift(12)

    # Rolling features (past-only)
    monthly["case_count_roll3_mean"] = cc.shift(1).rolling(3).mean()
    monthly["case_count_roll6_mean"] = cc.shift(1).rolling(6).mean()
    monthly["case_count_roll3_max"] = cc.shift(1).rolling(3).max()

    # Growth (lagged by 1 so no leakage)
    prev1 = cc.shift(1)
    prev2 = cc.shift(2)
    monthly["case_growth_lag1"] = (prev1 - prev2) / prev2.replace(0, np.nan)

    # Positivity — no denominators in `cases`, use neutral 1.0 for confirmed
    monthly["positivity_rate_lag1"] = 1.0

    # Cyclical month encoding
    monthly["month_sin"] = np.sin(2 * np.pi * monthly["month_num"] / 12)
    monthly["month_cos"] = np.cos(2 * np.pi * monthly["month_num"] / 12)

    # Human-readable month label
    monthly["month"] = monthly["month_ts"].dt.strftime("%Y-%m")

    # Drop rows with NaN in any feature we need
    needed = FORECAST_FEATURES + ["case_count"]
    monthly = monthly.dropna(subset=needed).reset_index(drop=True)

    return monthly


async def retrain(db) -> dict:
    """Rebuild forecast + risk-flag models from the `cases` collection.

    Hot-swaps in-memory artifacts on success. Persists to disk.
    Returns a metrics dict. Raises RuntimeError on failure.
    """
    # 1. Read confirmed cases
    cursor = db.cases.find(
        {
            "result": {"$regex": "positive", "$options": "i"},
            "year": {"$ne": None},
            "month": {"$ne": None},
        },
        {"_id": 0, "year": 1, "month": 1},
    )
    rows = await cursor.to_list(length=None)
    if len(rows) < 50:
        raise RuntimeError(
            f"Not enough confirmed cases to retrain ({len(rows)} rows)"
        )

    # 2. Feature engineering
    monthly = _build_monthly_features(rows)
    if len(monthly) < MIN_MONTHS_FOR_TRAIN:
        raise RuntimeError(
            f"Only {len(monthly)} usable months after lagging; "
            f"need at least {MIN_MONTHS_FOR_TRAIN}"
        )

    X = monthly[FORECAST_FEATURES].values
    y_reg = monthly["case_count"].values

    # 3. Build risk-flag labels: top-quartile case months
    threshold_cases = float(np.quantile(y_reg, RISK_FLAG_THRESHOLD_QUANTILE))
    y_cls = (y_reg >= threshold_cases).astype(int)

    # If label is degenerate, fall back to median split
    if len(np.unique(y_cls)) < 2:
        threshold_cases = float(np.median(y_reg))
        y_cls = (y_reg >= threshold_cases).astype(int)

    # 4. Train regressor
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y_reg)

    # 5. Train classifier
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    logit = LogisticRegression(max_iter=2000, class_weight="balanced")
    logit.fit(X_scaled, y_cls)

    # 6. Background distribution for the explainer
    background_X = scaler.transform(X[:BACKGROUND_SAMPLE_SIZE])

    # 7. Config snapshot
    config = {
        "features": FORECAST_FEATURES,
        "risk_flag_threshold_cases": threshold_cases,
        "risk_flag_threshold_quantile": RISK_FLAG_THRESHOLD_QUANTILE,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "n_samples": int(len(monthly)),
        "n_features": len(FORECAST_FEATURES),
    }

    # 8. Persist to disk
    base_dir = _artifacts.get("_base_dir")
    if base_dir is None:
        raise RuntimeError(
            "Cannot determine model directory. "
            "load_artifacts() must be called once at startup."
        )

    model_dir = os.path.join(base_dir, "models")
    data_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    joblib.dump(rf, os.path.join(model_dir, "final_rf_regressor.pkl"))
    joblib.dump(logit, os.path.join(model_dir, "final_logit_riskflag.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "final_riskflag_scaler.pkl"))
    joblib.dump(config, os.path.join(model_dir, "final_config.pkl"))
    monthly.to_csv(os.path.join(data_dir, "monthly_features.csv"), index=False)

    # 9. Hot-swap in-memory artifacts
    _artifacts["rf"] = rf
    _artifacts["logit"] = logit
    _artifacts["scaler"] = scaler
    _artifacts["config"] = config
    _artifacts["monthly_df"] = monthly
    _artifacts["background_X"] = background_X
    _artifacts["error"] = None

    # 10. In-sample metrics for the API response
    y_pred_reg = rf.predict(X)
    y_pred_cls = logit.predict(X_scaled)

    return {
        "trained_at": config["trained_at"],
        "n_samples": int(len(monthly)),
        "n_months": int(len(monthly)),
        "risk_threshold_cases": threshold_cases,
        "mae": float(mean_absolute_error(y_reg, y_pred_reg)),
        "r2": float(r2_score(y_reg, y_pred_reg)),
        "risk_accuracy": float(accuracy_score(y_cls, y_pred_cls)),
    }