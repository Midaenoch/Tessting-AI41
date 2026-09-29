import os
import shutil
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

# --- Retrain safety (versioned backups + held-out evaluation gate) --------
HOLDOUT_MONTHS = 6              # most recent N months withheld from training, used to score the candidate
REGRESSION_TOLERANCE = 1.10     # candidate MAE may be up to 10% worse than current model before it's blocked
MODEL_FILES = [
    "final_rf_regressor.pkl",
    "final_logit_riskflag.pkl",
    "final_riskflag_scaler.pkl",
    "final_config.pkl",
]

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


def get_config_summary() -> Optional[dict]:
    """Small, JSON-safe summary of the currently-live model's config."""
    config = _artifacts.get("config")
    if not config:
        return None
    return {
        "trained_at": config.get("trained_at"),
        "n_samples": config.get("n_samples"),
        "risk_flag_threshold_cases": config.get("risk_flag_threshold_cases"),
        "candidate_holdout_mae": config.get("candidate_holdout_mae"),
    }


async def get_model_history(db, limit: int = 50) -> list[dict]:
    """Most recent model events (retrain/rollback), newest first."""
    if db is None:
        return []
    cursor = db.model_events.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)
    return await cursor.to_list(length=limit)


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


def _model_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "models")


def archive_current_models(base_dir: str) -> Optional[str]:
    """Copy the current .pkl files into a timestamped archive folder before
    they get overwritten. Returns the archive path, or None if there was
    nothing on disk yet to archive (e.g. very first training run)."""
    model_dir = _model_dir(base_dir)
    if not any(os.path.exists(os.path.join(model_dir, f)) for f in MODEL_FILES):
        return None

    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    archive_dir = os.path.join(model_dir, "archive", stamp)
    os.makedirs(archive_dir, exist_ok=True)

    for fname in MODEL_FILES:
        src = os.path.join(model_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(archive_dir, fname))

    return archive_dir


def list_archived_versions(base_dir: str) -> list[str]:
    """Timestamps of archived model generations, most recent first."""
    archive_root = os.path.join(_model_dir(base_dir), "archive")
    if not os.path.isdir(archive_root):
        return []
    return sorted(os.listdir(archive_root), reverse=True)


async def log_model_event(db, event: dict) -> None:
    """Persist one model-lifecycle event (retrain or rollback) to MongoDB.

    Never raises — a logging failure should not block or fail the retrain/
    rollback it's trying to record. If db is None (no DB connection), the
    event is silently skipped.
    """
    if db is None:
        return
    try:
        await db.model_events.insert_one(event)
    except Exception as e:
        print(f"[model_events] WARNING: failed to log event: {e}")


def _user_summary(user: Optional[dict]) -> Optional[dict]:
    if not user:
        return None
    return {"id": user.get("id"), "email": user.get("email"), "role": user.get("role")}


async def rollback_to_version(
    base_dir: str,
    version: Optional[str] = None,
    db=None,
    triggered_by: Optional[dict] = None,
) -> dict:
    """Restore an archived model generation as the live one.

    version=None rolls back to the most recent archived snapshot (i.e. the
    generation that was live immediately before the last retrain).
    """
    versions = list_archived_versions(base_dir)
    if not versions:
        raise RuntimeError("No archived model versions to roll back to")

    target = version or versions[0]
    if target not in versions:
        raise RuntimeError(f"Unknown archived version '{target}'. Available: {versions}")

    model_dir = _model_dir(base_dir)
    archive_dir = os.path.join(model_dir, "archive", target)

    # Safety copy of what's about to be replaced, so a bad rollback is also reversible.
    archive_current_models(base_dir)

    for fname in MODEL_FILES:
        src = os.path.join(archive_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(model_dir, fname))

    # Reload the restored files into memory so the swap takes effect immediately.
    load_artifacts(base_dir)
    if get_error():
        raise RuntimeError(f"Rollback restored files but failed to reload them: {get_error()}")

    result = {"rolled_back_to": target, "available_versions": versions}

    await log_model_event(db, {
        "event_type": "rollback",
        "timestamp": datetime.utcnow(),
        "triggered_by": _user_summary(triggered_by),
        "rolled_back_to": target,
        "available_versions": versions,
    })

    return result


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


async def retrain(db, triggered_by: Optional[dict] = None) -> dict:
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

    # 3. Time-aware holdout split — the last HOLDOUT_MONTHS months are withheld
    #    from training entirely, so evaluation reflects genuine forecasting
    #    skill rather than the model grading its own training data.
    n_holdout = min(HOLDOUT_MONTHS, max(len(monthly) - MIN_MONTHS_FOR_TRAIN, 0))
    if n_holdout < 2:
        raise RuntimeError(
            f"Not enough months ({len(monthly)}) to both train "
            f"(min {MIN_MONTHS_FOR_TRAIN}) and hold out a meaningful "
            f"evaluation window."
        )

    train_df = monthly.iloc[:-n_holdout]
    holdout_df = monthly.iloc[-n_holdout:]

    X_train = train_df[FORECAST_FEATURES].values
    y_train_reg = train_df["case_count"].values
    X_holdout = holdout_df[FORECAST_FEATURES].values
    y_holdout_reg = holdout_df["case_count"].values

    # 4. Train the candidate regressor on train-only data
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train_reg)

    # 5. Honest evaluation: score on the held-out months the model never saw
    candidate_holdout_mae = float(mean_absolute_error(y_holdout_reg, rf.predict(X_holdout)))
    candidate_holdout_r2 = float(r2_score(y_holdout_reg, rf.predict(X_holdout))) if n_holdout > 1 else None

    # 6. Compare against the currently-live model on the SAME held-out months,
    #    so this is an apples-to-apples comparison, not just a fixed threshold.
    base_dir = _artifacts.get("_base_dir")
    if base_dir is None:
        raise RuntimeError(
            "Cannot determine model directory. "
            "load_artifacts() must be called once at startup."
        )

    current_holdout_mae = None
    if is_loaded():
        try:
            current_holdout_mae = float(
                mean_absolute_error(y_holdout_reg, _artifacts["rf"].predict(X_holdout))
            )
        except Exception:
            # Old model may have a different feature contract (e.g. very first
            # retrain after a schema change) — treat as "no baseline available".
            current_holdout_mae = None

    promoted = True
    reason = "No previous model to compare against — promoting first trained model."
    if current_holdout_mae is not None:
        if candidate_holdout_mae > current_holdout_mae * REGRESSION_TOLERANCE:
            promoted = False
            reason = (
                f"Candidate held-out MAE ({candidate_holdout_mae:.2f}) is worse than the "
                f"current model's held-out MAE ({current_holdout_mae:.2f}) by more than "
                f"the {int((REGRESSION_TOLERANCE - 1) * 100)}% tolerance. Not promoted; "
                f"current model remains live."
            )
        else:
            reason = (
                f"Candidate held-out MAE ({candidate_holdout_mae:.2f}) is within tolerance "
                f"of the current model's ({current_holdout_mae:.2f}). Promoted."
            )

    result = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "n_samples": int(len(monthly)),
        "n_months": int(len(monthly)),
        "n_holdout_months": int(n_holdout),
        "promoted": promoted,
        "reason": reason,
        "candidate_holdout_mae": candidate_holdout_mae,
        "candidate_holdout_r2": candidate_holdout_r2,
        "current_holdout_mae": current_holdout_mae,
    }

    if not promoted:
        # Nothing on disk or in memory changes — current model keeps serving.
        await log_model_event(db, {
            "event_type": "retrain",
            "timestamp": datetime.utcnow(),
            "triggered_by": _user_summary(triggered_by),
            **result,
        })
        return result

    # 7. Only now — having decided this candidate is good enough — retrain on
    #    ALL available months (train + holdout) so the promoted model doesn't
    #    waste the most recent, most relevant months, then fit the risk
    #    classifier and persist everything.
    X_full = monthly[FORECAST_FEATURES].values
    y_full_reg = monthly["case_count"].values

    rf_final = RandomForestRegressor(
        n_estimators=300, max_depth=None, min_samples_leaf=2, random_state=42, n_jobs=-1,
    )
    rf_final.fit(X_full, y_full_reg)

    threshold_cases = float(np.quantile(y_full_reg, RISK_FLAG_THRESHOLD_QUANTILE))
    y_cls = (y_full_reg >= threshold_cases).astype(int)
    if len(np.unique(y_cls)) < 2:
        threshold_cases = float(np.median(y_full_reg))
        y_cls = (y_full_reg >= threshold_cases).astype(int)

    scaler = StandardScaler().fit(X_full)
    X_scaled = scaler.transform(X_full)
    logit = LogisticRegression(max_iter=2000, class_weight="balanced")
    logit.fit(X_scaled, y_cls)

    background_X = scaler.transform(X_full[:BACKGROUND_SAMPLE_SIZE])

    config = {
        "features": FORECAST_FEATURES,
        "risk_flag_threshold_cases": threshold_cases,
        "risk_flag_threshold_quantile": RISK_FLAG_THRESHOLD_QUANTILE,
        "trained_at": result["trained_at"],
        "n_samples": int(len(monthly)),
        "n_features": len(FORECAST_FEATURES),
        "candidate_holdout_mae": candidate_holdout_mae,
    }

    # 8. Archive whatever is currently live BEFORE overwriting it — this is
    #    the versioned backup that makes rollback possible.
    archived_to = archive_current_models(base_dir)
    result["archived_previous_version"] = archived_to

    model_dir = os.path.join(base_dir, "models")
    data_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    joblib.dump(rf_final, os.path.join(model_dir, "final_rf_regressor.pkl"))
    joblib.dump(logit, os.path.join(model_dir, "final_logit_riskflag.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "final_riskflag_scaler.pkl"))
    joblib.dump(config, os.path.join(model_dir, "final_config.pkl"))
    monthly.to_csv(os.path.join(data_dir, "monthly_features.csv"), index=False)

    # 9. Hot-swap in-memory artifacts — only reached once the model has
    #     passed the gate above.
    _artifacts["rf"] = rf_final
    _artifacts["logit"] = logit
    _artifacts["scaler"] = scaler
    _artifacts["config"] = config
    _artifacts["monthly_df"] = monthly
    _artifacts["background_X"] = background_X
    _artifacts["error"] = None

    y_pred_cls = logit.predict(X_scaled)
    result["risk_threshold_cases"] = threshold_cases
    result["risk_accuracy"] = float(accuracy_score(y_cls, y_pred_cls))
    result["full_train_r2"] = float(r2_score(y_full_reg, rf_final.predict(X_full)))

    await log_model_event(db, {
        "event_type": "retrain",
        "timestamp": datetime.utcnow(),
        "triggered_by": _user_summary(triggered_by),
        **result,
    })

    return result
