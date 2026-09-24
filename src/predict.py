"""
AI4Lassa — predict.py

Produces a 1-month-ahead national case-count forecast, a provisional
risk label, and a plain-language explanation of the main contributing
factors, for the most recent month in the processed feature table
(or any row passed in).

This is a decision-support output, not a diagnosis or a guarantee.
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)  # so `from src.explain import ...` resolves when run directly

MODEL_DIR = os.path.join(_PROJECT_ROOT, "models")
DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "processed", "monthly_features.csv")


def load_artifacts():
    try:
        rf = joblib.load(f"{MODEL_DIR}/final_rf_regressor.pkl")
        logit = joblib.load(f"{MODEL_DIR}/final_logit_riskflag.pkl")
        scaler = joblib.load(f"{MODEL_DIR}/final_riskflag_scaler.pkl")
        config = joblib.load(f"{MODEL_DIR}/final_config.pkl")
        df = pd.read_csv(DATA_PATH).sort_values("month_ts")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Couldn't find a required model or data file ({e}). "
            f"Expected to find the trained models under {MODEL_DIR} and the "
            f"processed dataset at {DATA_PATH} — make sure notebook 01 "
            "(preprocessing) and notebook 05 (final model training) have "
            "both been run, and that this script is being run from within "
            "the project folder."
        ) from e

    background_X_scaled = scaler.transform(df[config["features"]].iloc[:80])
    return rf, logit, scaler, config, background_X_scaled


def predict_next_month(row: pd.Series, rf, logit, scaler, config, background_X_scaled,
                        decision_threshold: float = None):
    from src.explain import explain_forecast_plain, explain_risk_flag_plain, ExplanationError

    threshold = decision_threshold if decision_threshold is not None \
        else config["classifier_decision_threshold_default"]

    try:
        fc = explain_forecast_plain(row, rf)
        rk = explain_risk_flag_plain(row, logit, scaler, background_X_scaled)
    except ExplanationError as e:
        # Surface the plain-language message as the result itself, rather
        # than letting the exception propagate as a raw traceback.
        return {
            "error": True,
            "message": e.user_message,
        }

    forecast = fc["forecast_cases"]
    proba_high_risk = rk["probability_percent"] / 100
    risk_level = "HIGH" if proba_high_risk >= threshold else \
                 ("ELEVATED" if proba_high_risk >= threshold * 0.5 else "LOW")

    return {
        "error": False,
        "forecast_next_month_cases": round(forecast),
        "outbreak_probability_provisional": round(proba_high_risk, 3),
        "risk_level_provisional": risk_level,
        "decision_threshold_used": threshold,
        "why_this_forecast": fc["explanation_sentences"],
        "why_this_risk_level": rk["explanation_sentences"],
        "note": "Risk level is provisional and based on limited historical high-risk "
                "events (see README limitations). This is a decision-support signal, "
                "not a diagnosis or confirmed outbreak declaration.",
    }


if __name__ == "__main__":
    try:
        rf, logit, scaler, config, background_X_scaled = load_artifacts()
    except FileNotFoundError as e:
        print(f"❌ Couldn't start: {e}")
        raise SystemExit(1)

    df = pd.read_csv(DATA_PATH)
    latest_row = df.iloc[-1]
    result = predict_next_month(latest_row, rf, logit, scaler, config, background_X_scaled,
                                 decision_threshold=0.15)

    if result["error"]:
        print(f"⚠️ Couldn't generate a forecast: {result['message']}")
    else:
        print(f"Forecast based on month: {latest_row['month']}")
        for k, v in result.items():
            if k == "error":
                continue
            print(f"  {k}: {v}")
