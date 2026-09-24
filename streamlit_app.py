# app.py
import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import joblib

BASE_DIR = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)  # so `from src.explain import ...` resolves

FORECAST_FEATURES = [
    "case_count", "case_count_lag1", "case_count_lag2", "case_count_lag3",
    "case_count_lag6", "case_count_lag12",
    "case_count_roll3_mean", "case_count_roll6_mean", "case_count_roll3_max",
    "case_growth_lag1", "positivity_rate_lag1",
    "month_sin", "month_cos", "year",
]

st.set_page_config(page_title="AI4Lassa", layout="wide")
st.title("📈 AI4Lassa — National Outbreak Early-Warning")


@st.cache_resource
def load_forecast_artifacts():
    """Loads the final trained models. Returns (None, missing_files, load_error)
    so the caller can show a specific, helpful message for whichever failure
    mode actually happened."""
    model_dir = os.path.join(BASE_DIR, "models")
    data_path = os.path.join(BASE_DIR, "data", "processed", "monthly_features.csv")

    required = [
        os.path.join(model_dir, "final_rf_regressor.pkl"),
        os.path.join(model_dir, "final_logit_riskflag.pkl"),
        os.path.join(model_dir, "final_riskflag_scaler.pkl"),
        os.path.join(model_dir, "final_config.pkl"),
        data_path,
    ]
    missing = [os.path.basename(p) for p in required if not os.path.exists(p)]
    if missing:
        return None, missing, None

    try:
        rf = joblib.load(os.path.join(model_dir, "final_rf_regressor.pkl"))
        logit = joblib.load(os.path.join(model_dir, "final_logit_riskflag.pkl"))
        fc_scaler = joblib.load(os.path.join(model_dir, "final_riskflag_scaler.pkl"))
        config = joblib.load(os.path.join(model_dir, "final_config.pkl"))
        monthly_df = pd.read_csv(data_path)

        # Background sample for SHAP's LinearExplainer masker (reference distribution
        # for "what's a typical value for this feature")
        background_X_scaled = fc_scaler.transform(monthly_df[FORECAST_FEATURES].iloc[:80])
    except Exception as e:
        return None, None, str(e)

    return (rf, logit, fc_scaler, config, monthly_df, background_X_scaled), None, None


def forecast_from_row(row, rf, logit, fc_scaler, background_X_scaled, decision_threshold):
    from src.explain import explain_forecast_plain, explain_risk_flag_plain, ExplanationError

    fc = explain_forecast_plain(row, rf)
    rk = explain_risk_flag_plain(row, logit, fc_scaler, background_X_scaled)

    forecast = fc["forecast_cases"]
    proba = rk["probability_percent"] / 100
    risk_level = (
        "HIGH" if proba >= decision_threshold
        else ("ELEVATED" if proba >= decision_threshold * 0.5 else "LOW")
    )

    return forecast, proba, risk_level, fc["explanation_sentences"], rk["explanation_sentences"]


st.markdown(
    "Forecasts **national next-month Lassa fever case volume** from historical "
    "case counts and seasonality. See caveats below before treating this as an "
    "alert system."
)

artifacts, missing_files, load_error = load_forecast_artifacts()

if missing_files:
    st.error(
        "⚠️ This app can't find its model files, so it has nothing to forecast with.\n\n"
        f"**Missing:** {', '.join(missing_files)}\n\n"
        "**What this usually means:** the `models/` folder and "
        "`data/processed/monthly_features.csv` weren't deployed alongside "
        "`streamlit_app.py` — check that both were included when this app "
        "was pushed/uploaded to its hosting platform."
    )
    st.stop()

if load_error:
    st.error(
        "⚠️ The model files were found, but something went wrong while loading "
        "them, so the app can't run right now.\n\n"
        f"**Technical detail (for whoever's debugging this):** {load_error}\n\n"
        "**What this usually means:** the files may be corrupted, incomplete, "
        "or saved with a different version of scikit-learn than what's "
        "installed here. Try re-downloading/re-generating the model files, "
        "or check that `requirements.txt`'s scikit-learn version matches "
        "what the models were trained with."
    )
    st.stop()

rf, logit, fc_scaler, config, monthly_df, background_X_scaled = artifacts

st.info(
    "**Read before relying on this tool:** the case-count forecast tested well "
    "on 2024–2025 held-out data (MAE ≈ 49 cases, R² ≈ 0.57). The HIGH/ELEVATED/LOW "
    "risk label is more provisional — it missed the one true high-risk month in "
    "that same test period at the default threshold, and its decision threshold "
    "was chosen after inspecting that test result rather than independently "
    "validated. Treat the risk label as a soft, exploratory signal, and the case-count "
    "forecast as the primary, more trustworthy output. This is decision-support, not "
    "a diagnosis or a confirmed outbreak declaration."
)

st.subheader("Forecast for the next available month")

monthly_df = monthly_df.sort_values("month_ts")
latest_row = monthly_df.iloc[-1]

threshold = st.slider(
    "Risk-flag decision threshold (probability)",
    min_value=0.05, max_value=0.95, value=0.15, step=0.05,
    help=(
        "Lower = flags more months as risky (higher recall, more false alarms). "
        "0.5 was the default used in validation; 0.15 is the provisional, "
        "recall-favoring value discussed in the project report — neither is "
        "independently confirmed on new data yet."
    ),
)

if st.button("🔮 Generate Forecast", type="primary"):
    from src.explain import ExplanationError

    try:
        with st.spinner("Running forecast..."):
            forecast, proba, risk_level, forecast_factors, risk_factors = forecast_from_row(
                latest_row, rf, logit, fc_scaler, background_X_scaled, threshold
            )
    except ExplanationError as e:
        st.error(
            f"⚠️ Couldn't complete the forecast. {e.user_message}"
        )
        st.stop()

    st.markdown(f"**Based on data through:** {latest_row['month']}")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Forecast — next month's national case count", f"{round(forecast)}")
    with col2:
        st.metric("Provisional outbreak probability", f"{proba:.1%}")

    if risk_level == "HIGH":
        st.error(f"🚨 Risk level: **{risk_level}** (provisional — see caveats above)")
    elif risk_level == "ELEVATED":
        st.warning(f"⚠️ Risk level: **{risk_level}** (provisional — see caveats above)")
    else:
        st.success(f"✅ Risk level: **{risk_level}** (provisional — see caveats above)")

    st.caption(
        "The explanations below are specific to this month, not a fixed list — "
        "they'll change as the underlying numbers change."
    )
    if forecast_factors:
        st.markdown("**Why this case-count forecast (a pattern the model found, not a proven cause):**")
        for f in forecast_factors:
            st.markdown(f"- {f}")
    if risk_factors:
        st.markdown("**Why this risk probability:**")
        for f in risk_factors:
            st.markdown(f"- {f}")

with st.expander("📊 Recent monthly case counts"):
    st.dataframe(
        monthly_df[["month", "case_count", "positivity_rate"]].tail(12).reset_index(drop=True)
    )

with st.expander("ℹ️ About this model"):
    st.markdown(
        f"""
- **Model:** {config.get('regression_model', 'Random Forest Regressor')} for case-count forecast;
  {config.get('classifier_model', 'Logistic Regression')} for the risk flag.
- **Trained on:** {config.get('train_period', '—')} + {config.get('validation_period', '—')}
- **Tested on:** {config.get('test_period', '—')} (held out from all model selection)
- **Test performance:** MAE = {config.get('test_regression_mae', '—')},
  R² = {config.get('test_regression_r2', '—')}
- **Risk threshold definition:** {config.get('risk_threshold_definition', '—')}
- **Known limitation:** {config.get('classifier_threshold_caveat', '—')}

Full methodology, benchmarking, and honest limitations are in `AI4Lassa_Final_Report.md`
and the `notebooks/` folder in this repository.
        """
    )
