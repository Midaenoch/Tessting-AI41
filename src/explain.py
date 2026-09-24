"""
AI4Lassa — src/explain.py

Genuine per-prediction explanations for both models, replacing the earlier
global-importance-only approach.

- Random Forest (case-count forecast): shap.TreeExplainer — exact for tree
  ensembles, no approximation.
- Logistic Regression (risk flag): shap.LinearExplainer — exact for linear
  models given the independence-masker background.

Both are sanity-checked at call time: base_value + sum(shap_values) must
equal the model's actual output for that row, or something is wrong.
"""
import numpy as np
import pandas as pd
import shap

FULL_FEATURES = [
    "case_count", "case_count_lag1", "case_count_lag2", "case_count_lag3",
    "case_count_lag6", "case_count_lag12",
    "case_count_roll3_mean", "case_count_roll6_mean", "case_count_roll3_max",
    "case_growth_lag1", "positivity_rate_lag1",
    "month_sin", "month_cos", "year",
]

# Plain-English names for each feature — shown to end users instead of the
# raw column names.
FRIENDLY_NAMES = {
    "case_count": "this month's case count",
    "case_count_lag1": "last month's case count",
    "case_count_lag2": "cases 2 months ago",
    "case_count_lag3": "cases 3 months ago",
    "case_count_lag6": "cases 6 months ago",
    "case_count_lag12": "cases this time last year",
    "case_count_roll3_mean": "the average case count over the last 3 months",
    "case_count_roll6_mean": "the average case count over the last 6 months",
    "case_count_roll3_max": "the highest monthly case count in the last 3 months",
    "case_growth_lag1": "how fast cases were rising or falling recently",
    "positivity_rate_lag1": "the share of tested samples that came back positive last month",
    "month_sin": "the time of year (seasonal pattern)",
    "month_cos": "the time of year (seasonal pattern)",
    "year": "the year",
}


def _friendly_name(feature: str) -> str:
    return FRIENDLY_NAMES.get(feature, feature.replace("_", " "))


def _magnitude_word(fraction_of_base: float) -> str:
    """Converts a relative impact size into a plain-language intensity word."""
    fraction_of_base = abs(fraction_of_base)
    if fraction_of_base < 0.05:
        return "slightly"
    if fraction_of_base < 0.20:
        return "moderately"
    return "strongly"


# Features whose raw numeric value is meaningless to a non-technical reader
# (cyclical encodings, mostly) — shown without a value in parentheses.
_NO_VALUE_SHOWN = {"month_sin", "month_cos"}


def _formatted_value(feature: str, value: float) -> str:
    if feature == "positivity_rate_lag1":
        return f"{value:.0%}"
    if feature == "year":
        return f"{int(value)}"
    return f"{value:.0f}"


def _describe_factor(feature: str, value: float) -> str:
    """The 'name (value)' phrase used in a sentence, e.g. 'last month's case
    count (169)' — or just the name alone for features with no meaningful
    raw value to show."""
    name = _friendly_name(feature)
    if feature in _NO_VALUE_SHOWN:
        return name
    return f"{name} ({_formatted_value(feature, value)})"


def explain_forecast(row: pd.Series, rf, top_n: int = 5) -> dict:
    """Per-prediction SHAP explanation for the Random Forest case-count forecast."""
    X = pd.DataFrame([row[FULL_FEATURES].values], columns=FULL_FEATURES)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)[0]
    base_value = float(np.array(explainer.expected_value).flatten()[0])
    prediction = float(rf.predict(X)[0])

    contrib = pd.Series(shap_values, index=FULL_FEATURES).sort_values(key=abs, ascending=False)
    factors = []
    for feat, val in contrib.head(top_n).items():
        direction = "increases" if val > 0 else "decreases"
        factors.append({
            "feature": feat,
            "value": float(row[feat]),
            "direction": direction,
            "impact_cases": round(float(abs(val)), 1),
        })

    # sanity check — should always hold for TreeExplainer to a small numerical tolerance
    check = abs((base_value + shap_values.sum()) - prediction) < 1e-6

    return {
        "base_value": round(base_value, 1),
        "prediction": round(prediction, 1),
        "top_factors": factors,
        "sanity_check_passed": bool(check),
    }


def explain_risk_flag(row: pd.Series, logit, scaler, background_X_scaled, top_n: int = 5) -> dict:
    """Per-prediction SHAP explanation for the Logistic Regression risk classifier,
    in log-odds space. `background_X_scaled` should be a sample of scaled training
    rows (used as the masker's reference distribution)."""
    X = pd.DataFrame([row[FULL_FEATURES].values], columns=FULL_FEATURES)
    X_scaled = scaler.transform(X)

    masker = shap.maskers.Independent(background_X_scaled, max_samples=100)
    explainer = shap.LinearExplainer(logit, masker)
    shap_values = explainer.shap_values(X_scaled)[0]
    base_value = float(np.array(explainer.expected_value).flatten()[0])

    proba = float(logit.predict_proba(X_scaled)[0, 1])
    logodds = float(np.log(proba / (1 - proba)))

    contrib = pd.Series(shap_values, index=FULL_FEATURES).sort_values(key=abs, ascending=False)
    factors = []
    for feat, val in contrib.head(top_n).items():
        direction = "increases" if val > 0 else "decreases"
        factors.append({
            "feature": feat,
            "value": float(row[feat]),
            "direction": direction,
            "impact_log_odds": round(float(abs(val)), 3),
        })

    check = abs((base_value + shap_values.sum()) - logodds) < 1e-6

    return {
        "base_log_odds": round(base_value, 3),
        "prediction_log_odds": round(logodds, 3),
        "prediction_probability": round(proba, 3),
        "top_factors": factors,
        "sanity_check_passed": bool(check),
    }


# ---------------------------------------------------------------------------
# Plain-language wrappers — what end users actually see. Translates the
# technical SHAP output above into ordinary sentences, and turns a failed
# internal sanity check into a clear "couldn't generate an explanation"
# message rather than silently showing untrustworthy numbers.
# ---------------------------------------------------------------------------
class ExplanationError(Exception):
    """Raised when an explanation can't be safely generated. Carries a
    plain-language message (`.user_message`) suitable for showing directly
    to a non-technical user, and the original technical error for logs."""
    def __init__(self, user_message: str, technical_detail: str = ""):
        self.user_message = user_message
        self.technical_detail = technical_detail
        super().__init__(user_message)


def explain_forecast_plain(row: pd.Series, rf, top_n: int = 3) -> dict:
    """Plain-language version of explain_forecast(). Raises ExplanationError
    with a friendly message if anything goes wrong or the internal
    consistency check fails, instead of returning numbers that can't be
    trusted."""
    try:
        result = explain_forecast(row, rf, top_n=top_n)
    except Exception as e:
        raise ExplanationError(
            "We couldn't work out the forecast for this month. This is usually "
            "a data problem (a missing or unexpected value) rather than "
            "anything wrong with the model itself — try a different month, "
            "or check that the input data looks complete.",
            technical_detail=str(e),
        )

    if not result["sanity_check_passed"]:
        raise ExplanationError(
            "We generated a forecast, but our internal double-check found the "
            "explanation didn't add up correctly, so we're not showing it "
            "rather than risk showing something misleading. The forecast "
            "number itself is still reliable — this only affects the "
            "'why' explanation.",
            technical_detail="SHAP sanity check failed for forecast explanation",
        )

    base = result["base_value"]
    sentences = []
    for f in result["top_factors"]:
        phrase = _describe_factor(f["feature"], f["value"])
        word = _magnitude_word(f["impact_cases"] / base if base else 0)
        verb = "pushed the forecast up" if f["direction"] == "increases" else "pulled the forecast down"
        sentences.append(
            f"{phrase.capitalize()} {word} {verb} by about {round(f['impact_cases'])} cases."
        )

    return {
        "forecast_cases": round(result["prediction"]),
        "explanation_sentences": sentences,
    }


def explain_risk_flag_plain(row: pd.Series, logit, scaler, background_X_scaled, top_n: int = 3) -> dict:
    """Plain-language version of explain_risk_flag(). Same safety behavior
    as explain_forecast_plain(): raises ExplanationError rather than
    returning an explanation that failed its consistency check."""
    try:
        result = explain_risk_flag(row, logit, scaler, background_X_scaled, top_n=top_n)
    except Exception as e:
        raise ExplanationError(
            "We couldn't work out the risk probability for this month. This "
            "is usually a data problem rather than anything wrong with the "
            "model itself — try a different month, or check that the input "
            "data looks complete.",
            technical_detail=str(e),
        )

    if not result["sanity_check_passed"]:
        raise ExplanationError(
            "We generated a risk probability, but our internal double-check "
            "found the explanation didn't add up correctly, so we're not "
            "showing it rather than risk showing something misleading. The "
            "risk probability itself is still reliable — this only affects "
            "the 'why' explanation.",
            technical_detail="SHAP sanity check failed for risk-flag explanation",
        )

    sentences = []
    for f in result["top_factors"]:
        phrase = _describe_factor(f["feature"], f["value"])
        word = _magnitude_word(f["impact_log_odds"] / 2.0)  # log-odds ~2 is already a strong swing
        verb = "pushed the risk higher" if f["direction"] == "increases" else "pushed the risk lower"
        sentences.append(f"{phrase.capitalize()} {word} {verb}.")

    return {
        "probability_percent": round(result["prediction_probability"] * 100, 1),
        "explanation_sentences": sentences,
    }
