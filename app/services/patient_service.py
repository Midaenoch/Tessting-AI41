"""
app/services/patient_service.py

Individual patient outcome risk model — genuinely trained on the real
confirmed-case line-list (age, sex, presenting symptoms → survived / died),
not the old, unaudited per-patient SVM that shipped with the original repo
(see legacy_svm_model/ — archived, never loaded, training data unrecoverable).

This predicts a *risk signal* for a single confirmed or suspected Lassa
fever patient at presentation, using the same discipline already applied to
the national forecast model:
  - time-aware train/holdout split (by YEAR — never randomly)
  - genuine per-prediction SHAP explanations
  - a promotion gate: a retrained candidate only replaces the live model if
    it isn't meaningfully worse on held-out patients
  - versioned backups + rollback
  - every retrain/rollback logged to db.model_events (same collection the
    forecast model uses, distinguished by "model_type")

IMPORTANT — this is decision support, not a diagnosis. See the `note` field
returned with every prediction.
"""
import os
import shutil
from datetime import datetime
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

# Feature contract
# Raw source column -> clean feature key. Order here fixes FEATURE order
# everywhere else in this module.
SYMPTOM_COLUMNS: dict[str, str] = {
    "GENERAL WEAKNESS": "general_weakness",
    "FEVER": "fever",
    "COUGH": "cough",
    "ABDOMINAL PAIN": "abdominal_pain",
    "HEADACHE": "headache",
    "CHEST PAIN": "chest_pain",
    "NAUSEA": "nausea",
    "VOMITING": "vomiting",
    "DIARRHOEA": "diarrhoea",
    "DIFFICULTY IN BREATHING": "difficulty_breathing",
    "CATARRH": "catarrh",
    "DIFFICULTY IN SWALLOWING/ SORE THROAT": "sore_throat",
    "JOINT PAIN": "joint_pain",
    "RED EYES": "red_eyes",
    "JAUNDICE": "jaundice",
    "OEDEMA": "oedema",
    "BLEEDING FROM INJECTION SITE": "bleeding_injection_site",
    "BLEEDING FROM THE NOSE": "bleeding_nose",
    "BLEEDING OF THE GUMS OR MOUTH": "bleeding_gums",
    "UNEXPLAINED BLEEDING FROM ANY SITE": "unexplained_bleeding",
    "PASSING OUT OF COKE COLOURED URINE": "coke_coloured_urine",
    "PASSING OUT OF BLOOD IN FAECES": "blood_in_faeces",
    "SEIZURE": "seizure",
    "DEAFNESS": "deafness",
    "PAIN BEHIND EYES/SENSITIVE TO LIGHT": "photophobia",
    "HICCUPS": "hiccups",
    "SKIN RASH": "skin_rash",
    "COMA/UNCONSIOUS": "coma",
    "CONFUSED OR DISORIENTED": "confused",
    "CHILLS": "chills",
}

SYMPTOM_FEATURES = list(SYMPTOM_COLUMNS.values())
FEATURES = ["age", "sex_male"] + SYMPTOM_FEATURES

FRIENDLY_NAMES = {
    "age": "age",
    "sex_male": "being male",
    "general_weakness": "general weakness",
    "fever": "fever",
    "cough": "cough",
    "abdominal_pain": "abdominal pain",
    "headache": "headache",
    "chest_pain": "chest pain",
    "nausea": "nausea",
    "vomiting": "vomiting",
    "diarrhoea": "diarrhoea",
    "difficulty_breathing": "difficulty breathing",
    "catarrh": "catarrh",
    "sore_throat": "sore throat / difficulty swallowing",
    "joint_pain": "joint pain",
    "red_eyes": "red eyes",
    "jaundice": "jaundice",
    "oedema": "oedema (swelling)",
    "bleeding_injection_site": "bleeding from an injection site",
    "bleeding_nose": "nosebleed",
    "bleeding_gums": "bleeding gums or mouth",
    "unexplained_bleeding": "unexplained bleeding",
    "coke_coloured_urine": "dark (coke-coloured) urine",
    "blood_in_faeces": "blood in stool",
    "seizure": "seizure",
    "deafness": "deafness",
    "photophobia": "sensitivity to light / pain behind the eyes",
    "hiccups": "hiccups",
    "skin_rash": "skin rash",
    "coma": "coma / unconsciousness",
    "confused": "confusion or disorientation",
    "chills": "chills",
}

MIN_CASES_FOR_TRAIN = 200
HOLDOUT_YEARS = 2          # most recent N calendar years withheld from training
REGRESSION_TOLERANCE = 1.10  # candidate may be up to 10% worse (by AUC-complement) before blocked
BACKGROUND_SAMPLE_SIZE = 100

MODEL_FILES = ["patient_risk_logit.pkl", "patient_risk_scaler.pkl", "patient_config.pkl"]

_artifacts: dict = {"_base_dir": None}


# ===========================================================================
# Loading (mirrors forecast_service's startup pattern)
# ===========================================================================
def load_artifacts(base_dir: str) -> None:
    _artifacts["_base_dir"] = base_dir
    model_dir = os.path.join(base_dir, "models", "patient")
    try:
        _artifacts["logit"] = joblib.load(os.path.join(model_dir, "patient_risk_logit.pkl"))
        _artifacts["scaler"] = joblib.load(os.path.join(model_dir, "patient_risk_scaler.pkl"))
        _artifacts["config"] = joblib.load(os.path.join(model_dir, "patient_config.pkl"))
        _artifacts["error"] = None
    except Exception as e:
        _artifacts["logit"] = None
        _artifacts["scaler"] = None
        _artifacts["config"] = None
        _artifacts["error"] = str(e)


def is_loaded() -> bool:
    return _artifacts.get("logit") is not None


def get_error() -> Optional[str]:
    return _artifacts.get("error")


def get_config_summary() -> Optional[dict]:
    config = _artifacts.get("config")
    if not config:
        return None
    return {
        "trained_at": config.get("trained_at"),
        "n_samples": config.get("n_samples"),
        "n_deaths": config.get("n_deaths"),
        "holdout_auc": config.get("holdout_auc"),
    }


# ===========================================================================
# Feature engineering
# ===========================================================================
def _clean_yes_no(value) -> int:
    """Maps messy Yes/No variants to 1/0. Missing/unrecognized -> 0 (treated
    as 'not reported', which is the honest reading of this dataset — see
    Technical Documentation caveats)."""
    if value is None:
        return 0
    s = str(value).strip().lower()
    return 1 if s in ("yes", "y", "1", "true") else 0


def _clean_sex_male(value) -> Optional[int]:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in ("m", "male"):
        return 1
    if s in ("f", "female"):
        return 0
    return None


def build_features_from_excel_df(df: pd.DataFrame) -> pd.DataFrame:
    """Builds the training table directly from the raw line-list spreadsheet.
    Used for the initial (offline) training run — see train_patient_model.py.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    result = df["RESULTS"].astype(str).str.strip().str.lower()
    condition = df["CONDITION"].astype(str).str.strip().str.lower()
    confirmed_mask = result.str.contains("positive", na=False) & condition.isin(["alive", "dead"])
    sub = df[confirmed_mask].copy()

    out = pd.DataFrame(index=sub.index)
    out["year"] = sub["YEAR"]
    out["age"] = pd.to_numeric(sub["AGE in Years"], errors="coerce")
    out["sex_male"] = sub["SEX"].apply(_clean_sex_male)
    for raw_col, feat in SYMPTOM_COLUMNS.items():
        out[feat] = sub[raw_col].apply(_clean_yes_no) if raw_col in sub.columns else 0
    out["died"] = (condition.loc[sub.index] == "dead").astype(int)

    out = out.dropna(subset=["age", "sex_male", "year"])
    out["age"] = out["age"].clip(lower=0, upper=110)
    return out.reset_index(drop=True)


def build_features_from_mongo_docs(docs: list[dict]) -> pd.DataFrame:
    """Builds the training table from db.cases documents (the same
    collection the forecast model reads). Documents must include the
    symptom fields — see the extended _line_list_row() in app/routers/admin.py.
    """
    rows = []
    for d in docs:
        result = str(d.get("result") or "").strip().lower()
        condition = str(d.get("condition") or "").strip().lower()
        if "positive" not in result or condition not in ("alive", "dead"):
            continue
        sex_male = _clean_sex_male(d.get("sex"))
        age = d.get("age")
        if sex_male is None or age is None:
            continue
        row = {
            "year": d.get("year"),
            "age": float(age),
            "sex_male": sex_male,
            "died": 1 if condition == "dead" else 0,
        }
        for feat in SYMPTOM_FEATURES:
            row[feat] = _clean_yes_no(d.get(feat))
        rows.append(row)

    out = pd.DataFrame(rows)
    if len(out):
        out["age"] = out["age"].clip(lower=0, upper=110)
    return out


# ===========================================================================
# Archiving / rollback (same pattern as forecast_service, separate namespace)
# ===========================================================================
def _model_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "models", "patient")


def archive_current_models(base_dir: str) -> Optional[str]:
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
    archive_root = os.path.join(_model_dir(base_dir), "archive")
    if not os.path.isdir(archive_root):
        return []
    return sorted(os.listdir(archive_root), reverse=True)


async def rollback_to_version(base_dir: str, version: Optional[str] = None, db=None, triggered_by: Optional[dict] = None) -> dict:
    versions = list_archived_versions(base_dir)
    if not versions:
        raise RuntimeError("No archived patient-model versions to roll back to")
    target = version or versions[0]
    if target not in versions:
        raise RuntimeError(f"Unknown archived version '{target}'. Available: {versions}")

    model_dir = _model_dir(base_dir)
    archive_dir = os.path.join(model_dir, "archive", target)
    archive_current_models(base_dir)  # safety copy of what's about to be replaced

    for fname in MODEL_FILES:
        src = os.path.join(archive_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(model_dir, fname))

    load_artifacts(base_dir)
    if get_error():
        raise RuntimeError(f"Rollback restored files but failed to reload them: {get_error()}")

    result = {"rolled_back_to": target, "available_versions": versions}

    await log_model_event(db, {
        "event_type": "patient_rollback",
        "model_type": "patient_risk",
        "timestamp": datetime.utcnow(),
        "triggered_by": _user_summary(triggered_by),
        "rolled_back_to": target,
        "available_versions": versions,
    })

    return result


# ===========================================================================
# Training (shared by the initial offline run and retrain())
# ===========================================================================
def _fit_and_score(train_df: pd.DataFrame, holdout_df: pd.DataFrame):
    X_train = train_df[FEATURES].values
    y_train = train_df["died"].values
    X_holdout = holdout_df[FEATURES].values
    y_holdout = holdout_df["died"].values

    scaler = StandardScaler().fit(X_train)
    logit = LogisticRegression(max_iter=2000, class_weight="balanced")
    logit.fit(scaler.transform(X_train), y_train)

    proba_holdout = logit.predict_proba(scaler.transform(X_holdout))[:, 1]
    pred_holdout = (proba_holdout >= 0.5).astype(int)

    metrics = {
        "holdout_auc": float(roc_auc_score(y_holdout, proba_holdout)) if len(np.unique(y_holdout)) > 1 else None,
        "holdout_accuracy": float(accuracy_score(y_holdout, pred_holdout)),
        "holdout_precision_death": float(precision_score(y_holdout, pred_holdout, zero_division=0)),
        "holdout_recall_death": float(recall_score(y_holdout, pred_holdout, zero_division=0)),
        "n_holdout": int(len(holdout_df)),
        "n_holdout_deaths": int(y_holdout.sum()),
    }
    return logit, scaler, metrics


async def log_model_event(db, event: dict) -> None:
    try:
        event.setdefault("timestamp", datetime.utcnow())
        await db.model_events.insert_one(event)
    except Exception as e:
        print(f"[patient_service] WARNING: failed to write audit log entry: {e}")


def _user_summary(user: Optional[dict]) -> Optional[dict]:
    if not user:
        return None
    return {
        "id": str(user.get("_id") or user.get("id") or ""),
        "email": user.get("email"),
        "role": user.get("role"),
    }


async def retrain(db, triggered_by: Optional[dict] = None) -> dict:
    """Retrain the patient outcome model from db.cases. Same safety pattern
    as forecast_service.retrain(): time-aware holdout, promotion gate,
    versioned backup, audit log — see module docstring.
    """
    base_dir = _artifacts.get("_base_dir")
    if base_dir is None:
        raise RuntimeError("load_artifacts() must be called once at startup before retrain().")

    cursor = db.cases.find(
        {"result": {"$exists": True}, "condition": {"$exists": True}},
        {"_id": 0, "year": 1, "age": 1, "sex": 1, "result": 1, "condition": 1,
         **{feat: 1 for feat in SYMPTOM_FEATURES}},
    )
    docs = await cursor.to_list(length=None)

    df = build_features_from_mongo_docs(docs)
    if len(df) < MIN_CASES_FOR_TRAIN:
        raise RuntimeError(
            f"Only {len(df)} usable confirmed cases with a known outcome and "
            f"symptom data in the database; need at least {MIN_CASES_FOR_TRAIN}. "
            f"(Cases uploaded before symptom fields were added to ingestion "
            f"won't have this data — re-upload the line-list to backfill it.)"
        )

    max_year = int(df["year"].max())
    holdout_mask = df["year"] > (max_year - HOLDOUT_YEARS)
    train_df, holdout_df = df[~holdout_mask], df[holdout_mask]
    if len(holdout_df) < 20 or len(train_df) < MIN_CASES_FOR_TRAIN:
        # Not enough spread across years for a meaningful time split — fall
        # back to a stratified split by outcome instead of failing outright.
        from sklearn.model_selection import train_test_split
        train_df, holdout_df = train_test_split(
            df, test_size=0.2, stratify=df["died"], random_state=42
        )

    candidate_logit, candidate_scaler, candidate_metrics = _fit_and_score(train_df, holdout_df)

    promoted = True
    reason = "No previous patient model to compare against — promoting first trained model."
    current_auc = None
    if is_loaded():
        try:
            X_h_scaled = _artifacts["scaler"].transform(holdout_df[FEATURES].values)
            proba = _artifacts["logit"].predict_proba(X_h_scaled)[:, 1]
            y_h = holdout_df["died"].values
            current_auc = float(roc_auc_score(y_h, proba)) if len(np.unique(y_h)) > 1 else None
        except Exception:
            current_auc = None

    candidate_auc = candidate_metrics["holdout_auc"]
    if current_auc is not None and candidate_auc is not None:
        # Lower AUC = worse. Block if candidate's "badness" (1 - AUC) is more
        # than REGRESSION_TOLERANCE times the current model's badness.
        current_badness = max(1.0 - current_auc, 1e-6)
        candidate_badness = 1.0 - candidate_auc
        if candidate_badness > current_badness * REGRESSION_TOLERANCE:
            promoted = False
            reason = (
                f"Candidate held-out AUC ({candidate_auc:.3f}) is worse than the "
                f"current model's ({current_auc:.3f}) beyond tolerance. Not "
                f"promoted; current model remains live."
            )
        else:
            reason = (
                f"Candidate held-out AUC ({candidate_auc:.3f}) is within tolerance "
                f"of the current model's ({current_auc:.3f}). Promoted."
            )

    result = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "n_samples": int(len(df)),
        "n_deaths": int(df["died"].sum()),
        "promoted": promoted,
        "reason": reason,
        "candidate_metrics": candidate_metrics,
        "current_holdout_auc": current_auc,
        "triggered_by": _user_summary(triggered_by),
    }

    if not promoted:
        await log_model_event(db, {
            "event_type": "patient_retrain_blocked",
            "model_type": "patient_risk",
            **result,
        })
        return result

    # Refit on ALL available data for the model that actually gets shipped.
    X_full = df[FEATURES].values
    y_full = df["died"].values
    final_scaler = StandardScaler().fit(X_full)
    final_logit = LogisticRegression(max_iter=2000, class_weight="balanced")
    final_logit.fit(final_scaler.transform(X_full), y_full)

    config = {
        "features": FEATURES,
        "trained_at": result["trained_at"],
        "n_samples": int(len(df)),
        "n_deaths": int(df["died"].sum()),
        "holdout_auc": candidate_auc,
    }

    archived_to = archive_current_models(base_dir)
    result["archived_previous_version"] = archived_to

    model_dir = _model_dir(base_dir)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(final_logit, os.path.join(model_dir, "patient_risk_logit.pkl"))
    joblib.dump(final_scaler, os.path.join(model_dir, "patient_risk_scaler.pkl"))
    joblib.dump(config, os.path.join(model_dir, "patient_config.pkl"))

    _artifacts["logit"] = final_logit
    _artifacts["scaler"] = final_scaler
    _artifacts["config"] = config
    _artifacts["error"] = None

    await log_model_event(db, {
        "event_type": "patient_retrain_promoted",
        "model_type": "patient_risk",
        **result,
    })
    return result


# ===========================================================================
# Prediction + explanation
# ===========================================================================
class PredictionError(Exception):
    def __init__(self, user_message: str, technical_detail: str = ""):
        self.user_message = user_message
        self.technical_detail = technical_detail
        super().__init__(user_message)


class InvalidPatientInput(Exception):
    """Raised for bad caller-supplied input (→ HTTP 400), distinct from
    PredictionError (→ HTTP 503, model itself unavailable)."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


def _row_from_input(patient: dict) -> pd.Series:
    row = {"age": float(patient["age"])}
    sex_male = _clean_sex_male(patient.get("sex"))
    if sex_male is None:
        raise InvalidPatientInput("`sex` must be 'M'/'male' or 'F'/'female'.")
    row["sex_male"] = sex_male
    for feat in SYMPTOM_FEATURES:
        row[feat] = 1 if patient.get(feat) else 0
    return pd.Series(row)[FEATURES]


def predict(patient: dict, decision_threshold: float = 0.5, top_n: int = 5) -> dict:
    if not is_loaded():
        raise PredictionError(
            "The patient risk model isn't currently available.",
            technical_detail=str(get_error()),
        )

    row = _row_from_input(patient)
    logit = _artifacts["logit"]
    scaler = _artifacts["scaler"]
    X = row.values.reshape(1, -1)
    X_scaled = scaler.transform(X)
    proba = float(logit.predict_proba(X_scaled)[0, 1])

    if proba >= decision_threshold:
        risk_level = "HIGH"
    elif proba >= decision_threshold / 2:
        risk_level = "ELEVATED"
    else:
        risk_level = "LOW"

    # Genuine per-prediction SHAP explanation (LinearExplainer — exact for
    # logistic regression), same pattern as the forecast model's risk flag.
    config = _artifacts["config"]
    n_train = config.get("n_samples", BACKGROUND_SAMPLE_SIZE)
    try:
        background = np.zeros((1, len(FEATURES)))  # mean-centered by scaler already
        masker = shap.maskers.Independent(background, max_samples=1)
        explainer = shap.LinearExplainer(logit, masker)
        shap_values = explainer.shap_values(X_scaled)[0]
        base_value = float(np.array(explainer.expected_value).flatten()[0])
        logodds = float(np.log(proba / (1 - proba))) if 0 < proba < 1 else base_value
        check_ok = abs((base_value + shap_values.sum()) - logodds) < 1e-4

        contrib = pd.Series(shap_values, index=FEATURES).sort_values(key=abs, ascending=False)
        sentences = []
        for feat, val in contrib.head(top_n).items():
            if abs(val) < 1e-6:
                continue
            name = FRIENDLY_NAMES.get(feat, feat.replace("_", " "))
            verb = "increased" if val > 0 else "decreased"
            magnitude = "strongly" if abs(val) > 0.5 else ("moderately" if abs(val) > 0.15 else "slightly")
            if feat == "age":
                sentences.append(f"Age ({int(row['age'])}) {magnitude} {verb} the estimated risk.")
            else:
                present = "presence" if row[feat] else "absence"
                sentences.append(f"The {present} of {name} {magnitude} {verb} the estimated risk.")

        explanation_ok = check_ok
    except Exception:
        sentences = []
        explanation_ok = False

    return {
        "mortality_risk_probability": round(proba, 3),
        "risk_level": risk_level,
        "decision_threshold_used": decision_threshold,
        "top_contributing_factors": sentences if explanation_ok else [],
        "explanation_available": explanation_ok,
        "model_trained_at": config.get("trained_at"),
        "model_n_samples": config.get("n_samples"),
        "note": (
            "This is a decision-support signal based on patterns in past confirmed "
            "cases, not a diagnosis. It does not replace clinical judgment. Model "
            f"trained on {config.get('n_samples', 'an unknown number of')} confirmed "
            "cases — treat with appropriate caution given this sample size."
        ),
    }
