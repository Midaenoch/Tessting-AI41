"""
train_patient_model.py

One-time (and re-runnable) script that trains the INITIAL patient outcome
risk model directly from the raw line-list spreadsheet, the same way the
original forecast model's notebooks produced models/final_*.pkl.

After this runs once, app/services/patient_service.retrain() takes over for
subsequent retrains — pulling from MongoDB's db.cases instead of this file,
once uploads include the symptom fields (see app/routers/admin.py).

Run: python3 train_patient_model.py
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from app.services.patient_service import (  # noqa: E402
    FEATURES, HOLDOUT_YEARS, build_features_from_excel_df,
)


def main():
    excel_path = os.path.join(BASE_DIR, "data", "raw", "2015-2025__3_.xlsx")
    print(f"Loading {excel_path} ...")
    df = pd.read_excel(excel_path)

    features_df = build_features_from_excel_df(df)
    print(f"\nConfirmed cases with known outcome and usable age/sex: {len(features_df)}")
    print(f"Deaths: {features_df['died'].sum()} ({features_df['died'].mean():.1%})")
    print(f"Year range: {features_df['year'].min()}–{features_df['year'].max()}")

    max_year = int(features_df["year"].max())
    holdout_mask = features_df["year"] > (max_year - HOLDOUT_YEARS)
    train_df, holdout_df = features_df[~holdout_mask], features_df[holdout_mask]
    print(f"\nTime-aware split: train on <= {max_year - HOLDOUT_YEARS} "
          f"({len(train_df)} cases), hold out {max_year - HOLDOUT_YEARS + 1}-"
          f"{max_year} ({len(holdout_df)} cases)")

    X_train, y_train = train_df[FEATURES].values, train_df["died"].values
    X_holdout, y_holdout = holdout_df[FEATURES].values, holdout_df["died"].values

    scaler = StandardScaler().fit(X_train)
    logit = LogisticRegression(max_iter=2000, class_weight="balanced")
    logit.fit(scaler.transform(X_train), y_train)

    proba = logit.predict_proba(scaler.transform(X_holdout))[:, 1]
    pred = (proba >= 0.5).astype(int)

    print("\n=== Held-out evaluation (years the model never trained on) ===")
    print(f"AUC:        {roc_auc_score(y_holdout, proba):.3f}")
    print(f"Accuracy:   {accuracy_score(y_holdout, pred):.3f}")
    print(f"Precision (death class): {precision_score(y_holdout, pred, zero_division=0):.3f}")
    print(f"Recall (death class):    {recall_score(y_holdout, pred, zero_division=0):.3f}")
    print("\nConfusion matrix [[TN, FP], [FN, TP]] (positive = died):")
    print(confusion_matrix(y_holdout, pred))
    print("\n" + classification_report(y_holdout, pred, target_names=["alive", "dead"], zero_division=0))

    # Refit on ALL data for the shipped model (same pattern as the forecast model).
    X_full, y_full = features_df[FEATURES].values, features_df["died"].values
    final_scaler = StandardScaler().fit(X_full)
    final_logit = LogisticRegression(max_iter=2000, class_weight="balanced")
    final_logit.fit(final_scaler.transform(X_full), y_full)

    top_coef = pd.Series(final_logit.coef_[0], index=FEATURES).sort_values(key=abs, ascending=False)
    print("\n=== Top 10 features by |coefficient| (full-data fit) ===")
    print(top_coef.head(10))

    config = {
        "features": FEATURES,
        "trained_at": pd.Timestamp.now("UTC").isoformat(),
        "n_samples": int(len(features_df)),
        "n_deaths": int(features_df["died"].sum()),
        "holdout_auc": float(roc_auc_score(y_holdout, proba)),
    }

    model_dir = os.path.join(BASE_DIR, "models", "patient")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(final_logit, os.path.join(model_dir, "patient_risk_logit.pkl"))
    joblib.dump(final_scaler, os.path.join(model_dir, "patient_risk_scaler.pkl"))
    joblib.dump(config, os.path.join(model_dir, "patient_config.pkl"))
    print(f"\nSaved to {model_dir}/")


if __name__ == "__main__":
    main()
