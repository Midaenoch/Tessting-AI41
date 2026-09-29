"""
app/services/outbreak_service.py

Declares whether a given month's case count constitutes an outbreak.

DELIBERATELY NOT a trained ML classifier. There is no ground-truth
"declared_outbreak" label anywhere in the source data (no public health
authority's retrospective outbreak declarations are recorded) — so a
supervised model would have nothing real to learn from; it would just be
an arbitrary threshold wearing an ML costume, harder to audit than the
threshold itself.

Instead this uses the standard epidemiological surveillance approach (the
same family as CDC's EARS and WHO's epidemic-threshold methods): compare
the current month's case count to a HISTORICAL BASELINE FOR THAT SAME
CALENDAR MONTH (mean + k * standard deviation). This matters because Lassa
fever is strongly seasonal in this data — January/February average
~440-500 cases/month, May-September average ~170-210 — so a single
flat, year-round threshold would falsely declare an outbreak most
Januaries and could miss a genuine outbreak in June. Every number this
module produces is a transparent, reproducible statistic computed
directly from the historical data — not a black box.
"""
from typing import Optional

import numpy as np
import pandas as pd

from app.services import forecast_service

DEFAULT_K = 1.5  # standard "elevated" epidemic threshold multiplier (mean + k*SD)
MIN_YEARS_FOR_BASELINE = 3  # need at least this many observations of a calendar month to trust its baseline


def _monthly_df() -> pd.DataFrame:
    df = forecast_service.get_monthly_df()
    if df is None:
        raise RuntimeError("Monthly case data isn't loaded yet.")
    return df


def compute_baselines(k: float = DEFAULT_K) -> pd.DataFrame:
    """Per-calendar-month (1-12) historical mean, standard deviation, and
    the resulting outbreak threshold, computed from every year of data
    currently loaded. This is the full, auditable reference table —
    exposed directly via GET /api/v1/outbreaks/baseline.
    """
    df = _monthly_df()
    stats = df.groupby("calendar_month")["case_count"].agg(
        baseline_mean="mean", baseline_std="std", n_years="count",
    )
    stats["threshold"] = stats["baseline_mean"] + k * stats["baseline_std"].fillna(0)
    stats["k"] = k
    stats["reliable"] = stats["n_years"] >= MIN_YEARS_FOR_BASELINE
    return stats.reset_index().sort_values("calendar_month")


def declare(case_count: float, calendar_month: int, k: float = DEFAULT_K) -> dict:
    """Declares whether `case_count` in a given calendar month constitutes
    an outbreak, against that month's own historical baseline."""
    baselines = compute_baselines(k=k)
    row = baselines[baselines["calendar_month"] == calendar_month]
    if row.empty:
        raise RuntimeError(f"No baseline data for calendar month {calendar_month}")
    row = row.iloc[0]

    threshold = float(row["threshold"])
    declared = bool(case_count >= threshold)
    z_score = (
        float((case_count - row["baseline_mean"]) / row["baseline_std"])
        if row["baseline_std"] and row["baseline_std"] > 0
        else None
    )

    return {
        "declared_outbreak": declared,
        "case_count": float(case_count),
        "calendar_month": int(calendar_month),
        "baseline_mean": round(float(row["baseline_mean"]), 1),
        "baseline_std": round(float(row["baseline_std"]), 1) if pd.notna(row["baseline_std"]) else None,
        "threshold": round(threshold, 1),
        "z_score": round(z_score, 2) if z_score is not None else None,
        "k": k,
        "baseline_reliable": bool(row["reliable"]),
        "n_years_in_baseline": int(row["n_years"]),
        "method": (
            f"Statistical surveillance threshold: outbreak declared if case count "
            f"≥ historical mean + {k}×SD for this calendar month, computed from "
            f"{int(row['n_years'])} years of data for this month."
        ),
    }


def declare_latest(k: float = DEFAULT_K) -> dict:
    """Declares an outbreak status for the most recent month in the dataset."""
    df = _monthly_df()
    latest = df.iloc[-1]
    result = declare(float(latest["case_count"]), int(latest["calendar_month"]), k=k)
    result["month"] = str(latest.get("month", latest.get("month_ts", "")))
    result["year"] = int(latest["year"])
    return result


def declaration_history(k: float = DEFAULT_K) -> list[dict]:
    """Applies the SAME rule retroactively across every month in the
    dataset. This is the honest replacement for the old
    'total_outbreaks = any month with >=1 confirmed case' placeholder —
    see app/services/dashboard_service.get_kpi().
    """
    df = _monthly_df()
    baselines = compute_baselines(k=k).set_index("calendar_month")

    out = []
    for _, row in df.iterrows():
        b = baselines.loc[int(row["calendar_month"])]
        threshold = float(b["threshold"])
        out.append({
            "year": int(row["year"]),
            "calendar_month": int(row["calendar_month"]),
            "month": str(row.get("month", row.get("month_ts", ""))),
            "case_count": float(row["case_count"]),
            "threshold": round(threshold, 1),
            "declared_outbreak": bool(row["case_count"] >= threshold),
        })
    return out


def count_declared_outbreak_months(k: float = DEFAULT_K) -> int:
    """The single number that should replace the old placeholder KPI."""
    history = declaration_history(k=k)
    return sum(1 for h in history if h["declared_outbreak"])
