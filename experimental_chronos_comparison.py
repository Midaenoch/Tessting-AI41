"""
AI4Lassa — Experimental: zero-shot forecast using Amazon's Chronos
(a pretrained time-series foundation model from Hugging Face)

⚠️ UNTESTED — this sandbox can't reach huggingface.co (network-restricted)
and doesn't have enough disk space for the torch/CUDA dependency chain.
Run this on your own machine, where both of those are non-issues.

Install first:
    pip install chronos-forecasting torch

This compares Chronos's zero-shot forecast against our tuned Random Forest's
actual test-set performance (MAE 49.3, R² 0.567), so you can see directly
whether a pretrained foundation model beats our custom-trained one here.
"""
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline
from sklearn.metrics import mean_absolute_error, r2_score

DATA_PATH = "data/processed/monthly_features.csv"

# --- Load the same data and the same test period used throughout this project ---
df = pd.read_csv(DATA_PATH)
df["month_ts"] = pd.to_datetime(df["month_ts"])
df = df.sort_values("month_ts").reset_index(drop=True)

test = df[(df.month_ts >= "2024-01-01")].reset_index(drop=True)

# --- Load the pretrained model (downloads from Hugging Face on first run) ---
# "small" balances speed and quality; "tiny"/"base"/"large" are also available.
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cpu",  # use "cuda" if you have a GPU
    torch_dtype=torch.float32,
)

# --- Rolling 1-step-ahead forecast through the test period, same protocol as
#     the SARIMA comparison in notebook 03, for a fair apples-to-apples comparison ---
history = df[df.month_ts < "2024-01-01"]["case_count"].tolist()
preds = []

for i in range(len(test)):
    context = torch.tensor(history, dtype=torch.float32)
    # Chronos returns multiple sampled trajectories; take the median as the point forecast
    forecast = pipeline.predict(context=context, prediction_length=1, num_samples=100)
    median_forecast = float(np.median(forecast[0].numpy()))
    preds.append(median_forecast)
    # Append the actual observed next value (rolling forecast, same as SARIMA test)
    history.append(test["target_next_month_cases"].iloc[i])

preds = np.array(preds)
actual = test["target_next_month_cases"].values

mae = mean_absolute_error(actual, preds)
r2 = r2_score(actual, preds)

print("=== Chronos zero-shot forecast vs. actual (test set, 2024-2025) ===")
print(pd.DataFrame({
    "month": test["month"], "actual": actual.astype(int), "chronos_pred": preds.round(0).astype(int)
}).to_string(index=False))
print()
print(f"Chronos zero-shot: MAE={mae:.1f}  R2={r2:.3f}")
print(f"Our tuned Random Forest (already measured, for comparison): MAE=49.3  R2=0.567")
print()
if mae < 49.3:
    print("Chronos beat our Random Forest on this test set.")
else:
    print("Chronos did not beat our Random Forest on this test set — "
          "plausible given the very short series (only ~72-118 months) "
          "and Chronos not having access to our engineered lag/seasonality/"
          "positivity-rate features, only the raw case-count sequence.")
