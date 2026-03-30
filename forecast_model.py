import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import os

# ===============================
# PATH CONFIGURATION
# ===============================

DATA_PATH = r"D:\Infosys DV\Project\Forcaasting\data\hospital_claims_60k_realistic_v2.csv"
OUTPUT_DIR = r"D:\Infosys DV\Project\Forcaasting\data"

MONTHLY_FILE = os.path.join(OUTPUT_DIR, "monthly_revenue_history.csv")
FORECAST_FILE = os.path.join(OUTPUT_DIR, "revenue_forecast.csv")

# ===============================
# LOAD DATASET
# ===============================

print("Loading hospital claims dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset Loaded")
print("Total Records:", len(df))

# ===============================
# DATA CLEANING
# ===============================

print("Cleaning data...")

df["Settlement_Date"] = pd.to_datetime(df["Settlement_Date"], errors="coerce")

df["Payment_Received"] = pd.to_numeric(df["Payment_Received"], errors="coerce")

df = df.dropna(subset=["Settlement_Date"])

df["Payment_Received"] = df["Payment_Received"].fillna(0)

# ===============================
# CREATE MONTHLY REVENUE
# ===============================

print("Creating monthly revenue aggregation...")

df["Month"] = df["Settlement_Date"].dt.to_period("M")

monthly_revenue = (
    df.groupby("Month")["Payment_Received"]
    .sum()
    .reset_index()
)

monthly_revenue["Month"] = monthly_revenue["Month"].dt.to_timestamp()

monthly_revenue = monthly_revenue.sort_values("Month")

# Remove incomplete months (very low revenue)
threshold = monthly_revenue["Payment_Received"].mean() * 0.3

monthly_revenue = monthly_revenue[
    monthly_revenue["Payment_Received"] > threshold
]
monthly_revenue = monthly_revenue[monthly_revenue["Payment_Received"] > 30000000]
# Save monthly dataset
monthly_revenue.rename(columns={"Payment_Received":"Actual_Revenue"}, inplace=True)

monthly_revenue.to_csv(MONTHLY_FILE, index=False)

print("Monthly revenue saved:", MONTHLY_FILE)

# ===============================
# PREPARE TIME SERIES
# ===============================

ts = monthly_revenue.set_index("Month")

# Log transform for stability
ts["log_revenue"] = np.log(ts["Actual_Revenue"])

# ===============================
# TRAIN FORECAST MODEL
# ===============================

print("Training ARIMA forecasting model...")

model = ARIMA(ts["log_revenue"], order=(1,1,1))

model_fit = model.fit()

print("Model training completed")

# ===============================
# FORECAST NEXT 6 MONTHS
# ===============================

forecast_steps = 6

log_forecast = model_fit.forecast(steps=forecast_steps)

# Convert back from log
forecast_values = np.exp(log_forecast)

last_date = ts.index[-1]

forecast_dates = pd.date_range(
    start=last_date + pd.DateOffset(months=1),
    periods=forecast_steps,
    freq="MS"
)

forecast_df = pd.DataFrame({
    "Month": forecast_dates,
    "Forecast_Revenue": forecast_values
})

# Prevent negative values
forecast_df["Forecast_Revenue"] = forecast_df["Forecast_Revenue"].clip(lower=0)

forecast_df.to_csv(FORECAST_FILE, index=False)

print("Forecast saved:", FORECAST_FILE)

# ===============================
# DISPLAY RESULTS
# ===============================

print("\nForecast Results\n")
print(forecast_df)

print("\nRevenue Forecasting Completed Successfully")