# predict_aqi.py
import pandas as pd
import joblib
import os
from datetime import datetime, timedelta

# -----------------------------
# 1. Paths
# -----------------------------
data_file = "data/cleaned_data.csv"
models_dir = "models"
output_file = "data/predicted_aqi_next3days.csv"

if not os.path.exists(data_file):
    raise FileNotFoundError(f"❌ Data file not found: {data_file}")

# -----------------------------
# 2. Load historical data
# -----------------------------
df = pd.read_csv(data_file)
if "date" not in df.columns:
    raise ValueError("❌ 'date' column missing in data!")

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.sort_values("date").reset_index(drop=True)

# -----------------------------
# 3. Start prediction from today
# -----------------------------
today = datetime.today().date()
print(f"📆 Predicting AQI from today: {today}")

# Latest row from historical data for lag/rolling features
latest_row = df.iloc[-1:].copy()

# -----------------------------
# 4. Feature Engineering
# -----------------------------
latest_row["day_of_week"] = latest_row["date"].dt.weekday
latest_row["month"] = latest_row["date"].dt.month
latest_row["is_weekend"] = (latest_row["day_of_week"] >= 5).astype(int)

# Interaction features
latest_row["temp_humidity_interaction"] = latest_row["temperature"] * latest_row["humidity"]
latest_row["pressure_wind_interaction"] = latest_row["pressure"] * latest_row["wind_speed"]
latest_row["pm25_pm10_ratio"] = latest_row["pm2_5"] / (latest_row["pm10"] + 1e-6)
latest_row["co_no2_ratio"] = latest_row["co"] / (latest_row["no2"] + 1e-6)

# Lag features
latest_row["aqi_lag_1"] = df["aqi_index"].iloc[-1]
latest_row["aqi_lag_2"] = df["aqi_index"].iloc[-2]
latest_row["aqi_lag_7"] = df["aqi_index"].iloc[-7]

# Rolling averages
latest_row["aqi_3day_avg"] = df["aqi_index"].iloc[-3:].mean()
latest_row["aqi_7day_avg"] = df["aqi_index"].iloc[-7:].mean()
latest_row["aqi_30day_avg"] = df["aqi_index"].iloc[-30:].mean()

# Drop unused columns
drop_cols = ["date", "aqi_index", "aqi_24h", "aqi_48h", "aqi_72h"]
latest_row = latest_row.drop(columns=[c for c in drop_cols if c in latest_row.columns])

# Ensure feature order exactly matches training
features_order = [
    "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
    "temperature", "humidity", "pressure", "wind_speed",
    "day_of_week", "month", "is_weekend",
    "temp_humidity_interaction", "pressure_wind_interaction",
    "pm25_pm10_ratio", "co_no2_ratio",
    "aqi_lag_1", "aqi_lag_2", "aqi_lag_7",
    "aqi_3day_avg", "aqi_7day_avg", "aqi_30day_avg"
]
latest_row = latest_row[features_order]

# -----------------------------
# 5. Load models and predict next 3 days
# -----------------------------
models = {
    1: "model_24h_1year.pkl",
    2: "model_48h_1year.pkl",
    3: "model_72h_1year.pkl"
}

# SCALE FACTOR → adjust according to how model was trained
# Example: if training target was normalized to 0-10, multiply by 50 to get real AQI
SCALE_FACTOR = 50

predictions = []

for day, model_file in models.items():
    model_path = os.path.join(models_dir, model_file)
    if not os.path.exists(model_path):
        print(f"⚠️ Model not found: {model_path}")
        continue

    model = joblib.load(model_path)
    pred_scaled = model.predict(latest_row)[0]

    # Convert to real AQI scale
    pred_real = round(float(pred_scaled) * SCALE_FACTOR)

    # Use tomorrow, +2, +3 relative to "today"
    date_pred = today + timedelta(days=day)
    predictions.append({
        "Date": date_pred.strftime("%Y-%m-%d"),
        "Predicted_AQI": pred_real
    })

# -----------------------------
# 6. Display predictions
# -----------------------------
print("\n🌤️ Predicted AQI for Next 3 Days (Real AQI Scale):\n")
for p in predictions:
    print(f"   {p['Date']}: {p['Predicted_AQI']}")

# -----------------------------
# 7. Save results to CSV
# -----------------------------
pd.DataFrame(predictions).to_csv(output_file, index=False)
print(f"\n💾 Predictions saved to: {output_file}")
