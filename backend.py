from fastapi import FastAPI
from pydantic import BaseModel
import requests
from datetime import datetime
import pandas as pd
import os
import numpy as np
import joblib  # safer for XGBoost models

# ---------------- APP SETUP ----------------
app = FastAPI(title="AQI Prediction Backend")

# ---------------- PATHS ----------------
MODEL_PATH = os.path.join("models", "model_72h.pkl")  # Make sure this exists

# ---------------- LOAD MODEL ----------------
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ AQI model loaded successfully!")
    except Exception as e:
        print(f"⚠️ Failed to load model: {e}")
else:
    print("⚠️ Model not found. Using dummy fallback.")

# ---------------- FEATURES ----------------
FEATURES = [
    "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
    "temperature", "humidity", "pressure", "wind_speed",
    "day_of_week", "month", "season", "is_weekend",
    "aqi_lag_1", "aqi_lag_2", "aqi_lag_7",
    "aqi_3day_avg", "aqi_7day_avg", "aqi_30day_avg",
    "temp_humidity_interaction", "pressure_wind_interaction",
    "pm25_pm10_ratio", "co_no2_ratio"
]

# ---------------- INPUT SCHEMA ----------------
class AQIInput(BaseModel):
    co: float
    no: float
    no2: float
    o3: float
    so2: float
    pm2_5: float
    pm10: float
    nh3: float
    temperature: float
    humidity: float
    pressure: float
    wind_speed: float

# ---------------- ROOT ROUTE ----------------
@app.get("/")
def root():
    return {"message": "✅ FastAPI AQI backend is running successfully!"}

# ---------------- PREDICT ROUTE ----------------
@app.post("/predict")
def predict(data: AQIInput):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Convert input to DataFrame
    df_input = pd.DataFrame([data.dict()])

    # Feature engineering
    df_input["day_of_week"] = datetime.now().weekday()
    df_input["month"] = datetime.now().month
    df_input["season"] = ((df_input["month"] % 12 + 3) // 3)
    df_input["is_weekend"] = int(df_input["day_of_week"] >= 5)
    df_input["temp_humidity_interaction"] = df_input["temperature"] * df_input["humidity"]
    df_input["pressure_wind_interaction"] = df_input["pressure"] * df_input["wind_speed"]
    df_input["pm25_pm10_ratio"] = df_input["pm2_5"] / (df_input["pm10"] + 1e-6)
    df_input["co_no2_ratio"] = df_input["co"] / (df_input["no2"] + 1e-6)

    # Fill missing columns if needed
    for col in FEATURES:
        if col not in df_input.columns:
            df_input[col] = 0  # fallback

    # Prediction
    if model:
        predicted_aqi = model.predict(df_input[FEATURES])[0]
        note = "✅ Prediction made using trained AQI model."
    else:
        # Dummy fallback
        predicted_aqi = (df_input["pm2_5"] + df_input["pm10"] + df_input["no2"] + df_input["so2"]) / 4
        note = "⚠️ Dummy formula used (no trained model)."

    return {
        "predicted_aqi": round(float(predicted_aqi), 1),
        "timestamp": timestamp,
        "status": "success",
        "note": note
    }

# ---------------- LIVE AQI ROUTE ----------------
@app.get("/live_aqi")
def live_aqi():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    token = "94c63d68dec20d25e09d7fade01bc26a5c731524"
    try:
        url = f"https://api.waqi.info/feed/karachi/?token={token}"
        response = requests.get(url, timeout=10)
        data = response.json()
        live_aqi_value = data["data"]["aqi"]
        note = "✅ Live AQI fetched successfully from AQICN."
    except Exception as e:
        live_aqi_value = np.random.randint(50, 200)
        note = f"⚠️ Using dummy live AQI due to error: {e}"

    return {
        "live_aqi": live_aqi_value,
        "timestamp": timestamp,
        "status": "success" if live_aqi_value is not None else "error",
        "note": note
    }

# ---------------- RUN LOCALLY ----------------
# uvicorn backend:app --reload
