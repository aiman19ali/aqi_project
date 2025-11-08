from fastapi import FastAPI
from pydantic import BaseModel
import requests
from datetime import datetime
import pandas as pd
import os
import numpy as np
import joblib                            


app = FastAPI(title="AQI Prediction Backend")


MODEL_PATH = os.path.join("models", "model_72h_1year.pkl")                         
AQI_SCALE_FACTOR = 50.0


model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("AQI model loaded successfully!")
    except Exception as e:
        print(f"WARNING Failed to load model: {e}")
else:
    print("WARNING Model not found. Using dummy fallback.")
FEATURES = [
    'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3',
    'temperature', 'humidity', 'pressure', 'wind_speed',
    'day_of_week', 'month', 'is_weekend',
    'temp_humidity_interaction', 'pressure_wind_interaction',
    'pm25_pm10_ratio', 'co_no2_ratio',
    'aqi_lag_1', 'aqi_lag_2', 'aqi_lag_7',
    'aqi_3day_avg', 'aqi_7day_avg', 'aqi_30day_avg'
]


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
@app.get("/")
def root():
    return {"message": "FastAPI AQI backend is running successfully!"}
@app.post("/predict")
def predict(data: AQIInput):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    df_input = pd.DataFrame([data.dict()])


    df_input["day_of_week"] = datetime.now().weekday()
    df_input["month"] = datetime.now().month
    df_input["is_weekend"] = int(df_input["day_of_week"] >= 5)

    df_input["temp_humidity_interaction"] = df_input["temperature"] * df_input["humidity"]
    df_input["pressure_wind_interaction"] = df_input["pressure"] * df_input["wind_speed"]
    df_input["pm25_pm10_ratio"] = df_input["pm2_5"] / (df_input["pm10"] + 1e-6)
    df_input["co_no2_ratio"] = df_input["co"] / (df_input["no2"] + 1e-6)


    df_input["aqi_lag_1"] = 80
    df_input["aqi_lag_2"] = 78
    df_input["aqi_lag_7"] = 75
    df_input["aqi_3day_avg"] = 77
    df_input["aqi_7day_avg"] = 74
    df_input["aqi_30day_avg"] = 72


    for col in FEATURES:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[FEATURES]


    if model:
        predicted_aqi = float(model.predict(df_input)[0]) * AQI_SCALE_FACTOR


        predicted_aqi = max(0.0, min(predicted_aqi, 500.0))
        note = "Prediction made using trained AQI model."
    else:
        predicted_aqi = (df_input["pm2_5"] + df_input["pm10"] +
                         df_input["no2"] + df_input["so2"]) / 4
        note = "WARNING Dummy formula used (no trained model)."
    return {
        "predicted_aqi": round(float(predicted_aqi), 1),
        "timestamp": timestamp,
        "status": "success",
        "note": note
    }
@app.get("/live_aqi")
def live_aqi():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    token = "94c63d68dec20d25e09d7fade01bc26a5c731524"
    try:
        url = f"https://api.waqi.info/feed/karachi/?token={token}"
        response = requests.get(url, timeout=10)
        data = response.json()
        live_aqi_value = data["data"]["aqi"]
        note = "Live AQI fetched successfully from AQICN."
    except Exception as e:
        live_aqi_value = np.random.randint(50, 200)
        note = f"WARNING Using dummy live AQI due to error: {e}"
    return {
        "live_aqi": live_aqi_value,
        "timestamp": timestamp,
        "status": "success" if live_aqi_value is not None else "error",
        "note": note
    }
@app.get("/get_forecast")
def get_forecast():
    try:

        df = pd.read_csv("data/clean_aqi.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        latest = df.iloc[-1]                         

        forecasts = []
        for i in range(1, 4):               
            future_date = latest["date"] + pd.Timedelta(days=i)

            sample = latest.copy()
            sample["date"] = future_date


            sample["day_of_week"] = future_date.weekday()
            sample["month"] = future_date.month
            sample["is_weekend"] = int(sample["day_of_week"] >= 5)

            sample["temp_humidity_interaction"] = sample["temperature"] * sample["humidity"]
            sample["pressure_wind_interaction"] = sample["pressure"] * sample["wind_speed"]
            sample["pm25_pm10_ratio"] = sample["pm2_5"] / (sample["pm10"] + 1e-6)
            sample["co_no2_ratio"] = sample["co"] / (sample["no2"] + 1e-6)


            X = sample[FEATURES].values.reshape(1, -1)

            pred = float(model.predict(X)[0]) * AQI_SCALE_FACTOR
            pred = max(0.0, min(pred, 500.0))

            forecasts.append({
                "date": future_date.strftime("%Y-%m-%d"),
                "predicted_aqi": round(float(pred), 2)
            })
        return {"status": "success", "forecasts": forecasts}
    except Exception as e:
        return {"status": "error", "message": str(e)}
if __name__ == "__main__":
    import uvicorn
    print("Starting Starting FastAPI AQI server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
