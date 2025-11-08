                
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict

app = FastAPI(title="AQI Prediction API")


_MODEL_CANDIDATES = [
    "models/model_24h_1year.pkl",
    "models/model_24h.pkl",
    "models/aqi_model.pkl",
    "models/xgb_aqi_24h.joblib",
]

def _try_load_model():
    for path in _MODEL_CANDIDATES:
        try:
            return joblib.load(path)
        except Exception:
            continue
    return None
model = _try_load_model()


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
def _compute_aqi_from_payload(data: AQIInput) -> float:
    """Compute AQI using EPA breakpoints from the request payload."""
    PM25 = [
        (0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    PM10 = [
        (0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150), (255, 354, 151, 200),
        (355, 424, 201, 300), (425, 504, 301, 400), (505, 604, 401, 500),
    ]
    O3 = [
        (0.0, 54.0, 0, 50), (54.1, 70.0, 51, 100), (70.1, 85.0, 101, 150),
        (85.1, 105.0, 151, 200), (105.1, 200.0, 201, 300),
    ]
    NO2 = [
        (0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150), (361, 649, 151, 200),
        (650, 1249, 201, 300),
    ]
    SO2 = [
        (0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150), (186, 304, 151, 200),
        (305, 604, 201, 300),
    ]

    def calc(val: float, bps):
        for c_low, c_high, aqi_low, aqi_high in bps:
            if c_low <= val <= c_high:
                span = (c_high - c_low) or 1.0
                return aqi_low + (val - c_low) / span * (aqi_high - aqi_low)
        return float(bps[-1][3])
    vals = [
        calc(float(data.pm2_5), PM25),
        calc(float(data.pm10), PM10),
        calc(float(data.o3), O3),
        calc(float(data.no2), NO2),
        calc(float(data.so2), SO2),
    ]
    return float(np.nanmax(vals))
@app.post("/predict")
def predict(data: AQIInput):

    if model is not None:
        try:
            features = np.array([[data.co, data.no, data.no2, data.o3, data.so2,
                                  data.pm2_5, data.pm10, data.nh3, data.temperature,
                                  data.humidity, data.pressure, data.wind_speed]])
            prediction = model.predict(features)[0]
            return {"predicted_aqi": float(prediction)}
        except Exception:
            pass
    computed = _compute_aqi_from_payload(data)
    return {"predicted_aqi": computed}
def _read_forecast_csvs() -> List[Dict]:
    """
    Load next-3-day forecast from both known sources and combine:
      - data/forecast_daily_next3.csv (from api_project.py)
      - data/predicted_aqi_next3days.csv (from predict_aqi.py)
    Returns list of dicts: {"date": str, "predicted_aqi": float}
    """
    frames: List[pd.DataFrame] = []


    try:
        a = pd.read_csv("data/forecast_daily_next3.csv")
        if "date" in a.columns:
            if "predicted_aqi" in a.columns:
                a = a[["date", "predicted_aqi"]].dropna()
            elif "aqi_index" in a.columns:
                a = a[["date", "aqi_index"]].dropna().rename(columns={"aqi_index": "predicted_aqi"})
            else:
                a = None
            if a is not None:
                a["predicted_aqi"] = a["predicted_aqi"].astype(float)
                frames.append(a)
    except Exception:
        pass
    try:
        b = pd.read_csv("data/predicted_aqi_next3days.csv")
        if {"Date", "Predicted_AQI"}.issubset(b.columns):
            b = b.rename(columns={"Date": "date", "Predicted_AQI": "predicted_aqi"})
            b["predicted_aqi"] = b["predicted_aqi"].astype(float)
            frames.append(b[["date", "predicted_aqi"]])
    except Exception:
        pass
    if not frames:
        return []
    merged = pd.concat(frames, ignore_index=True)

    merged = merged.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="first")
    return merged.to_dict(orient="records")
def _select_next3(records: List[Dict]) -> List[Dict]:
    if not records:
        return []
    df = pd.DataFrame(records)
    if "date" not in df.columns:
        return records
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    from datetime import datetime
    today = datetime.today().date()
    df = df[df["date"] > today].sort_values("date").head(3)
    return df.to_dict(orient="records")
@app.get("/get_forecast")
def get_forecast():
    """HTTP endpoint to return next-3-day forecast for the dashboard."""
    records = _read_forecast_csvs()
    if not records:
        raise HTTPException(status_code=404, detail="No forecast data available. Run data collection/prediction first.")
    return {"status": "success", "forecasts": _select_next3(records)}
@app.get("/")
def root():
    return {"status": "ok"}
