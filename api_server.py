# fastapi_app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="AQI Prediction API")

# Load your trained model
model = joblib.load("models/model_24h.pkl")

# Define the data format we’ll accept
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

@app.post("/predict")
def predict(data: AQIInput):
    features = np.array([[data.co, data.no, data.no2, data.o3, data.so2,
                          data.pm2_5, data.pm10, data.nh3, data.temperature,
                          data.humidity, data.pressure, data.wind_speed]])
    
    prediction = model.predict(features)[0]
    return {"predicted_aqi": float(prediction)}
