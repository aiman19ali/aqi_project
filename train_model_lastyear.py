                         

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os
from datetime import datetime, timedelta





data_file = "data/cleaned_data.csv"
if not os.path.exists(data_file):
    raise FileNotFoundError(f"{data_file} not found!")
data = pd.read_csv(data_file)
print(f"Loaded data with shape: {data.shape}")




if "date" in data.columns:
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    one_year_ago = data["date"].max() - pd.Timedelta(days=365)
    data = data[data["date"] >= one_year_ago]
    print(f"Using data from {data['date'].min().date()} to {data['date'].max().date()}")
features = [
    "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
    "temperature", "humidity", "pressure", "wind_speed"
]


if "date" in data.columns:
    data["day_of_week"] = data["date"].dt.weekday
    data["month"] = data["date"].dt.month
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    features += ["day_of_week", "month", "is_weekend"]
data["temp_humidity_interaction"] = data["temperature"] * data["humidity"]
data["pressure_wind_interaction"] = data["pressure"] * data["wind_speed"]
data["pm25_pm10_ratio"] = data["pm2_5"] / (data["pm10"] + 1e-6)
data["co_no2_ratio"] = data["co"] / (data["no2"] + 1e-6)

features += [
    "temp_humidity_interaction",
    "pressure_wind_interaction",
    "pm25_pm10_ratio",
    "co_no2_ratio"
]


for lag in [1, 2, 7]:
    data[f"aqi_lag_{lag}"] = data["aqi_index"].shift(lag)
    features.append(f"aqi_lag_{lag}")
data["aqi_3day_avg"] = data["aqi_index"].rolling(3).mean()
data["aqi_7day_avg"] = data["aqi_index"].rolling(7).mean()
data["aqi_30day_avg"] = data["aqi_index"].rolling(30).mean()
features += ["aqi_3day_avg", "aqi_7day_avg", "aqi_30day_avg"]


data = data.bfill().ffill()
numeric_cols = data.select_dtypes(include=np.number).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())




data["aqi_24h"] = data["aqi_index"].shift(-24)
data["aqi_48h"] = data["aqi_index"].shift(-48)
data["aqi_72h"] = data["aqi_index"].shift(-72)
data.dropna(subset=["aqi_24h", "aqi_48h", "aqi_72h"], inplace=True)

targets = {
    "24h": "aqi_24h",
    "48h": "aqi_48h",
    "72h": "aqi_72h",
}




models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

for horizon, target in targets.items():
    print(f"\nTraining model for {horizon} prediction...")

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=700,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.2,
        reg_alpha=0.6,
        random_state=42,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"{horizon} model trained: MAE={mae:.2f}, RMSE={rmse:.2f}")

    model_file = os.path.join(models_dir, f"model_{horizon}_1year.pkl")
    joblib.dump(model, model_file)
    print(f"Saved model to {model_file}")
print("\nAll 1-year AQI models trained and saved successfully!")
