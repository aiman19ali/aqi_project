                              

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os





data_file = 'data/cleaned_data.csv'
if not os.path.exists(data_file):
    raise FileNotFoundError(f"{data_file} not found!")
data = pd.read_csv(data_file)
print(f"Loaded data with shape: {data.shape}")






features = [
    'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3',
    'temperature', 'humidity', 'pressure', 'wind_speed',
    'day_of_week'
]


for lag in [1, 2, 3]:
    data[f'aqi_lag_{lag}'] = data['aqi_7day_avg'].shift(lag)
    features.append(f'aqi_lag_{lag}')
data = data.bfill()


numeric_cols = data.select_dtypes(include=np.number).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())


data['aqi_24h'] = data['aqi_index'].shift(-24)
data['aqi_48h'] = data['aqi_index'].shift(-48)
data['aqi_72h'] = data['aqi_index'].shift(-72)

data.dropna(subset=['aqi_24h','aqi_48h','aqi_72h'], inplace=True)

targets = {
    '24h': 'aqi_24h',
    '48h': 'aqi_48h',
    '72h': 'aqi_72h'
}





models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)

for horizon, target in targets.items():
    print(f"\nTraining model for {horizon} prediction...")

    if target not in data.columns:
        print(f"WARNING Target column '{target}' not found in data. Skipping.")
        continue
    X = data[features]
    y = data[target]


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )


    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"{horizon} model trained:")
    print(f"   MAE  = {mae:.3f}")
    print(f"   RMSE = {rmse:.3f}")
    print(f"   R^2  = {r2:.3f}")


    model_file = os.path.join(models_dir, f"xgb_aqi_{horizon}.joblib")
    joblib.dump(model, model_file)
    print(f"Saved model: {model_file}")
print("\nAll improved models trained and saved successfully!")
