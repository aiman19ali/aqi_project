                        

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def make_dir(path: str):
    """Create the directory if it doesn't already exist."""
    if not os.path.exists(path):
        os.makedirs(path)
def load_history(csv_path: str) -> pd.DataFrame:
    """
    Load 1-year daily AQI history (CSV) saved from API.
    Cleans up and returns a properly formatted DataFrame.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}. Run API data collection script first.")
    df = pd.read_csv(csv_path)

    if "date" not in df.columns:
        raise ValueError("CSV must have a 'date' column!")
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.date
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df
def save_summary(df: pd.DataFrame, out_dir: str):
    """Generate descriptive stats and correlations."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    summary = df[numeric_cols].describe().T
    corr = df[numeric_cols].corr()

    summary.to_csv(os.path.join(out_dir, "summary_stats.csv"))
    corr.to_csv(os.path.join(out_dir, "correlations.csv"))
def create_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create enhanced features for ML while keeping original inputs intact."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])


    cols_to_fill = [
        "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
        "temperature", "humidity", "pressure", "wind_speed"
    ]
    df[cols_to_fill] = df[cols_to_fill].ffill().bfill()


    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["season"] = ((df["month"] % 12 + 3) // 3)                                        
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)


    df["aqi_lag_1"] = df["aqi_index"].shift(1)
    df["aqi_lag_2"] = df["aqi_index"].shift(2)
    df["aqi_lag_7"] = df["aqi_index"].shift(7)


    df["aqi_3day_avg"] = df["aqi_index"].rolling(window=3, min_periods=1).mean()
    df["aqi_7day_avg"] = df["aqi_index"].rolling(window=7, min_periods=1).mean()
    df["aqi_30day_avg"] = df["aqi_index"].rolling(window=30, min_periods=1).mean()


    df["temp_humidity_interaction"] = df["temperature"] * df["humidity"]
    df["pressure_wind_interaction"] = df["pressure"] * df["wind_speed"]


    df["pm25_pm10_ratio"] = df["pm2_5"] / (df["pm10"] + 1e-6)
    df["co_no2_ratio"] = df["co"] / (df["no2"] + 1e-6)


    df = df.ffill().bfill()

    return df
def save_enhanced_data(df: pd.DataFrame, output_path: str):
    """Save enhanced dataset for ML training."""
    df.to_csv(output_path, index=False)
    print(f"Enhanced dataset saved to: {output_path}")
def main():
    """Run feature engineering workflow."""
    reports_dir = os.path.join("reports", "eda")
    make_dir(reports_dir)

    data_path = os.path.join("data", "history_daily_1y.csv")
    df = load_history(data_path)

    print("Creating enhanced features for machine learning...")
    df_enhanced = create_enhanced_features(df)


    clean_data_path = os.path.join("data", "cleaned_data.csv")
    save_enhanced_data(df_enhanced, clean_data_path)


    save_summary(df, reports_dir)

    print(f"Enhanced features created with {len(df_enhanced.columns)} columns")
    print(f"EDA & summary saved in: {reports_dir}")
if __name__ == "__main__":
    main()
