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
    Load 1-year daily AQI history (CSV) saved from api_project.py.
    Cleans up and returns a properly formatted DataFrame.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}. Run api_project.py first.")

    df = pd.read_csv(csv_path)

    if "date" not in df.columns:
        raise ValueError("CSV must have a 'date' column!")

    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.date
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def save_summary(df: pd.DataFrame, out_dir: str):
    """Generate basic descriptive stats and correlations."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    summary = df[numeric_cols].describe().T
    corr = df[numeric_cols].corr()

    summary.to_csv(os.path.join(out_dir, "summary_stats.csv"))
    corr.to_csv(os.path.join(out_dir, "correlations.csv"))


def plot_aqi_trend(df: pd.DataFrame, out_dir: str):
    """Plot AQI daily trend with 7-day and 30-day rolling averages."""
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.set_index("date")

    plt.figure(figsize=(10, 4))
    plt.plot(data.index, data["aqi_index"], label="Daily AQI", color="#1f77b4")
    plt.plot(data.index, data["aqi_index"].rolling(7).mean(), label="7D MA", color="#ff7f0e")
    plt.plot(data.index, data["aqi_index"].rolling(30).mean(), label="30D MA", color="#2ca02c")
    plt.title("Daily AQI – Last 1 Year (with Rolling Means)")
    plt.ylabel("AQI Index")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "aqi_trend_rolling.png"), dpi=150)
    plt.close()


def plot_monthly_box(df: pd.DataFrame, out_dir: str):
    """Boxplot to visualize AQI distribution across months."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month

    plt.figure(figsize=(10, 4))
    sns.boxplot(x="month", y="aqi_index", data=df)
    plt.title("AQI Distribution by Month")
    plt.xlabel("Month")
    plt.ylabel("AQI Index")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "aqi_box_by_month.png"), dpi=150)
    plt.close()


def plot_corr_heatmap(df: pd.DataFrame, out_dir: str):
    """Show correlations between AQI and pollutants."""
    numeric_cols = df.select_dtypes(include=np.number).columns
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, cmap="vlag", center=0)
    plt.title("Correlation Heatmap (AQI vs Components)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "corr_heatmap.png"), dpi=150)
    plt.close()


def plot_acf_pacf(df: pd.DataFrame, out_dir: str):
    """Plot ACF and PACF to check autocorrelation structure."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    series = df["aqi_index"].astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_acf(series, ax=axes[0], lags=40)
    plot_pacf(series, ax=axes[1], lags=40, method="ywm")
    axes[0].set_title("ACF – Daily AQI")
    axes[1].set_title("PACF – Daily AQI")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "aqi_acf_pacf.png"), dpi=150)
    plt.close()


def create_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create enhanced features for machine learning."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    
    # Time-based features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["season"] = ((df["month"] % 12 + 3) // 3)  # 1=Winter, 2=Spring, 3=Summer, 4=Fall
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    
    # Lag features (past AQI values)
    df["aqi_lag_1"] = df["aqi_index"].shift(1)
    df["aqi_lag_2"] = df["aqi_index"].shift(2)
    df["aqi_lag_7"] = df["aqi_index"].shift(7)
    
    # Rolling statistics
    df["aqi_3day_avg"] = df["aqi_index"].rolling(window=3, min_periods=1).mean()
    df["aqi_7day_avg"] = df["aqi_index"].rolling(window=7, min_periods=1).mean()
    df["aqi_30day_avg"] = df["aqi_index"].rolling(window=30, min_periods=1).mean()
    
    # Weather interaction features
    df["temp_humidity_interaction"] = df["temperature"] * df["humidity"]
    df["pressure_wind_interaction"] = df["pressure"] * df["wind_speed"]
    
    # Pollutant ratios
    df["pm25_pm10_ratio"] = df["pm2_5"] / (df["pm10"] + 1e-6)  # Avoid division by zero
    df["co_no2_ratio"] = df["co"] / (df["no2"] + 1e-6)
    
    # Fill missing values
    df = df.ffill().bfill()
    
    return df


def save_enhanced_data(df: pd.DataFrame, output_path: str):
    """Save enhanced dataset for ML training."""
    df.to_csv(output_path, index=False)
    print(f"Enhanced dataset saved to: {output_path}")


def main():
    """Run the full feature engineering workflow on the 1-year AQI dataset."""
    reports_dir = os.path.join("reports", "eda")
    make_dir(reports_dir)

    data_path = os.path.join("data", "history_daily_1y.csv")
    df = load_history(data_path)

    # Create enhanced features for ML
    print("Creating enhanced features for machine learning...")
    df_enhanced = create_enhanced_features(df)
    
    # Save enhanced dataset
    clean_data_path = os.path.join("data", "clean_aqi.csv")
    save_enhanced_data(df_enhanced, clean_data_path)

    # Save stats
    save_summary(df, reports_dir)

    # Generate visuals
    plot_aqi_trend(df, reports_dir)
    plot_monthly_box(df, reports_dir)
    plot_corr_heatmap(df, reports_dir)

    try:
        plot_acf_pacf(df, reports_dir)
    except Exception as e:
        print("ACF/PACF skipped due to error:", e)

    print(f"EDA results saved in: {reports_dir}")
    print(f"Enhanced features created with {len(df_enhanced.columns)} columns")


if __name__ == "__main__":
    main()
