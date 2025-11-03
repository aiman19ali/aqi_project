import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------
def ensure_reports_dir(path: str) -> None:
    """Create reports directory if missing."""
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------
# Data Loading
# ---------------------------------------------------------
def load_daily_history(csv_path: str) -> pd.DataFrame:
    """Load 1-year daily AQI data from CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Expected daily history at {csv_path}. Run api_project11.py first."
        )

    df = pd.read_csv(csv_path)

    if "date" not in df.columns:
        raise ValueError("CSV must contain a 'date' column")

    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.date
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    return df


# ---------------------------------------------------------
# Summaries
# ---------------------------------------------------------
def summarize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive stats."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        print("⚠️ No numeric columns found.")
        return pd.DataFrame()
    return df[numeric_cols].describe().T


# ---------------------------------------------------------
# Plot 1: AQI Trend
# ---------------------------------------------------------
def plot_trends(df: pd.DataFrame):
    """Return AQI trend and 7/30-day rolling means as a figure."""
    ts = df.copy()
    ts["date"] = pd.to_datetime(ts["date"])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts["date"], ts["aqi_index"], label="AQI (daily)", color="#1f77b4")
    ax.plot(ts["date"], ts["aqi_index"].rolling(7, min_periods=1).mean(),
            label="7D Rolling Mean", color="#ff7f0e")
    ax.plot(ts["date"], ts["aqi_index"].rolling(30, min_periods=1).mean(),
            label="30D Rolling Mean", color="#2ca02c")

    ylabel = "AQI Index (0–500)" if ts["aqi_index"].max() > 50 else "AQI Level (1–5)"
    ax.set_title("Daily AQI – 1 Year with Rolling Means")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------
# Plot 2: AQI Boxplot by Month
# ---------------------------------------------------------
def plot_box_by_month(df: pd.DataFrame):
    """Return boxplot of AQI by calendar month."""
    tmp = df.copy()
    tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
    tmp["month"] = tmp["date"].dt.month

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(data=tmp, x="month", y="aqi_index", palette="coolwarm", ax=ax)
    ax.set_title("AQI Distribution by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("AQI Index (0–500)" if tmp["aqi_index"].max() > 50 else "AQI Level (1–5)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------
# Plot 3: Correlation Heatmap
# ---------------------------------------------------------
def plot_components_corr_heatmap(df: pd.DataFrame):
    """Return correlation heatmap of pollutants and features."""
    tmp = df.copy()
    tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
    tmp["year"] = tmp["date"].dt.year
    tmp["month"] = tmp["date"].dt.month
    tmp["day"] = tmp["date"].dt.day

    numeric_cols = tmp.select_dtypes(include=[np.number]).columns.tolist()
    corr = tmp[numeric_cols].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, cmap="vlag", center=0, annot=False, linewidths=0.3, ax=ax)
    ax.set_title("Correlation Heatmap: AQI, Pollutants & Time Features")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------
# Plot 4: Pollutant Concentrations
# ---------------------------------------------------------
def plot_pollutant_concentrations(df: pd.DataFrame):
    """Return bar chart of average pollutant concentrations."""
    pollutant_cols = ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]
    existing_cols = [c for c in pollutant_cols if c in df.columns]

    fig, ax = plt.subplots(figsize=(10, 5))
    if not existing_cols:
        ax.text(0.5, 0.5, "No pollutant columns found!", ha="center", va="center")
        return fig

    avg_values = df[existing_cols].mean().sort_values(ascending=False)
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(avg_values)))
    bars = ax.bar(avg_values.index, avg_values.values, color=colors, edgecolor="black")
    for bar, val in zip(bars, avg_values.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_title("Average Pollutant Concentrations (µg/m³)")
    ax.set_xlabel("Pollutant Type")
    ax.set_ylabel("Concentration (µg/m³)")
    fig.tight_layout()
    return fig
