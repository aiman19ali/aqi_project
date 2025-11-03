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
# Plot 1: AQI Trend (Category Bands)
# ---------------------------------------------------------
def plot_aqi_trend(df: pd.DataFrame, save_path: str):
    """Plot AQI trend over time with AQI category bands."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    fig, ax = plt.subplots(figsize=(12, 5))

    # AQI category bands
    aqi_bands = [
        (0, 50, "Good", "lightgreen"),
        (51, 100, "Moderate", "khaki"),
        (101, 150, "Unhealthy (Sensitive)", "orange"),
        (151, 200, "Unhealthy", "tomato"),
        (201, 300, "Very Unhealthy", "violet"),
    ]

    for low, high, label, color in aqi_bands:
        ax.axhspan(low, high, facecolor=color, alpha=0.3, label=label)

    # Plot AQI line
    ax.plot(df["date"], df["aqi_index"], color="steelblue", linewidth=2, marker="o", markersize=3)

    ax.set_title("AQI Trend Over Time (Category-based)")
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI Index (0–300)")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------
# Plot 2: AQI Distribution by Month
# ---------------------------------------------------------
def plot_box_by_month(df: pd.DataFrame, save_path: str):
    """Boxplot of AQI distribution by month."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month_name()

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df, x="month", y="aqi_index", palette="coolwarm", ax=ax)
    ax.set_title("AQI Distribution by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("AQI Index (0–300)")
    plt.xticks(rotation=45)
    fig.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------
# Plot 3: Correlation Heatmap
# ---------------------------------------------------------
def plot_correlation_heatmap(df: pd.DataFrame, save_path: str):
    """Heatmap of correlation between pollutants and AQI."""
    numeric_cols = df.select_dtypes(include=np.number).columns
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, cmap="RdBu_r", center=0, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap (AQI vs Pollutants)")
    fig.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------
# Plot 4: Average Pollutant Concentrations
# ---------------------------------------------------------
def plot_pollutant_concentrations(df: pd.DataFrame, save_path: str):
    """Bar chart of average pollutant concentrations."""
    pollutants = ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]
    existing = [p for p in pollutants if p in df.columns]

    avg_values = df[existing].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(avg_values.index, avg_values.values, color="salmon", edgecolor="black")
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f"{height:.2f}",
                ha="center", va="bottom", fontsize=8)

    ax.set_title("Average Pollutant Concentrations (µg/m³)")
    ax.set_xlabel("Pollutant Type")
    ax.set_ylabel("Average Concentration (µg/m³)")
    fig.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    data_path = "data/clean_aqi.csv"
    reports_dir = "reports/eda_charts"
    ensure_reports_dir(reports_dir)

    df = load_daily_history(data_path)

    print("✅ Generating EDA charts...")

    plot_aqi_trend(df, os.path.join(reports_dir, "aqi_trend.png"))
    plot_box_by_month(df, os.path.join(reports_dir, "aqi_boxplot_by_month.png"))
    plot_correlation_heatmap(df, os.path.join(reports_dir, "correlation_heatmap.png"))
    plot_pollutant_concentrations(df, os.path.join(reports_dir, "pollutant_concentrations.png"))

    print(f"✅ Charts saved in: {os.path.abspath(reports_dir)}")
