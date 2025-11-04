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
import calendar

def plot_box_by_month(df: pd.DataFrame, save_path: str):
    """Boxplot of AQI distribution by month, ordered Jan → Dec."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    # numeric month and month name
    df["month_num"] = df["date"].dt.month
    df["month"] = df["date"].dt.month_name()

    # define full calendar order and make month a categorical with that order
    month_order = [calendar.month_name[i] for i in range(1, 13)]
    df["month"] = pd.Categorical(df["month"], categories=month_order, ordered=True)

    # If you only want months that actually appear in the data (keeps order)
    months_present = [m for m in month_order if m in df["month"].cat.categories and m in df["month"].unique()]

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df, x="month", y="aqi_index", order=months_present, palette="coolwarm", ax=ax)
    ax.set_title("AQI Distribution by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("AQI Index (0–300)")
    plt.xticks(rotation=45)
    fig.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



# ---------------------------------------------------------
# ✅ Plot 3: Enhanced Correlation Heatmap (Fixed)
# ---------------------------------------------------------
def plot_correlation_heatmap(df: pd.DataFrame, save_path: str):
    """Correlation heatmap of AQI, pollutants, and time features."""
    tmp = df.copy()
    tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
    tmp["year"] = tmp["date"].dt.year
    tmp["month"] = tmp["date"].dt.month
    tmp["day"] = tmp["date"].dt.day

    # Select numeric columns only
    numeric_cols = tmp.select_dtypes(include=[np.number]).columns.tolist()
    corr = tmp[numeric_cols].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        corr,
        cmap="vlag",
        center=0,
        annot=False,
        linewidths=0.3,
        cbar_kws={"label": "Correlation Coefficient"},
        ax=ax
    )
    ax.set_title("Correlation Heatmap: AQI, Pollutants & Time Features")
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
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

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
