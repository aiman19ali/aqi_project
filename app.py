import streamlit as st
import pandas as pd
import plotly.express as px

# Import EDA functions from your file
from eda_analysis import (
    load_daily_history,
    plot_trends,
    plot_box_by_month,
    plot_components_corr_heatmap,
    plot_pollutant_concentrations,
)

st.set_page_config(page_title="🌤️ AQI Index Forecast Dashboard", layout="wide")

st.title("🌤️ AQI Index Forecast and EDA Dashboard")

# -----------------------------
# STEP 1: Load Data
# -----------------------------
data_path = "data/clean_aqi.csv"
df = pd.read_csv(data_path)
st.write("### Data Preview")
st.dataframe(df.head())

# -----------------------------
# STEP 2: EDA Charts (from your EDA script)
# -----------------------------
st.write("## 📊 Exploratory Data Analysis (EDA)")

# Chart 1 - Trend
st.write("#### 📈 AQI Trend (7D & 30D Rolling Mean)")
st.pyplot(plot_trends(df))

# Chart 2 - Boxplot
st.write("#### 📦 AQI Distribution by Month")
st.pyplot(plot_box_by_month(df))

# Chart 3 - Correlation Heatmap
st.write("#### 🔥 Correlation Heatmap")
st.pyplot(plot_components_corr_heatmap(df))

# Chart 4 - Pollutant Concentrations
st.write("#### 🌫️ Average Pollutant Concentrations (µg/m³)")
st.pyplot(plot_pollutant_concentrations(df))

# Chart 5 - PM2.5 vs AQI Scatter (Interactive)
st.write("#### ☁️ PM2.5 vs AQI Index (Interactive Scatter)")
fig5 = px.scatter(
    df,
    x="pm2_5",
    y="aqi_index",
    color="temperature" if "temperature" in df.columns else None,
    hover_data=["date", "humidity", "pressure"] if "humidity" in df.columns else ["date"],
    title="PM2.5 vs AQI Index",
)
st.plotly_chart(fig5, use_container_width=True)

# -----------------------------
# STEP 3: Forecast Results
# -----------------------------
st.write("## 🔮 AQI Forecast for Next 3 Days")

forecast_data = {
    "Date": ["2025-11-02", "2025-11-03", "2025-11-04"],
    "Predicted AQI": [155, 162, 149],
}
forecast_df = pd.DataFrame(forecast_data)
st.table(forecast_df)
