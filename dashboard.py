import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests

# ✅ Streamlit settings
st.set_page_config(page_title="🌤️ AQI Dashboard", layout="wide")
st.title("🌤️ AQI Index Forecast & EDA Dashboard")

# ✅ Load Data
data_path = "data/clean_aqi.csv"

try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    st.error(f"❌ File not found: {data_path}")
    st.stop()

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

st.write("### 🧾 Data Preview")
st.dataframe(df.head())

# ✅ Create Tabs for Interactive Charts
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 AQI Trend",
    "📦 Monthly Distribution",
    "🔥 Heatmap",
    "🌫️ Pollutants"
])

# --------------------------------------------------------------
# ✅ TAB 1 — AQI TREND (Category Bands + 7D & 30D Rolling Average)
# --------------------------------------------------------------
with tab1:
    st.write("#### 📈 AQI Trend Over Time with Health Categories")

    # Rolling averages
    df["AQI_7D"] = df["aqi_index"].rolling(7, min_periods=1).mean()
    df["AQI_30D"] = df["aqi_index"].rolling(30, min_periods=1).mean()

    fig1 = go.Figure()

    # AQI Category Ranges (Colored Bands)
    aqi_bands = [
        (0, 50, "Good", "rgba(0,255,0,0.20)"),
        (51, 100, "Moderate", "rgba(255,255,0,0.20)"),
        (101, 150, "Unhealthy (Sensitive)", "rgba(255,165,0,0.20)"),
        (151, 200, "Unhealthy", "rgba(255,80,80,0.20)"),
        (201, 300, "Very Unhealthy", "rgba(180,0,150,0.20)"),
    ]

    for low, high, label, color in aqi_bands:
        fig1.add_shape(
            type="rect",
            x0=df["date"].min(),
            x1=df["date"].max(),
            y0=low,
            y1=high,
            fillcolor=color,
            opacity=0.25,
            line_width=0,
            layer="below"
        )
        fig1.add_annotation(
            x=df["date"].max(),
            y=high - 8,
            text=label,
            showarrow=False,
            font=dict(size=10),
            xanchor="left"
        )

    # Daily AQI Line
    fig1.add_trace(go.Scatter(
        x=df["date"],
        y=df["aqi_index"],
        mode="lines+markers",
        name="Daily AQI",
        line=dict(color="royalblue", width=2),
        marker=dict(size=4)
    ))

    # Rolling lines
    fig1.add_trace(go.Scatter(
        x=df["date"],
        y=df["AQI_7D"],
        mode="lines",
        name="7-Day Moving Avg",
        line=dict(color="darkorange", width=2)
    ))

    fig1.add_trace(go.Scatter(
        x=df["date"],
        y=df["AQI_30D"],
        mode="lines",
        name="30-Day Moving Avg",
        line=dict(color="green", width=2)
    ))

    fig1.update_layout(
        title="AQI Trend Over Time (Category Bands + Rolling Average)",
        xaxis_title="Date",
        yaxis_title="AQI Index (0–300)",
        template="plotly_white",
        hovermode="x unified",
        height=450
    )

    st.plotly_chart(fig1, use_container_width=True)

# --------------------------------------------------------------
# ✅ TAB 2 — AQI BOXPLOT BY MONTH
# --------------------------------------------------------------
with tab2:
    st.write("#### 📦 AQI Distribution by Month")
    df["month"] = df["date"].dt.month_name()

    fig2 = px.box(df, x="month", y="aqi_index", color="month",
                  title="AQI Distribution by Month")
    fig2.update_layout(xaxis_title="Month", yaxis_title="AQI Index", showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------------------
# ✅ TAB 3 — CORRELATION HEATMAP
# --------------------------------------------------------------
with tab3:
    st.write("#### 🔥 Correlation Heatmap of Pollutants & AQI")
    numeric = df.select_dtypes(include='number')
    corr = numeric.corr().round(2)

    fig3 = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                     title="Correlation Heatmap")
    st.plotly_chart(fig3, use_container_width=True)

# --------------------------------------------------------------
# ✅ TAB 4 — POLLUTANT BAR CHART
# --------------------------------------------------------------
with tab4:
    st.write("#### 🌫️ Average Pollutant Concentrations (µg/m³)")
    pollutants = ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]

    existing = [p for p in pollutants if p in df.columns]

    if existing:
        avg_vals = df[existing].mean().reset_index()
        avg_vals.columns = ["Pollutant", "Avg_Concentration"]

        fig4 = px.bar(avg_vals, x="Pollutant", y="Avg_Concentration",
                      color="Pollutant", title="Average Pollutant Levels")
        fig4.update_layout(yaxis_title="µg/m³")
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.warning("⚠️ Pollutant columns missing!")

# --------------------------------------------------------------
# ✅ Forecast table (example)
# --------------------------------------------------------------
st.subheader("🔮 AQI Forecast (Next 3 Days)")
try:
    resp = requests.get("http://127.0.0.1:8000/get_forecast")
    data = resp.json()

    if data["status"] == "success":
        forecast_df = pd.DataFrame(data["forecasts"])
        st.dataframe(forecast_df)

        # highest AQI warning
        highest = forecast_df["predicted_aqi"].max()
        st.warning(f"Highest forecasted AQI: {highest}")

    else:
        st.error("Backend error: " + data.get("message", "unknown"))

except:
    st.error("❌ Cannot connect to backend for forecast.")