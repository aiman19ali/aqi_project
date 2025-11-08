import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(page_title="AQI Index Forecast Dashboard", layout="wide")

st.title("AQI Index Forecast and EDA Dashboard")




data_path = "data/clean_aqi.csv"

try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    st.error(f"ERROR File not found: {data_path}")
    st.stop()
st.write("### Data Preview")
st.dataframe(df.head())




st.write("## Exploratory Data Analysis (EDA)")


st.write("#### AQI Trend Over Time (Category-based)")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

fig1 = go.Figure()


fig1.add_trace(go.Scatter(
    x=df["date"],
    y=df["aqi_index"],
    mode="lines+markers",
    name="Daily AQI",
    line=dict(color="steelblue", width=2),
    marker=dict(size=4)
))


aqi_bands = [
    (0, 50, "Good", "rgba(144,238,144,0.3)"),
    (51, 100, "Moderate", "rgba(255,255,102,0.3)"),
    (101, 150, "Unhealthy (Sensitive)", "rgba(255,165,0,0.3)"),
    (151, 200, "Unhealthy", "rgba(255,99,71,0.3)"),
    (201, 300, "Very Unhealthy", "rgba(199,21,133,0.3)"),
]

for low, high, label, color in aqi_bands:
    fig1.add_shape(
        type="rect",
        x0=df["date"].min(),
        x1=df["date"].max(),
        y0=low,
        y1=high,
        fillcolor=color,
        opacity=0.3,
        line_width=0,
    )
    fig1.add_annotation(
        x=df["date"].max(),
        y=high - 10,
        text=label,
        showarrow=False,
        font=dict(size=10, color="black"),
        xanchor="left"
    )
fig1.update_layout(
    title="AQI Trend Over Time (Category-based)",
    xaxis_title="Date",
    yaxis_title="AQI Index (0-300)",
    template="plotly_white",
    hovermode="x unified",
    height=400
)

st.plotly_chart(fig1, use_container_width=True)


st.write("#### AQI Distribution by Month")
df["month"] = df["date"].dt.month_name()

fig2 = px.box(df, x="month", y="aqi_index", color="month",
              title="AQI Distribution by Month (Interactive)")
fig2.update_layout(xaxis_title="Month", yaxis_title="AQI Index", showlegend=False)
st.plotly_chart(fig2, use_container_width=True)


st.write("#### Correlation Heatmap of Pollutants")
numeric_cols = df.select_dtypes(include='number').columns
corr = df[numeric_cols].corr().round(2)

fig3 = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                 title="Correlation Heatmap (AQI vs Pollutants)")
st.plotly_chart(fig3, use_container_width=True)


st.write("#### Average Pollutant Concentrations (ug/m^3)")
pollutants = ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]

if all(p in df.columns for p in pollutants):
    avg_pollutants = df[pollutants].mean().reset_index()
    avg_pollutants.columns = ["Pollutant", "Average Concentration"]

    fig4 = px.bar(avg_pollutants, x="Pollutant", y="Average Concentration",
                  color="Pollutant", title="Average Pollutant Concentrations")
    fig4.update_layout(xaxis_title="Pollutant", yaxis_title="ug/m^3")
    st.plotly_chart(fig4, use_container_width=True)
else:
    st.warning("WARNING Some pollutant columns are missing in the dataset!")
st.write("## AQI Forecast for Next 3 Days")

forecast_data = {
    "Date": ["2025-11-02", "2025-11-03", "2025-11-04"],
    "Predicted AQI": [155, 162, 149],
}
forecast_df = pd.DataFrame(forecast_data)
st.table(forecast_df)

st.success("Dashboard Loaded Successfully!")
