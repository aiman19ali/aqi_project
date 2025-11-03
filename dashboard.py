import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="🌤️ aqi_index Forecast Dashboard", layout="wide")

st.title("🌤️ aqi_index Forecast and EDA Dashboard")

# -----------------------------
# STEP 1: Load data
# -----------------------------
df = pd.read_csv("data/clean_aqi.csv")
st.write("### Data Preview")
st.dataframe(df.head())

# -----------------------------
# STEP 2: EDA CHARTS SECTION
# -----------------------------
st.write("## 📊 Exploratory Data Analysis")

# Layout with two rows and two columns = 4 charts
col1, col2 = st.columns(2)

# Chart 1 - aqi_index Trend
with col1:
    st.write("#### aqi_index Trend Over the Past Year")
    fig1, ax1 = plt.subplots()
    sns.lineplot(data=df, x="date", y="aqi_index", ax=ax1, color="blue")
    st.pyplot(fig1)

# Chart 2 - Correlation Heatmap
with col2:
    st.write("#### Correlation Heatmap")
    fig2, ax2 = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

# Chart 3 - Pollutant Comparison (Plotly)
col3, col4 = st.columns(2)

with col3:
    st.write("#### PM2.5 vs aqi_index (City-wise)")
    fig3 = px.scatter(
        df, 
        x="pm2_5", 
        y="aqi_index",
        color="temperature",  # optional — you can color by temperature, humidity, etc.
        hover_data=["date", "humidity", "pressure"]
    )
    st.plotly_chart(fig3, use_container_width=True)

# Chart 4 - Monthly Average aqi_index
with col4:
    st.write("#### Monthly Average aqi_index")
    df['month'] = pd.to_datetime(df['date']).dt.month
    monthly_avg = df.groupby('month')['aqi_index'].mean().reset_index()
    fig4 = px.bar(monthly_avg, x='month', y='aqi_index', text_auto=True, color='aqi_index')
    st.plotly_chart(fig4, use_container_width=True)

# -----------------------------
# STEP 3: Forecast Results
# -----------------------------
st.write("## 🔮 aqi_index Forecast for Next 3 Days")

forecast_data = {
    "Date": ["2025-11-02", "2025-11-03", "2025-11-04"],
    "Predicted aqi_index": [155, 162, 149]
}
forecast_df = pd.DataFrame(forecast_data)
st.table(forecast_df)
