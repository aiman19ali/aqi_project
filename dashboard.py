import streamlit as st
import requests
from datetime import datetime

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="🌍 AQI Prediction Dashboard", layout="wide")
st.title("🌤️ Air Quality Index (AQI) Prediction Dashboard")
st.markdown("Compare **Predicted AQI (from your ML model)** vs **Live AQI (from AQICN API)** for **Karachi**.")

# ---------- INPUT SECTION ----------
st.subheader("🧾 Enter Environmental Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    co = st.number_input("CO", value=0.5)
    no = st.number_input("NO", value=0.1)
    no2 = st.number_input("NO₂", value=10.0)
    o3 = st.number_input("O₃", value=20.0)

with col2:
    so2 = st.number_input("SO₂", value=5.0)
    pm2_5 = st.number_input("PM2.5", value=35.0)
    pm10 = st.number_input("PM10", value=50.0)
    nh3 = st.number_input("NH₃", value=1.0)

with col3:
    temperature = st.number_input("Temperature (°C)", value=25.0)
    humidity = st.number_input("Humidity (%)", value=60.0)
    pressure = st.number_input("Pressure (hPa)", value=1013.0)
    wind_speed = st.number_input("Wind Speed (m/s)", value=2.0)

# ---------- PREDICTION BUTTON ----------
if st.button("🚀 Predict AQI"):
    data = {
        "co": co, "no": no, "no2": no2, "o3": o3, "so2": so2,
        "pm2_5": pm2_5, "pm10": pm10, "nh3": nh3,
        "temperature": temperature, "humidity": humidity,
        "pressure": pressure, "wind_speed": wind_speed
    }

    # ---------- CALL FASTAPI BACKEND ----------
    backend_url = "http://127.0.0.1:8000"
    predicted_aqi, live_aqi = None, None

    # --- Get predicted AQI ---
    try:
        res = requests.post(f"{backend_url}/predict", json=data)
        if res.status_code == 200:
            predicted = res.json()
            predicted_aqi = predicted.get("predicted_aqi")
            predicted_time = predicted.get("timestamp")
        else:
            st.error(" Could not get prediction from backend.")
    except Exception:
        st.error(" Cannot connect to FastAPI backend. Please ensure it’s running.")
    
    # --- Get live AQI ---
    try:
        live_res = requests.get(f"{backend_url}/live_aqi")
        if live_res.status_code == 200:
            live_data = live_res.json()
            live_aqi = live_data.get("live_aqi")
            live_time = live_data.get("timestamp")
        else:
            st.error(" Could not fetch live AQI.")
    except Exception:
        st.error(" Cannot connect to FastAPI backend for live AQI.")

    # ---------- DISPLAY RESULTS ----------
    if predicted_aqi is not None and live_aqi is not None:
        delta = round(live_aqi - predicted_aqi, 1)
        st.subheader("📊 AQI Comparison")

        col1, col2, col3 = st.columns(3)

        # Helper: AQI category color and label
        def aqi_category(aqi):
            if aqi <= 50:
                return "🟢 Good", "green"
            elif aqi <= 100:
                return "🟡 Moderate", "yellow"
            elif aqi <= 150:
                return "🟠 Unhealthy (Sensitive Groups)", "orange"
            elif aqi <= 200:
                return "🔴 Unhealthy", "red"
            elif aqi <= 300:
                return "🟣 Very Unhealthy", "purple"
            else:
                return "⚫ Hazardous", "maroon"

        pred_label, pred_color = aqi_category(predicted_aqi)
        live_label, live_color = aqi_category(live_aqi)

        col1.metric("Predicted AQI", f"{predicted_aqi}", delta=None)
        col1.markdown(f"<h5 style='color:{pred_color}'>{pred_label}</h5>", unsafe_allow_html=True)

        col2.metric("Live AQI", f"{live_aqi}", delta=None)
        col2.markdown(f"<h5 style='color:{live_color}'>{live_label}</h5>", unsafe_allow_html=True)

        col3.metric("Difference (Live - Predicted)", f"{delta}", delta_color="inverse")

        st.success(f" Data Updated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.warning(" AQI comparison not available. Please check backend connection or API response.")
