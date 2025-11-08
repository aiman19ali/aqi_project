import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


st.set_page_config(page_title="AQI Dashboard", layout="wide")
st.title("AQI Index Forecast & EDA Dashboard")


data_path = "data/clean_aqi.csv"

try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    st.error(f"Error: file not found: {data_path}")
    st.stop()
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")




@st.cache_data(show_spinner=False)
def compute_model_metrics() -> pd.DataFrame:
    """
    Compute hold-out evaluation metrics for the trained AQI forecast models.
    """
    data_path_metrics = "data/cleaned_data.csv"
    if not os.path.exists(data_path_metrics):
        raise FileNotFoundError(f"{data_path_metrics} not found.")
    raw = pd.read_csv(data_path_metrics)
    features = [
        "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
        "temperature", "humidity", "pressure", "wind_speed",
        "day_of_week"
    ]
    for lag in [1, 2, 3]:
        raw[f"aqi_lag_{lag}"] = raw["aqi_7day_avg"].shift(lag)
        features.append(f"aqi_lag_{lag}")
    raw = raw.bfill()
    numeric_columns = raw.select_dtypes(include=np.number).columns
    raw[numeric_columns] = raw[numeric_columns].fillna(raw[numeric_columns].mean())
    raw["aqi_24h"] = raw["aqi_index"].shift(-24)
    raw["aqi_48h"] = raw["aqi_index"].shift(-48)
    raw["aqi_72h"] = raw["aqi_index"].shift(-72)
    raw.dropna(subset=["aqi_24h", "aqi_48h", "aqi_72h"], inplace=True)
    targets = {
        "24h": "aqi_24h",
        "48h": "aqi_48h",
        "72h": "aqi_72h",
    }
    records = []
    for horizon, target_column in targets.items():
        model_path = os.path.join("models", f"xgb_aqi_{horizon}.joblib")
        if not os.path.exists(model_path):
            continue
        X = raw[features]
        y = raw[target_column]
        _X_train, X_test, _y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = joblib.load(model_path)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        r2 = r2_score(y_test, predictions)
        records.append({
            "Horizon": horizon,
            "MAE": mae,
            "RMSE": rmse,
            "R²": r2,
        })
    return pd.DataFrame(records)



def build_metrics_plot(metrics_df: pd.DataFrame) -> go.Figure:
    """
    Build comparative bar chart for MAE, RMSE, and R² across forecast horizons.
    """
    metric_names = {
        "MAE": "Mean Absolute Error",
        "RMSE": "Root Mean Squared Error",
        "R²": "R² Score",
    }
    melted = metrics_df.melt(
        id_vars="Horizon",
        value_vars=list(metric_names.keys()),
        var_name="Metric",
        value_name="Value"
    )
    melted["Metric"] = melted["Metric"].map(metric_names)
    fig = px.bar(
        melted,
        x="Horizon",
        y="Value",
        color="Metric",
        barmode="group",
        title="Model metrics by forecast horizon"
    )
    fig.update_layout(yaxis_title="Score", legend_title="Metric")
    return fig



tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "AQI Trend",
    "Monthly Distribution",
    "Heatmap",
    "Pollutants",
    "Explainability",
    "Predict",
    "Data"
])

with tab7:
    st.write("### Data Preview")
    st.dataframe(df.head())
with tab6:
    st.write("#### Make Prediction")

    col_w, col_p1, col_p2 = st.columns([1, 1, 1])

    with col_w:
        st.write("Weather Features")
        temperature = st.slider("Temperature (degC)", -10.0, 50.0, 28.0, 0.1)
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 65.0, 0.1)
        pressure = st.slider("Pressure (hPa)", 950.0, 1050.0, 1008.0, 0.1)
        wind_speed = st.slider("Wind Speed (m/s)", 0.0, 30.0, 5.0, 0.1)
    with col_p1:
        st.write("Pollutant Features")
        pm2_5 = st.slider("PM2.5 (ug/m^3)", 0.0, 500.0, 60.0, 0.1)
        pm10 = st.slider("PM10 (ug/m^3)", 0.0, 600.0, 100.0, 0.1)
        no = st.slider("NO (ug/m^3)", 0.0, 200.0, 10.0, 0.1)
        no2 = st.slider("NO2 (ug/m^3)", 0.0, 400.0, 40.0, 0.1)
    with col_p2:
        st.write("Additional Pollutants")
        o3 = st.slider("O3 (ug/m^3)", 0.0, 200.0, 70.0, 0.1)
        so2 = st.slider("SO2 (ug/m^3)", 0.0, 600.0, 20.0, 0.1)
        co = st.slider("CO (ug/m^3)", 0.0, 10000.0, 800.0, 1.0)
        nh3 = st.slider("NH3 (ug/m^3)", 0.0, 200.0, 8.0, 0.1)
    payload = {
        "co": float(co),
        "no": float(no),
        "no2": float(no2),
        "o3": float(o3),
        "so2": float(so2),
        "pm2_5": float(pm2_5),
        "pm10": float(pm10),
        "nh3": float(nh3),
        "temperature": float(temperature),
        "humidity": float(humidity),
        "pressure": float(pressure),
        "wind_speed": float(wind_speed),
    }

    def aqi_category(val: float) -> str:
        if val <= 50: return "Good"
        if val <= 100: return "Moderate"
        if val <= 150: return "Unhealthy (Sensitive)"
        if val <= 200: return "Unhealthy"
        if val <= 300: return "Very Unhealthy"
        return "Hazardous"
    if st.button("Predict AQI", use_container_width=True):
        try:
            resp = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=20)
            if resp.status_code != 200:
                st.error(f"Backend error: {resp.status_code} {resp.text}")
            else:
                data = resp.json()
                aqi_val = float(data.get("predicted_aqi", 0))
                st.subheader("Prediction Results")
                st.metric("AQI Value", f"{aqi_val:.1f}")
                st.info(aqi_category(aqi_val))
        except Exception as e:
            st.error(f"Request failed: {e}")
with tab5:
    st.write("#### Model Explainability - SHAP & LIME")
    try:
        metrics_df = compute_model_metrics()
        if metrics_df.empty:
            st.info("Train models to view validation metrics for 24h, 48h, and 72h horizons.")
        else:
            formatted = metrics_df.copy()
            formatted["MAE"] = formatted["MAE"].round(3)
            formatted["RMSE"] = formatted["RMSE"].round(3)
            formatted["R²"] = formatted["R²"].round(3)
            st.write("Model validation metrics (hold-out split)")
            st.dataframe(formatted, use_container_width=True)
            metrics_fig = build_metrics_plot(formatted)
            st.plotly_chart(metrics_fig, use_container_width=True)
    except FileNotFoundError as missing_file_error:
        st.warning(str(missing_file_error))
    except Exception as metrics_error:
        st.error(f"Failed to compute metrics: {metrics_error}")
    compute_explain = st.button("Compute explanations", key="explain_compute")
    if compute_explain:
        try:
            try:
                import shap
                from lime.lime_tabular import LimeTabularExplainer
            except Exception as e:
                st.error("Install explainability packages: pip install shap lime joblib")
                raise
            model_candidates = [
                "models/model_24h_1year.pkl",
                "models/model_24h.pkl",
                "models/aqi_model.pkl",
                "models/xgb_aqi_24h.joblib",
            ]
            model = None
            for mp in model_candidates:
                try:
                    model = joblib.load(mp)
                    break
                except Exception:
                    continue
            numeric_cols = [c for c in df.select_dtypes(include='number').columns if c != "aqi_index"]

            model_feature_columns = None

            if model is not None:

                names_in = getattr(model, "feature_names_in_", None)
                if names_in is not None:
                    try:
                        model_feature_columns = list(names_in)
                    except Exception:
                        pass
                if model_feature_columns is None:
                    try:
                        booster = getattr(model, "get_booster", lambda: None)()
                        names = getattr(booster, "feature_names", None)
                        if names:
                            model_feature_columns = list(names)
                    except Exception:
                        pass
            if model_feature_columns is None:
                for fc_path in [
                    "models/feature_columns.pkl",
                    "models/pollutants/feature_columns.pkl",
                ]:
                    try:
                        if os.path.exists(fc_path):
                            model_feature_columns = joblib.load(fc_path)
                            break
                    except Exception:
                        pass
            if model_feature_columns is None:
                model_feature_columns = list(numeric_cols)
            work = df.copy()
            if "date" in work.columns:
                work["date"] = pd.to_datetime(work["date"], errors="coerce")
                work["year"] = work["date"].dt.year
                work["day"] = work["date"].dt.day
                work["day_of_year"] = work["date"].dt.dayofyear
            if "aqi_index" in work.columns:
                for lag in [1, 2, 3, 7, 14]:
                    name = f"aqi_index_lag_{lag}"
                    if name in model_feature_columns and name not in work.columns:
                        work[name] = work["aqi_index"].shift(lag)
                for win in [3, 7, 14, 30]:
                    base = f"aqi_index_rolling_"
                    if f"{base}mean_{win}" in model_feature_columns:
                        work[f"{base}mean_{win}"] = work["aqi_index"].rolling(win, min_periods=1).mean()
                    if f"{base}std_{win}" in model_feature_columns:
                        work[f"{base}std_{win}"] = work["aqi_index"].rolling(win, min_periods=1).std().fillna(0.0)
                    if f"{base}max_{win}" in model_feature_columns:
                        work[f"{base}max_{win}"] = work["aqi_index"].rolling(win, min_periods=1).max()
                    if f"{base}min_{win}" in model_feature_columns:
                        work[f"{base}min_{win}"] = work["aqi_index"].rolling(win, min_periods=1).min()
            if "temp_pressure_interaction" in model_feature_columns:
                work["temp_pressure_interaction"] = work.get("temperature", 0.0) * work.get("pressure", 0.0)
            if "wind_temp_interaction" in model_feature_columns:
                work["wind_temp_interaction"] = work.get("wind_speed", 0.0) * work.get("temperature", 0.0)
            if "humidity_pressure_interaction" in model_feature_columns:
                work["humidity_pressure_interaction"] = work.get("humidity", 0.0) * work.get("pressure", 0.0)
            if "wind_cleaning_effect" in model_feature_columns:

                work["wind_cleaning_effect"] = work.get("wind_speed", 0.0) * (100.0 - work.get("humidity", 0.0))
            if "temp_humidity_interaction" in model_feature_columns:
                work["temp_humidity_interaction"] = work.get("temperature", 0.0) * work.get("humidity", 0.0)
            if "pressure_wind_interaction" in model_feature_columns:
                work["pressure_wind_interaction"] = work.get("pressure", 0.0) * work.get("wind_speed", 0.0)
            if "pm25_pm10_ratio" in model_feature_columns and ("pm2_5" in work.columns and "pm10" in work.columns):
                work["pm25_pm10_ratio"] = work["pm2_5"] / (work["pm10"] + 1e-6)
            if "co_no2_ratio" in model_feature_columns and ("co" in work.columns and "no2" in work.columns):
                work["co_no2_ratio"] = work["co"] / (work["no2"] + 1e-6)
            final_features = []
            for col in model_feature_columns:
                if col in work.columns:
                    final_features.append(col)
                else:
                    work[col] = 0.0
                    final_features.append(col)
            if not final_features:
                st.error("No numeric feature columns available for explainability.")
            else:
                X = work[final_features].copy()

                X = X.fillna(X.mean(numeric_only=True)).head(300)
                if X.empty:
                    st.error("Not enough data for explainability.")
                else:

                    st.write("SHAP (interactive)")
                    with st.spinner("Computing SHAP..."):
                        if model is not None:
                            def _predict_nd(arr):
                                try:

                                    return model.predict(pd.DataFrame(arr, columns=final_features))
                                except Exception:
                                    return model.predict(arr)
                            explainer = shap.Explainer(_predict_nd, X)
                        else:
                            explainer = shap.Explainer(lambda x: np.repeat(df["aqi_index"].mean(), len(x)), X)
                        sv = explainer(X)
                        vals = sv.values if hasattr(sv, "values") else np.array(sv)
                        mean_abs = np.mean(np.abs(vals), axis=0)
                        imp_df = pd.DataFrame({"feature": final_features, "importance": mean_abs}).sort_values("importance", ascending=False)


                        os.makedirs("reports/explainability", exist_ok=True)
                        imp_df.to_csv("reports/explainability/shap_importance.csv", index=False)

                        fig_imp = px.bar(imp_df, x="feature", y="importance", title="Feature importance (mean |SHAP|)")
                        fig_imp.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig_imp, use_container_width=True)

                        dep_feature = st.selectbox("Dependence feature", imp_df["feature"].tolist(), index=0)
                        color_by = st.selectbox("Color by", ["None"] + final_features, index=0)
                        j = final_features.index(dep_feature)
                        yvals = vals[:, j]
                        fig_dep = px.scatter(
                            x=X[dep_feature], y=yvals,
                            color=None if color_by == "None" else X[color_by],
                            labels={"x": dep_feature, "y": "SHAP value"},
                            title=f"SHAP dependence: {dep_feature}"
                        )
                        st.plotly_chart(fig_dep, use_container_width=True)
                    st.write("LIME (top contributions)")
                    with st.spinner("Computing LIME..."):
                        def _predict_fn(arr):
                            if model is None:
                                return [float(df["aqi_index"].mean())] * len(arr)
                            try:
                                return model.predict(pd.DataFrame(arr, columns=final_features))
                            except Exception:
                                return model.predict(arr)
                        lime_exp = LimeTabularExplainer(
                            X.values,
                            feature_names=final_features,
                            mode="regression",
                            discretize_continuous=True,
                        )
                        idx = 0
                        exp = lime_exp.explain_instance(X.values[idx], _predict_fn, num_features=min(10, len(final_features)))
                        contrib = pd.DataFrame(exp.as_list(), columns=["feature", "weight"]).sort_values("weight")
                        contrib.to_csv("reports/explainability/lime_explanation.csv", index=False)
                        fig_lime = px.bar(contrib, x="weight", y="feature", orientation="h", title="LIME local explanation")
                        st.plotly_chart(fig_lime, use_container_width=True)
        except Exception as e:
            st.error(f"Explainability failed: {e}")
    else:
        st.caption("Click 'Compute explanations' to load SHAP & LIME (heavy). Other tabs load at start.")
with tab1:
    st.write("#### AQI Trend Over Time with Health Categories")


    df["AQI_7D"] = df["aqi_index"].rolling(7, min_periods=1).mean()
    df["AQI_30D"] = df["aqi_index"].rolling(30, min_periods=1).mean()

    fig1 = go.Figure()


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
    fig1.add_trace(go.Scatter(
        x=df["date"],
        y=df["aqi_index"],
        mode="lines+markers",
        name="Daily AQI",
        line=dict(color="royalblue", width=2),
        marker=dict(size=4)
    ))


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
        yaxis_title="AQI Index (0-300)",
        template="plotly_white",
        hovermode="x unified",
        height=450
    )

    st.plotly_chart(fig1, use_container_width=True)
with tab2:
    st.write("#### AQI Distribution by Month")
    df["month"] = df["date"].dt.month_name()

    fig2 = px.box(df, x="month", y="aqi_index", color="month",
                  title="AQI Distribution by Month")
    fig2.update_layout(xaxis_title="Month", yaxis_title="AQI Index", showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)
with tab3:
    st.write("#### Correlation Heatmap of Pollutants & AQI")
    numeric = df.select_dtypes(include='number')
    corr = numeric.corr().round(2)

    fig3 = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                     title="Correlation Heatmap")
    st.plotly_chart(fig3, use_container_width=True)
with tab4:
    st.write("#### Average Pollutant Concentrations (ug/m^3)")
    pollutants = ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]

    existing = [p for p in pollutants if p in df.columns]

    if existing:
        avg_vals = df[existing].mean().reset_index()
        avg_vals.columns = ["Pollutant", "Avg_Concentration"]

        fig4 = px.bar(avg_vals, x="Pollutant", y="Avg_Concentration",
                      color="Pollutant", title="Average Pollutant Levels")
        fig4.update_layout(yaxis_title="ug/m^3")
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.warning("WARNING Pollutant columns missing!")
st.subheader("AQI Forecast (Next 3 Days)")
try:
    resp = requests.get("http://127.0.0.1:8000/get_forecast")
    data = resp.json()

    if data["status"] == "success":
        forecast_df = pd.DataFrame(data["forecasts"])
        st.dataframe(forecast_df)


        highest = forecast_df["predicted_aqi"].max()
        st.warning(f"Highest forecasted AQI: {highest}")
    else:
        st.error("Backend error: " + data.get("message", "unknown"))
except:
    st.error("ERROR Cannot connect to backend for forecast.")