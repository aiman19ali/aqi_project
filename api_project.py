import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import requests
import time
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None  # type: ignore


# Load environment variables from .env if present (non-fatal)
try:
    if load_dotenv:
        # Prefer loading the project's .env by absolute path to avoid cwd issues
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        # If the file exists, load it; otherwise call default loader which will search CWD
        if os.path.exists(env_path):
            load_dotenv(env_path)
        else:
            load_dotenv()
    else:
        print("⚠️  python-dotenv not available. Environment variables from .env will not be loaded. Install with: pip install python-dotenv")
except Exception as e:
    print(f"⚠️  Failed to load .env via python-dotenv: {e}")


# Lazy optional hopsworks import — don't hard-fail if sdk missing
try:
    import hopsworks
except Exception as e:
    hopsworks = None
    print(
        "⚠️  hopsworks import failed: {}\n".format(e)
        + "To enable feature-store uploads install the SDK: pip install hopsworks\n"
        + "If you already installed it, ensure your Python interpreter in your editor matches the one where you installed packages."
    )


def to_unix_seconds(dt: datetime) -> int:
    """Convert a datetime to UNIX seconds; coerce to UTC if naive."""
    return int(dt.timestamp()) if dt.tzinfo else int(dt.replace(tzinfo=timezone.utc).timestamp())


def http_get_json(url: str) -> Dict:
    """Perform HTTP GET and return JSON, raising on non-200 or malformed JSON."""
    try:
        r = requests.get(url, timeout=60)  # Increased timeout to 60 seconds
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, dict):
            raise RuntimeError("Unexpected non-object JSON response")
        return data
    except requests.exceptions.Timeout:
        print(f"⚠️  Timeout error for URL: {url[:100]}...")
        raise RuntimeError("API request timed out. Try again later or reduce data range.")
    except requests.exceptions.ConnectionError:
        print(f"⚠️  Connection error for URL: {url[:100]}...")
        raise RuntimeError("Connection failed. Check your internet connection.")


def fetch_history_chunked(lat: float, lon: float, api_key: str, start_dt: datetime, end_dt: datetime, *, chunk_days: int = 5) -> List[Dict]:
    """Fetch historical AQI records between two datetimes in chunked windows."""
    if start_dt >= end_dt:
        raise ValueError("start_dt must be earlier than end_dt")
    records: List[Dict] = []
    cursor = start_dt
    while cursor < end_dt:
        chunk_end = min(cursor + timedelta(days=chunk_days), end_dt)
        url = (
            "http://api.openweathermap.org/data/2.5/air_pollution/history"
            f"?lat={lat}&lon={lon}&start={to_unix_seconds(cursor)}&end={to_unix_seconds(chunk_end)}&appid={api_key}"
        )
        payload = http_get_json(url)
        chunk = payload.get("list")
        if not isinstance(chunk, list):
            raise RuntimeError("History response invalid")
        records.extend(chunk)
        cursor = chunk_end
        time.sleep(1)  # Add 1 second delay between API calls
    records.sort(key=lambda r: r.get("dt", 0))
    return records


def fetch_weather_history_openmeteo(lat: float, lon: float, start_dt: datetime, end_dt: datetime) -> List[Dict]:
    """Fetch historical weather records using free Open-Meteo API."""
    print("Fetching historical weather data from Open-Meteo (free API)...")
    
    # Convert dates to required format
    start_date = start_dt.strftime("%Y-%m-%d")
    end_date = end_dt.strftime("%Y-%m-%d")
    
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily=temperature_2m_mean,relative_humidity_2m_mean,pressure_msl_mean,wind_speed_10m_mean"
    )
    
    try:
        payload = http_get_json(url)
        return [payload] if payload else []
    except Exception as e:
        print(f"Warning: Could not fetch weather data from Open-Meteo: {e}")
        return []


def fetch_weather_forecast_openmeteo(lat: float, lon: float) -> Dict:
    """Fetch weather forecast data using free Open-Meteo API."""
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_mean,relative_humidity_2m_mean,pressure_msl_mean,wind_speed_10m_mean"
        f"&forecast_days=5"  # Get 5 days to ensure we have 3 days
    )
    return http_get_json(url)


def fetch_forecast(lat: float, lon: float, api_key: str) -> List[Dict]:
    """Fetch forecast records as returned by OpenWeather."""
    url = (
        "http://api.openweathermap.org/data/2.5/air_pollution/forecast"
        f"?lat={lat}&lon={lon}&appid={api_key}"
    )
    payload = http_get_json(url)
    lst = payload.get("list")
    if not isinstance(lst, list):
        raise RuntimeError("Forecast response invalid")
    return lst


def normalize_records_with_openmeteo_weather(aqi_records: List[Dict], weather_data: Dict) -> pd.DataFrame:
    """Normalize raw AQI records and Open-Meteo weather data into a tidy DataFrame."""
    rows: List[Dict] = []
    
    # Process AQI records
    aqi_data = {}
    for rec in aqi_records:
        ts = rec.get("dt")
        if ts is None:
            continue
        main = rec.get("main", {})
        comps = rec.get("components", {})
        aqi_data[ts] = {
            "aqi_index": main.get("aqi"),
            **comps
        }
    
    # Process Open-Meteo weather data
    weather_daily = {}
    if weather_data and "daily" in weather_data:
        daily = weather_data["daily"]
        dates = daily.get("time", [])
        temps = daily.get("temperature_2m_mean", [])
        humidity = daily.get("relative_humidity_2m_mean", [])
        pressure = daily.get("pressure_msl_mean", [])
        wind_speed = daily.get("wind_speed_10m_mean", [])
        
        for i, date_str in enumerate(dates):
            # Convert date to timestamp for matching with AQI data
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            ts = int(date_obj.timestamp())
            weather_daily[ts] = {
                "temperature": temps[i] if i < len(temps) else None,
                "humidity": humidity[i] if i < len(humidity) else None,
                "pressure": pressure[i] if i < len(pressure) else None,
                "wind_speed": wind_speed[i] if i < len(wind_speed) else None,
                "wind_deg": None,  # Not available in basic Open-Meteo
                "clouds": None,    # Not available in basic Open-Meteo
                "uvi": None        # Not available in Open-Meteo free tier
            }
    
    # Combine AQI and weather data
    all_timestamps = set(aqi_data.keys()) | set(weather_daily.keys())
    for ts in all_timestamps:
        row: Dict = {
            "timestamp": datetime.fromtimestamp(int(ts), tz=timezone.utc),
        }
        
        # Add AQI data if available
        if ts in aqi_data:
            row.update(aqi_data[ts])
        else:
            row.update({
                "aqi_index": None,
                "co": None, "no": None, "no2": None, "o3": None,
                "so2": None, "pm2_5": None, "pm10": None, "nh3": None
            })
        
        # Add weather data if available
        if ts in weather_daily:
            row.update(weather_daily[ts])
        else:
            row.update({
                "temperature": None, "humidity": None, "pressure": None,
                "wind_speed": None, "wind_deg": None, "clouds": None, "uvi": None
            })
        
        rows.append(row)
    
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values("timestamp").reset_index(drop=True)


def compute_daily_means(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate numeric columns to daily means by UTC date."""
    if df.empty:
        return df
    tmp = df.copy()
    tmp["date"] = tmp["timestamp"].dt.tz_convert(timezone.utc).dt.date
    numeric_cols = [c for c in tmp.columns if c not in ["timestamp", "date"]]
    out = tmp.groupby("date")[numeric_cols].mean(numeric_only=True).reset_index()
    return out.sort_values("date").reset_index(drop=True)


def fill_missing_weather_data(forecast_daily: pd.DataFrame, history_daily: pd.DataFrame) -> pd.DataFrame:
    """Fill missing weather data using interpolation and historical patterns."""
    forecast_daily = forecast_daily.copy()
    
    weather_cols = ['temperature', 'humidity', 'pressure', 'wind_speed']
    
    # Check if we have missing weather data
    missing_weather = forecast_daily[weather_cols].isnull().any(axis=1)
    
    if missing_weather.any():
        print("Filling missing weather data using interpolation and historical patterns...")
        
        # Method 1: Forward fill from available data
        for col in weather_cols:
            if forecast_daily[col].isnull().any():
                forecast_daily[col] = forecast_daily[col].fillna(method='ffill')
        
        # Method 2: If still missing, use historical averages for same month/day
        for col in weather_cols:
            if forecast_daily[col].isnull().any():
                # Get historical averages for similar dates
                forecast_daily['month'] = pd.to_datetime(forecast_daily['date']).dt.month
                forecast_daily['day'] = pd.to_datetime(forecast_daily['date']).dt.day
                
                historical_avg = history_daily.groupby(['month', 'day'])[col].mean()
                
                for idx, row in forecast_daily.iterrows():
                    if pd.isnull(row[col]):
                        month_day = (row['month'], row['day'])
                        if month_day in historical_avg.index:
                            forecast_daily.at[idx, col] = historical_avg[month_day]
                        else:
                            # Use overall historical average
                            forecast_daily.at[idx, col] = history_daily[col].mean()
        
        # Method 3: If still missing, use simple interpolation
        for col in weather_cols:
            if forecast_daily[col].isnull().any():
                forecast_daily[col] = forecast_daily[col].interpolate()
        
        # Clean up temporary columns
        forecast_daily = forecast_daily.drop(columns=['month', 'day'], errors='ignore')
    
    return forecast_daily


def select_next_three_days(forecast_daily: pd.DataFrame) -> pd.DataFrame:
    """Return the next three future days from a daily forecast frame."""
    if forecast_daily.empty:
        return forecast_daily
    today = datetime.now(timezone.utc).date()
    return forecast_daily[forecast_daily["date"] > today].head(3).reset_index(drop=True)


def save_pollutant_features_to_hopsworks(df: pd.DataFrame, feature_group_name: str = "aqi_pollutants", version: int = 1) -> None:
    """Save daily pollutant features (and aqi_index) to Hopsworks Feature Store.
    This is intentionally non-fatal: if hopsworks or credentials are missing the
    program will continue but will print helpful diagnostics.
    It tries multiple environment variable names to match your `.env`:
      - HOPSWORKS_API_KEY or hopswork_api
      - HOPSWORKS_PROJECT or hopsworks_project
      - HOPSWORKS_HOST (optional)
    """
    if df is None or df.empty:
        print("⚠️  No data to save to Hopsworks (empty dataframe).")
        return

    # Ensure pollutant columns exist
    cols = ["date", "aqi_index", "pm2_5", "pm10", "co", "no2", "o3", "so2", "nh3"]
    fg_df = df.copy()
    for c in cols:
        if c not in fg_df.columns:
            fg_df[c] = np.nan
    fg_df = fg_df[cols].copy()

    # Normalize date to datetime if possible
    try:
        fg_df["date"] = pd.to_datetime(fg_df["date"])
    except Exception:
        # fallback: keep as-is
        pass

    if hopsworks is None:
        print("⚠️  hopsworks Python package not installed. Install with: pip install hopsworks")
        return

    # Read credentials from environment — support multiple naming conventions
    api_key = os.environ.get("HOPSWORKS_API_KEY") or os.environ.get("hopswork_api") or os.environ.get("HOPSWORKS_APIKEY")
    project_name = os.environ.get("HOPSWORKS_PROJECT") or os.environ.get("hopsworks_project")
    host = os.environ.get("HOPSWORKS_HOST") or os.environ.get("hopsworks_host")

    if not api_key:
        print("⚠️  Hopsworks API key not found in environment (HOPSWORKS_API_KEY or hopswork_api). Skipping upload.")
        return

    try:
        # Attempt login — different client versions accept different args.
        try:
            # Prefer explicit api_key and optional host/project
            login_kwargs = {"api_key_value": api_key}
            if host:
                login_kwargs["host"] = host
            if project_name:
                login_kwargs["project"] = project_name
            project = hopsworks.login(**login_kwargs)
        except TypeError:
            # Fallback to simple call and hope env vars are used by the sdk
            project = hopsworks.login()

        fs = project.get_feature_store()

        # Create or get feature group (primary key = date)
        fg = fs.get_or_create_feature_group(name=feature_group_name,
                                           version=version,
                                           primary_key=["date"],
                                           description="Daily AQI pollutant & index features")
        # Insert data (append). Use write_options to avoid waiting long in blocking mode.
        fg.insert(fg_df, write_options={"wait_for_job": False})
        print(f"✅ Saved {len(fg_df)} pollutant records to Hopsworks feature group '{feature_group_name}:{version}'")
    except Exception as e:
        print(f"⚠️  Failed to save to Hopsworks Feature Store: {e}")


def main() -> None:
    """
    Fetch 1-year historical AQI data and predict next 3 days:
    - Fetch 1-year historical data from OpenWeather API (chunked to respect limits)
    - Get 3-day forecast from OpenWeather API
    - Process data into daily averages
    - Save predictions and display summary
    """
    # Inputs (env vars are optional; fall back to defaults)
    # Support multiple env var names and strip surrounding quotes if present
    _raw_api = (
        os.environ.get("OPENWEATHER_API_KEY")
        or os.environ.get("openweatherAPI_KEY")
        or os.environ.get("API_KEY")
        or os.environ.get("openweatherAPIKEY")
    )
    if _raw_api:
        api_key = str(_raw_api).strip().strip('"').strip("'")
    else:
        api_key = "93bd3884eb00e090c345413e03d6c01a"
    lat = float(os.environ.get("AQI_LAT", "24.8607"))
    lon = float(os.environ.get("AQI_LON", "67.0011"))

    # Output files
    history_hourly_csv = os.path.join("data", "history_hourly.csv")
    history_daily_csv = os.path.join("data", "history_daily_1y.csv")
    forecast_daily_csv = os.path.join("data", "forecast_daily_next3.csv")
    os.makedirs("data", exist_ok=True)

    # Build 1-year window
    now_utc = datetime.now(timezone.utc)
    year_start = now_utc - timedelta(days=365)

    # Fetch 1 year of hourly history using chunked approach
    print("Fetching 1-year historical AQI data...")
    try:
        history_aqi_records = fetch_history_chunked(lat, lon, api_key, year_start, now_utc, chunk_days=5)
    except Exception as e:
        print(f"⚠️  Failed to fetch 1-year data: {e}")
        print("🔄 Trying with 3-month data instead...")
        three_months_start = now_utc - timedelta(days=90)
        history_aqi_records = fetch_history_chunked(lat, lon, api_key, three_months_start, now_utc, chunk_days=5)
    
    print("Fetching 1-year historical weather data from Open-Meteo...")
    history_weather_data = fetch_weather_history_openmeteo(lat, lon, year_start, now_utc)
    
    history_df = normalize_records_with_openmeteo_weather(history_aqi_records, history_weather_data[0] if history_weather_data else {})
    if history_df.empty:
        raise RuntimeError("No historical AQI data for last 1 year")
    history_df.to_csv(history_hourly_csv, index=False)

    # Compute daily means for the full year
    history_daily = compute_daily_means(history_df)
    history_daily.to_csv(history_daily_csv, index=False)

    # Fetch forecast for next 3 days
    print("Fetching 3-day AQI forecast...")
    forecast_aqi_records = fetch_forecast(lat, lon, api_key)
    
    print("Fetching 3-day weather forecast from Open-Meteo...")
    forecast_weather_data = fetch_weather_forecast_openmeteo(lat, lon)
    
    forecast_df = normalize_records_with_openmeteo_weather(forecast_aqi_records, forecast_weather_data)
    if forecast_df.empty:
        raise RuntimeError("No forecast AQI data returned")
    forecast_daily = compute_daily_means(forecast_df)
    
    # Fill missing weather data using interpolation or historical patterns
    forecast_daily = fill_missing_weather_data(forecast_daily, history_daily)
    
    next3 = select_next_three_days(forecast_daily)
    if next3.empty:
        raise RuntimeError("Forecast missing next 3 days")
    next3.to_csv(forecast_daily_csv, index=False)

    # Summary
    print(f"✅ Data Collection Complete!")
    print(f"📊 Historical Data: {len(history_daily)} days from {history_daily['date'].min()} to {history_daily['date'].max()}")
    print(f"🔮 Forecast Data: {len(next3)} days")
    print()
    
    print("📈 Historical AQI Summary (last 1 year):")
    print(f"   Average AQI: {history_daily['aqi_index'].mean():.2f}")
    print(f"   Min AQI: {history_daily['aqi_index'].min():.2f}")
    print(f"   Max AQI: {history_daily['aqi_index'].max():.2f}")
    print()
    
    print("📅 Recent History (last 5 days) - AQI & Weather:")
    display_cols = ["date", "aqi_index", "temperature", "humidity", "wind_speed", "pressure"]
    available_cols = [col for col in display_cols if col in history_daily.columns]
    print(history_daily[available_cols].tail(5).to_string(index=False))
    print()
    
    print("📊 Recent History (last 5 days) - Pollutant Components:")
    pollutant_cols = ["date", "aqi_index", "pm2_5", "pm10", "co", "no2", "o3", "so2"]
    available_pollutant_cols = [col for col in pollutant_cols if col in history_daily.columns]
    print(history_daily[available_pollutant_cols].tail(5).to_string(index=False))
    print()
    
    print("🔮 Forecast (next 3 days) - AQI & Weather:")
    available_cols = [col for col in display_cols if col in next3.columns]
    print(next3[available_cols].to_string(index=False))
    print()
    
    print("🔮 Forecast (next 3 days) - Pollutant Components:")
    available_pollutant_cols = [col for col in pollutant_cols if col in next3.columns]
    print(next3[available_pollutant_cols].to_string(index=False))
    print()
    print(f"💾 Data saved to: {history_hourly_csv}, {history_daily_csv}, {forecast_daily_csv}")


if __name__ == "__main__":
    main()


