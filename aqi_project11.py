import requests, os
import pandas as pd
from datetime import datetime

API_KEY = "93bd3884eb00e090c345413e03d6c01a"   # relace
lat = 24.8607
lon = 67.0011
OUT_CSV = "data/raw_aqi.csv"
os.makedirs("data", exist_ok=True)

url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
resp = requests.get(url)
resp.raise_for_status()
j = resp.json()

# adapt this based on actual JSON you saw in Postman
entry = j['list'][0]['components']    # co, no, no2, o3, so2, pm2_5, pm10, nh3
entry['aqi_index'] = j['list'][0]['main']['aqi']
entry['timestamp'] = datetime.utcnow().isoformat()

df = pd.DataFrame([entry])
if not os.path.exists(OUT_CSV):
    df.to_csv(OUT_CSV, index=False)
else:
    df.to_csv(OUT_CSV, mode='a', header=False, index=False)

print("Saved row to", OUT_CSV)
print(df.tail(1).to_dict(orient="records")[0])