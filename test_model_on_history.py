import joblib, pandas as pd, os, numpy as np
MODEL_PATH = "models/aqi_model.pkl"
DATA_PATH = "data/cleaned_data.csv"

print("MODEL_PATH:", MODEL_PATH)
print("DATA_PATH:", DATA_PATH)

if not os.path.exists(MODEL_PATH):
    raise SystemExit("Model file not found at models/aqi_model.pkl")
if not os.path.exists(DATA_PATH):
    raise SystemExit("History file not found at data/cleaned_data.csv")
model = joblib.load(MODEL_PATH)
print("Loaded model type:", type(model))

df = pd.read_csv(DATA_PATH)
print("History rows:", len(df))
sample = df.drop(columns=[c for c in ["date","aqi_index"] if c in df.columns]).select_dtypes(include=[np.number]).tail(1)
true_aqi = df["aqi_index"].tail(1).values[0]

print("Sample feature columns (tail 1):")
print(sample.columns.tolist())
print("Sample values:")
print(sample.iloc[0].to_dict())

try:
    pred = model.predict(sample)[0]
    print("Model raw prediction:", pred)
    print("True aqi:", true_aqi)
    print("Prediction / True ratio:", pred/true_aqi if true_aqi!=0 else None)
except Exception as e:
    print("Error calling model.predict():", e)
