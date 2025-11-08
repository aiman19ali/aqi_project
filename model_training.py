import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib


df = pd.read_csv("data/cleaned_data.csv")


X = df.drop(columns=["date", "aqi_index"])                        
y = df["aqi_index"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


joblib.dump(model, "models/aqi_model.pkl")
print("Model trained and saved as models/aqi_model.pkl")
