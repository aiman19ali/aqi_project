import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load cleaned data
df = pd.read_csv("data/cleaned_data.csv")

# Features and target
X = df.drop(columns=["date", "aqi_index"])  # Drop target and date
y = df["aqi_index"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "models/aqi_model.pkl")
print("Model trained and saved as models/aqi_model.pkl")
