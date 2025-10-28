# model_registry.py
import os
import hopsworks
import joblib
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

def main():
    # 1) Login to Hopsworks using API key from environment
    api_key = os.environ.get("HOPSWORKS_API_KEY") or os.environ.get("hopswork_api")
    if not api_key:
        print("⚠️ Hopsworks API key not found in environment variables.")
        print("Please register at https://c.app.hopsworks.ai/account/api/generated")
        print("Then set your API key in environment variables as HOPSWORKS_API_KEY")
        return

    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()

    # 2) Load or create feature group
    try:
        # First try to read the data
        import pandas as pd
        try:
            aqi_data = pd.read_csv('data/clean_aqi.csv')
            print("✅ Successfully loaded AQI data")
        except Exception as e:
            print(f"Error reading AQI data: {str(e)}")
            return

        # Delete existing feature group if it exists
        try:
            fg = fs.get_feature_group(name="aqi_pollutants", version=1)
            if fg is not None:
                fg.delete()
                print("Deleted existing feature group")
        except:
            pass  # Feature group doesn't exist yet

        # Preprocess data: Convert date to datetime and extract useful features while keeping original date
        aqi_data['date'] = pd.to_datetime(aqi_data['date'])
        
        # Keep date for time series features but add numerical features for training
        train_data = aqi_data.copy()
        train_data['year'] = aqi_data['date'].dt.year
        train_data['month'] = aqi_data['date'].dt.month
        train_data['day'] = aqi_data['date'].dt.day
        train_data['dayofweek'] = aqi_data['date'].dt.dayofweek
        
        # Convert date to string format for Hopsworks storage
        train_data['date'] = train_data['date'].dt.strftime('%Y-%m-%d')
        
        # Create or get feature group with both time and numerical features
        fg = fs.get_or_create_feature_group(
            name="aqi_pollutants",
            version=1,
            primary_key=['date'],  # Keep date as primary key for time series analysis
            description="Air Quality Index and pollutant measurements with temporal features"
        )
        
        # Upload the data to the feature group
        fg.insert(train_data, write_options={"wait_for_job": True})
        print("✅ Successfully created/updated feature group 'aqi_pollutants'")

        # Clean up old feature view if it exists
        try:
            old_view = fs.get_feature_view(name="aqi_prediction_view", version=1)
            if old_view is not None:
                old_view.clean()
                print("Cleaned up old feature view")
        except:
            pass  # Feature view doesn't exist yet

        # Create query and feature view
        query = fg.select_all()
        # Use get_or_create to avoid "already exists" errors when re-running the script
        feature_view = fs.get_or_create_feature_view(
            name="aqi_prediction_view",
            version=1,
            description="Feature view for AQI prediction model",
            query=query,
            labels=['aqi_index']  # Specify our target variable
        )
    except Exception as e:
        print(f"Error accessing feature store: {e}")
        return

    # Split the data and get it as pandas DataFrames
    X_train, X_test = feature_view.train_test_split(test_size=0.2)
    
    # Convert to pandas DataFrames
    X_train = X_train.compute()
    X_test = X_test.compute()
    
    # Separate features and target
    y_train = X_train.pop('aqi_index')
    y_test = X_test.pop('aqi_index')
    
    # Remove date column for training but keep it for reference
    train_dates = X_train['date']
    test_dates = X_test['date']
    X_train = X_train.drop('date', axis=1)
    X_test = X_test.drop('date', axis=1)
    
    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)
    print("Features used for training:", X_train.columns.tolist())

    # 3) Train model
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)

    # 4) Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print("Model MAE:", mae)

    # 5) Save local artifact
    model_file = "xgboost_model.pkl"
    joblib.dump(model, model_file)

    # 6) Register model in Hopsworks Model Registry
    mr = project.get_model_registry()
    model_meta = mr.python.create_model(
        name="xgboost_weather_model",            # friendly name
        metrics={"mae": mae},                    # record metric(s)
        description="XGBoost trained on weather + AQI features"
    )
    # Upload file to registry (creates a new version if name exists)
    model_meta.save(model_file)

    print("✅ Model saved to Hopsworks Model Registry (name:", "xgboost_weather_model )")
    print("Version:", model_meta.version)

if __name__ == "__main__":
    main()
