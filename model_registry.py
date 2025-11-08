                   
import os
import hopsworks
from dotenv import load_dotenv
import joblib
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

def main():
    """Load feature data, train model, and register artifacts in Hopsworks."""
    load_dotenv()

    api_key = os.environ.get("HOPSWORKS_API_KEY") or os.environ.get("hopswork_api")
    if not api_key:
        print("WARNING Hopsworks API key not found in environment variables.")
        print("Please register at https://c.app.hopsworks.ai/account/api/generated")
        print("Then set your API key in environment variables as HOPSWORKS_API_KEY")
        return
    project_name = os.environ.get("HOPSWORKS_PROJECT")
    if not project_name:
        print("WARNING Hopsworks project name not found in environment variables.")
        print("Set HOPSWORKS_PROJECT to your project name from the Hopsworks UI.")
        return
    feature_group_version = int(os.environ.get("HOPSWORKS_FG_VERSION", "3"))
    feature_view_version = int(os.environ.get("HOPSWORKS_FV_VERSION", "3"))

    project = hopsworks.login(project=project_name, api_key_value=api_key)
    fs = project.get_feature_store()


    try:

        import pandas as pd
        try:
            aqi_data = pd.read_csv('data/clean_aqi.csv')
            print("Successfully loaded AQI data")
        except Exception as e:
            print(f"Error reading AQI data: {str(e)}")
            return
        aqi_data['date'] = pd.to_datetime(aqi_data['date'])


        train_data = aqi_data.copy()
        train_data['year'] = aqi_data['date'].dt.year
        train_data['month'] = aqi_data['date'].dt.month
        train_data['day'] = aqi_data['date'].dt.day
        train_data['dayofweek'] = aqi_data['date'].dt.dayofweek


        train_data['date'] = train_data['date'].dt.strftime('%Y-%m-%d')


        fg = fs.get_or_create_feature_group(
            name="aqi_pollutants",
            version=feature_group_version,
            primary_key=['date'],                                                     
            description="Air Quality Index and pollutant measurements with temporal features"
        )


        fg.insert(train_data, write_options={"wait_for_job": True})
        print("Successfully created/updated feature group 'aqi_pollutants'")



        try:
            old_view = fs.get_feature_view(name="aqi_prediction_view", version=feature_view_version)
            if old_view is not None:
                old_view.clean()
                print("Cleaned up old feature view")
        except Exception as stale_error:
            old_view = None
        query = fg.select_all()
        try:
            feature_view = fs.get_or_create_feature_view(
                name="aqi_prediction_view",
                version=feature_view_version,
                description="Feature view for AQI prediction model",
                query=query,
                labels=['aqi_index']                               
            )
        except Exception:
            if old_view is not None:
                try:
                    old_view.delete()
                except Exception:
                    pass
            feature_view = fs.create_feature_view(
                name="aqi_prediction_view",
                version=feature_view_version,
                description="Feature view for AQI prediction model",
                query=query,
                labels=['aqi_index']
            )
    except Exception as e:
        print(f"Error accessing feature store: {e}")
        return
    split_datasets = feature_view.train_test_split(test_size=0.2)
    if len(split_datasets) == 2:
        X_train, X_test = split_datasets
        y_train = None
        y_test = None
    elif len(split_datasets) == 4:
        X_train, X_test, y_train, y_test = split_datasets
    else:
        raise ValueError(f"Unexpected number of datasets returned from train_test_split: {len(split_datasets)}")
    X_train = X_train.compute() if hasattr(X_train, "compute") else X_train
    X_test = X_test.compute() if hasattr(X_test, "compute") else X_test


    if 'aqi_index' in X_train.columns:
        y_train = X_train.pop('aqi_index')
        y_test = X_test.pop('aqi_index')
    elif y_train is not None and y_test is not None:
        y_train = y_train.compute() if hasattr(y_train, 'compute') else y_train
        y_test = y_test.compute() if hasattr(y_test, 'compute') else y_test
    else:
        raise ValueError("Label column 'aqi_index' not found in feature view output.")
    if 'date' in X_train.columns:
        train_dates = X_train['date']
        test_dates = X_test['date']
        X_train = X_train.drop('date', axis=1)
        X_test = X_test.drop('date', axis=1)
    else:
        train_dates = None
        test_dates = None
    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)
    print("Features used for training:", X_train.columns.tolist())


    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print("Model MAE:", mae)


    model_file = "xgboost_model.pkl"
    joblib.dump(model, model_file)


    mr = project.get_model_registry()
    model_meta = mr.python.create_model(
        name="xgboost_weather_model",                           
        metrics={"mae": mae},                                      
        description="XGBoost trained on weather + AQI features"
    )

    model_meta.save(model_file)

    print("Model saved to Hopsworks Model Registry (name: xgboost_weather_model)")
    print("Version:", model_meta.version)
if __name__ == "__main__":
    main()
