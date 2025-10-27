# train_all_models.py
# This file trains 3 models to predict Air Quality Index (AQI)
# for the next 24, 48, and 72 hours.

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import numpy as np
from xgboost import XGBRegressor

# make a folder to save our models
os.makedirs("models", exist_ok=True)

print("Loading cleaned data...")
df = pd.read_csv("data/clean_aqi.csv")

# Create target columns for future predictions (shift AQI forward)
df['target_24h'] = df['aqi_index'].shift(-1)  # AQI 1 day later
df['target_48h'] = df['aqi_index'].shift(-2)  # AQI 2 days later  
df['target_72h'] = df['aqi_index'].shift(-3)  # AQI 3 days later

# Remove rows with missing targets (last 3 days)
df = df.dropna(subset=['target_24h', 'target_48h', 'target_72h'])

# remove columns we don't need for training
target_columns = ['target_24h', 'target_48h', 'target_72h']
columns_to_drop = target_columns + ['aqi_index', 'date']
X = df.drop(columns=columns_to_drop)

# Ensure all remaining columns are numeric
X = X.select_dtypes(include=[np.number])

print(f"Training with {X.shape[1]} features: {list(X.columns)}")
print(f"Training data shape: {X.shape}")

# define which targets we want to predict
targets = {
    'target_24h': "models/model_24h.pkl",
    'target_48h': "models/model_48h.pkl",
    'target_72h': "models/model_72h.pkl"
}

print("Training models with XGBoost (Gradient Boosting)...")
for target, model_path in targets.items():
    print(f"\nNow training for: {target}")

    y = df[target]

    # split data into train and test parts (time-ordered)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Use TimeSeriesSplit cross-validation on the training set to pick best params robustly
    # Only use TimeSeriesSplit when there's enough training data; otherwise fall back to single validation slice
    n_train = len(X_train)
    use_tscv = n_train >= 20

    param_grid = [
        {"max_depth": 4, "learning_rate": 0.1},
        {"max_depth": 6, "learning_rate": 0.1},
        {"max_depth": 6, "learning_rate": 0.01},
        {"max_depth": 8, "learning_rate": 0.05},
    ]

    best_params = None
    best_cv_mean = float("inf")
    best_cv_std = None

    if use_tscv:
        n_splits = 5
        tscv = TimeSeriesSplit(n_splits=n_splits)
        print(f" Using TimeSeriesSplit with {n_splits} splits for CV (n_train={n_train})")

        for params in param_grid:
            fold_rmses = []
            print(f"  CV try params: max_depth={params['max_depth']}, lr={params['learning_rate']}")
            for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
                X_tr_fold = X_train.iloc[tr_idx]
                y_tr_fold = y_train.iloc[tr_idx]
                X_val_fold = X_train.iloc[val_idx]
                y_val_fold = y_train.iloc[val_idx]

                try:
                    m = XGBRegressor(
                        n_estimators=1000,
                        max_depth=params["max_depth"],
                        learning_rate=params["learning_rate"],
                        random_state=42,
                        n_jobs=-1,
                        objective="reg:squarederror",
                    )
                    m.fit(
                        X_tr_fold,
                        y_tr_fold,
                        eval_set=[(X_val_fold, y_val_fold)],
                        eval_metric="rmse",
                        early_stopping_rounds=20,
                        verbose=False,
                    )
                    val_preds = m.predict(X_val_fold)
                    val_rmse = float(np.sqrt(mean_squared_error(y_val_fold, val_preds)))
                    fold_rmses.append(val_rmse)
                    print(f"    fold {fold+1}/{n_splits} val RMSE: {val_rmse:.4f}")
                except Exception as e:
                    print(f"    fold {fold+1} error: {e}")

            if fold_rmses:
                mean_rmse = float(np.mean(fold_rmses))
                std_rmse = float(np.std(fold_rmses))
                print(f"   -> mean RMSE: {mean_rmse:.4f} (std {std_rmse:.4f})")
                if mean_rmse < best_cv_mean:
                    best_cv_mean = mean_rmse
                    best_cv_std = std_rmse
                    best_params = params
    else:
        # Fallback: single validation slice from the end of training (time-ordered)
        val_size = max(1, int(0.1 * len(X_train)))
        X_tr = X_train.iloc[:-val_size]
        y_tr = y_train.iloc[:-val_size]
        X_val = X_train.iloc[-val_size:]
        y_val = y_train.iloc[-val_size:]
        print(f" Using single validation slice: train={len(X_tr)}, val={len(X_val)} (n_train={n_train})")

        for params in param_grid:
            print(f"  Try params: max_depth={params['max_depth']}, lr={params['learning_rate']}")
            try:
                m = XGBRegressor(
                    n_estimators=1000,
                    max_depth=params["max_depth"],
                    learning_rate=params["learning_rate"],
                    random_state=42,
                    n_jobs=-1,
                    objective="reg:squarederror",
                )
                m.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_val, y_val)],
                    eval_metric="rmse",
                    early_stopping_rounds=20,
                    verbose=False,
                )
                val_preds = m.predict(X_val)
                mean_rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))
                print(f"   -> val RMSE: {mean_rmse:.4f}")
                if mean_rmse < best_cv_mean:
                    best_cv_mean = mean_rmse
                    best_params = params
            except Exception as e:
                print(f"   Skipped params {params} due to error: {e}")

    if best_params is None:
        print("No parameter set produced a successful CV result for this target. Skipping.")
        continue

    print(f" Selected best params: {best_params} with CV mean RMSE={best_cv_mean:.4f}")

    # Retrain on full training set using a small validation slice for early stopping
    val_size = max(1, int(0.1 * len(X_train))) if len(X_train) >= 5 else 0
    if val_size > 0:
        X_tr_full = X_train.iloc[:-val_size]
        y_tr_full = y_train.iloc[:-val_size]
        X_val_full = X_train.iloc[-val_size:]
        y_val_full = y_train.iloc[-val_size:]
        eval_set_full = [(X_val_full, y_val_full)]
    else:
        X_tr_full = X_train
        y_tr_full = y_train
        eval_set_full = None

    best_model = XGBRegressor(
        n_estimators=1000,
        max_depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
    )
    if eval_set_full:
        best_model.fit(X_tr_full, y_tr_full, eval_set=eval_set_full, eval_metric="rmse", early_stopping_rounds=20, verbose=False)
    else:
        best_model.fit(X_tr_full, y_tr_full)

    # Evaluate on the held-out test set
    preds = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"Results for {target}:")
    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R²   : {r2:.3f}")

    # save the model
    joblib.dump(best_model, model_path)
    print(f"Model saved at: {model_path}")

    # save metrics and chosen params to CSV (append mode)
    # try to get best iteration from the trained booster (if early stopping was used)
    try:
        best_iter = getattr(best_model, "best_iteration", None)
        if best_iter is None:
            best_iter = best_model.get_booster().best_iteration
        if best_iter is not None:
            best_iter = int(best_iter)
    except Exception:
        best_iter = None

    metrics_row = {
        "target": target,
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "best_max_depth": int(best_params["max_depth"]),
        "best_learning_rate": float(best_params["learning_rate"]),
        "cv_mean_rmse": float(best_cv_mean) if best_cv_mean is not None else None,
        "cv_std_rmse": float(best_cv_std) if best_cv_std is not None else None,
        "best_iteration": int(best_iter) if best_iter is not None else None,
    }
    metrics_csv = os.path.join("models", "model_metrics.csv")
    # create file with header if missing
    write_header = not os.path.exists(metrics_csv)
    metrics_df = pd.DataFrame([metrics_row])
    metrics_df.to_csv(metrics_csv, mode="a", header=write_header, index=False)

print("\nAll models are trained and saved successfully!")
