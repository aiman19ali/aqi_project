# AQI Project

Air Quality Index (AQI) forecasting platform that combines data ingestion, model training, explainability, and interactive visualization. The repository contains a Streamlit dashboard for exploration, a FastAPI backend for serving predictions, and scheduled GitHub Actions workflows to keep features and models up to date.

## Features
- **Data exploration**: Streamlit dashboard with time-series trends, pollutant analysis, and correlation heatmaps.
- **Forecasting models**: Gradient boosted models predict AQI 24, 48, and 72 hours ahead with stored artefacts under `models/`.
- **Explainability**: SHAP and LIME integrations surface feature importance and local attributions.
- **API backend**: FastAPI service exposes `/predict`, `/get_forecast`, and `/live_aqi` endpoints.
- **Automation pipelines**: GitHub Actions trigger feature ingestion hourly and model retraining daily using the shared `requirements.txt`.

## Repository Layout
- `dashboard.py` – Streamlit UI entry point.
- `backend.py` – FastAPI application for live predictions.
- `trainall_models.py` – Batch trainer that refreshes all forecast horizons.
- `predict_aqi.py` – CLI to generate the next 3 day AQI forecast.
- `feature_pipeline.py` – Feature ingestion script referenced by CI.
- `model_registry.py` – Model registration workflow executed by CI.
- `data/` – Source, cleaned, and derived datasets.
- `models/` – Saved estimators, scalers, and metadata.
- `reports/` – Generated EDA charts and explainability artefacts.
- `.github/workflows/` – GitHub Actions definitions.

## Requirements
- Python 3.10 (aligns with GitHub Actions runners)
- Dependencies listed in `requirements.txt`

Install locally:
```bash
python -m venv .venv
.venv\Scripts\activate      # on Windows
source .venv/bin/activate   # on macOS/Linux
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Local Development

### 1. Prepare Data
Ensure `data/cleaned_data.csv` and `data/clean_aqi.csv` exist. Use `feature_pipeline.py` or provide equivalent CSVs.

### 2. Train Models
```bash
python trainall_models.py
```
Produces `models/xgb_aqi_24h.joblib`, `models/xgb_aqi_48h.joblib`, and `models/xgb_aqi_72h.joblib` with accompanying metrics.

### 3. Serve the API
```bash
uvicorn backend:app --reload
```
Endpoints:
- `GET /` – health check
- `POST /predict` – AQI prediction from feature payload (scaled to 0–500)
- `GET /get_forecast` – next 3-day forecast using latest data row
- `GET /live_aqi` – fetch live AQI (falls back to synthetic data on failure)

### 4. Launch the Dashboard
```bash
streamlit run dashboard.py
```
Open the Explainability tab to compute SHAP/LIME insights and review model metrics.

### 5. Command-Line Forecast
```bash
python predict_aqi.py
```
Writes `data/predicted_aqi_next3days.csv` and prints scaled forecasts.

## CI/CD Workflows
- `feature_ingestion.yml` – Hourly job that installs `requirements.txt`, runs the feature pipeline, and prepares fresh datasets.
- `ml_pipeline.yml` – Daily job that installs `requirements.txt` and executes the model registry pipeline.

Secrets required:
- `HOPSWORKS_API_KEY` – used by both workflows for feature store access.

## Testing and Quality
- Use `python -m pytest` for any unit tests you add.
- Lint with `flake8` or `ruff` if introduced to the toolchain.
- Inspect Streamlit warnings (`use_container_width` deprecations) and adjust chart sizing (`width="stretch"` where required).

## Contribution Guidelines
- Prefer functional, declarative Python with descriptive variable names.
- Ensure new functions include docstrings/block comments describing behaviour.
- Update the README when workflows, scripts, or dependencies change.


