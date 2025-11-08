import importlib
import json
import sys

modules = [
    'fastapi', 'pydantic', 'uvicorn', 'streamlit', 'pandas', 'numpy', 'plotly',
    'requests', 'sklearn', 'joblib', 'matplotlib', 'seaborn', 'statsmodels',
    'shap', 'lime', 'xgboost', 'dotenv'
]

failures = []
for name in modules:
    try:
        importlib.import_module(name)
    except Exception as e:
        failures.append({'module': name, 'error': str(e)})
if failures:
    print('IMPORT FAILURES:\n' + json.dumps(failures, indent=2))
    sys.exit(1)
else:
    print('All core imports OK')
