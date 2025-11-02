"""
Explain AQI Model using SHAP + LIME
- SHAP: global feature importance (bar chart)
- LIME: local explanation with <, >, <=, >= feature rules
"""

import pandas as pd
import joblib
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

print("✅ Loading test dataset...")
test_df = pd.read_csv("data/clean_test.csv")

# Remove target column
X_test = test_df.drop(columns=["AQI"])

print("✅ Loading trained model...")
model = joblib.load("models/xgboost_24hr.pkl")

# ---------------------------------------------------
# SHAP EXPLANATION (Global)
# ---------------------------------------------------
print("\n===== SHAP Explanation =====")

# Use only 50 rows for speed
sample = X_test[:50]

# SHAP works with any ML model using PermutationExplainer
explainer = shap.PermutationExplainer(model.predict, sample)
shap_values = explainer(sample)

# Create a bar summary plot and save as PNG
shap.summary_plot(shap_values.values, sample, plot_type="bar", show=False)
plt.savefig("shap_summary.png")
plt.close()

print("✅ SHAP summary saved as shap_summary.png")


# ---------------------------------------------------
# LIME EXPLANATION (Local)
# ---------------------------------------------------
print("\n===== LIME Explanation =====")

# Build LIME explainer
lime_explainer = LimeTabularExplainer(
    training_data=X_test.values,
    feature_names=X_test.columns,
    mode='regression'
)

# Explain first prediction
exp = lime_explainer.explain_instance(
    data_row=X_test.iloc[0].values,
    predict_fn=model.predict
)

# Print readable formatted output
print("\nTop LIME Feature Rules:")
for feature, weight in exp.as_list():
    print(f"{feature}  →  impact = {weight:.4f}")

# Save full HTML report
exp.save_to_file("lime_report.html")
print("✅ LIME report saved as lime_report.html")

print("\n✅ Done — SHAP & LIME explanations generated successfully!")
