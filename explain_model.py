import pandas as pd
import joblib
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

# ✅ Load model and data
model = joblib.load("models/model_24h_1year.pkl")
df = pd.read_csv("data/clean_aqi.csv")

# ✅ Use same features the model was trained on
trained_features = model.get_booster().feature_names
X = df[trained_features]

# ✅ Take 1 sample for explanation
sample = X.iloc[0:1]

# --------------------------
# ✅ SHAP EXPLANATIONS
# --------------------------
explainer = shap.Explainer(model.predict, X)
shap_values = explainer(X)

# ✅ SHAP beeswarm
shap.summary_plot(shap_values, X, show=False)
plt.savefig("shap_beeswarm.png")
plt.close()

# ✅ SHAP bar chart
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.savefig("shap_bar.png")
plt.close()


# --------------------------
# ✅ LIME EXPLANATION
# --------------------------
lime = LimeTabularExplainer(
    X.values,
    feature_names=trained_features,
    mode="regression",
    discretize_continuous=True
)

exp = lime.explain_instance(
    sample.values[0],
    model.predict
)

# ✅ Save textual feature impact
with open("lime_feature_details.txt", "w") as f:
    for feature, impact in exp.as_list():
        f.write(f"{feature} → {impact}\n")

# ✅ Generate and save LIME bar graph
fig = exp.as_pyplot_figure()
plt.tight_layout()
fig.savefig("lime_explanation_full.png", dpi=300)
plt.close()

print("✅ FIXED: SHAP & LIME generated successfully!")
