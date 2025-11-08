import pandas as pd
import joblib
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt


model = joblib.load("models/model_24h_1year.pkl")
df = pd.read_csv("data/clean_aqi.csv")


trained_features = model.get_booster().feature_names
X = df[trained_features]


sample = X.iloc[0:1]




explainer = shap.Explainer(model.predict, X)
shap_values = explainer(X)


shap.summary_plot(shap_values, X, show=False)
plt.savefig("shap_beeswarm.png")
plt.close()


shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.savefig("shap_bar.png")
plt.close()





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


with open("lime_feature_details.txt", "w", encoding="utf-8") as f:
    for feature, impact in exp.as_list():
        f.write(f"{feature} -> {impact}\n")
fig = exp.as_pyplot_figure()
plt.tight_layout()
fig.savefig("lime_explanation_full.png", dpi=300)
plt.close()

print("SHAP & LIME generated successfully!")