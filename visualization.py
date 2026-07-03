import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("employee_attrition.csv")

# Encode categorical columns
encoded = data.copy()

for col in encoded.select_dtypes(include=["object", "string"]).columns:
    encoded[col] = encoded[col].astype("category").cat.codes

# -------------------------
# Attrition Count Plot
# -------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="Attrition", data=data)
plt.title("Employee Attrition Count")
plt.tight_layout()
plt.savefig("attrition_count.png")

# -------------------------
# Correlation Heatmap
# -------------------------
plt.figure(figsize=(14,10))
sns.heatmap(encoded.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")

plt.show()