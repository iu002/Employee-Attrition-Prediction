import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE

# ==========================
# Load Dataset
# ==========================
data = pd.read_csv("employee_attrition.csv")

# Remove duplicates
data = data.drop_duplicates()

# Encode categorical columns
# Encode categorical columns
label_encoders = {}

categorical_columns = data.select_dtypes(include=["object", "string"]).columns

for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Features and Target
# Remove unnecessary columns
data = data.drop(
    columns=[
        "EmployeeNumber",
        "EmployeeCount",
        "StandardHours",
        "Over18"
    ]
)

# Features and target
X = data.drop("Attrition", axis=1)
y = data["Attrition"]

feature_names = X.columns.tolist()

joblib.dump(feature_names, "feature_names.pkl")

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# SMOTE
smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print(pd.Series(y_train).value_counts())

# Logistic Regression
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))
# Save the trained model
joblib.dump(model, "employee_attrition_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("\nModel saved successfully!")