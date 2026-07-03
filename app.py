import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================
# Page Configuration
# ==========================
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊",
    layout="wide"
)

# ==========================
# Load Saved Files
# ==========================
model = joblib.load("employee_attrition_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# ==========================
# Header
# ==========================
st.title("📊 Employee Attrition Prediction System")
st.markdown("### Logistic Regression Machine Learning Project")

st.success("✅ Model Loaded Successfully")

# ==========================
# Sidebar
# ==========================
st.sidebar.title("📌 Project Information")

st.sidebar.info("""
This project predicts whether an employee is likely to leave the company
using a Logistic Regression Machine Learning model.
""")

st.sidebar.markdown("### Tech Stack")

st.sidebar.write("""
- Python
- Pandas
- NumPy
- Scikit-learn
- SMOTE
- Streamlit
""")

# ==========================
# Dashboard Metrics
# ==========================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Dataset Size", "1470")

with col2:
    st.metric("Features", len(feature_names))

with col3:
    st.metric("Accuracy", "74.83%")

st.divider()

st.header("📝 Employee Information")

input_data = {}
# ==========================
# Employee Details
# ==========================

col1, col2 = st.columns(2)

with col1:
    input_data["Age"] = st.number_input("Age", 18, 60, 30)
    input_data["BusinessTravel"] = st.selectbox(
        "Business Travel",
        ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
    )
    input_data["DailyRate"] = st.number_input("Daily Rate", 100, 1500, 800)
    input_data["Department"] = st.selectbox(
        "Department",
        ["Sales", "Research & Development", "Human Resources"]
    )
    input_data["DistanceFromHome"] = st.slider(
        "Distance From Home", 1, 30, 5
    )
    input_data["Education"] = st.selectbox(
        "Education", [1, 2, 3, 4, 5]
    )
    input_data["EducationField"] = st.selectbox(
        "Education Field",
        [
            "Life Sciences",
            "Medical",
            "Marketing",
            "Technical Degree",
            "Human Resources",
            "Other"
        ]
    )
    input_data["EnvironmentSatisfaction"] = st.selectbox(
        "Environment Satisfaction",
        [1, 2, 3, 4]
    )

with col2:
    input_data["Gender"] = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    input_data["JobInvolvement"] = st.selectbox(
        "Job Involvement",
        [1, 2, 3, 4]
    )

    input_data["JobLevel"] = st.selectbox(
        "Job Level",
        [1, 2, 3, 4, 5]
    )

    input_data["JobRole"] = st.selectbox(
        "Job Role",
        [
            "Sales Executive",
            "Research Scientist",
            "Laboratory Technician",
            "Manufacturing Director",
            "Healthcare Representative",
            "Manager",
            "Sales Representative",
            "Research Director",
            "Human Resources"
        ]
    )

    input_data["JobSatisfaction"] = st.selectbox(
        "Job Satisfaction",
        [1, 2, 3, 4]
    )

    input_data["MaritalStatus"] = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"]
    )

    input_data["MonthlyIncome"] = st.number_input(
        "Monthly Income",
        1000,
        50000,
        10000
    )

    input_data["NumCompaniesWorked"] = st.slider(
        "Companies Worked",
        0,
        10,
        2
    )
    st.divider()

if st.button("🔮 Predict Attrition", use_container_width=True):

    # Default values for remaining numeric features
    input_data["HourlyRate"] = 65
    input_data["MonthlyRate"] = 15000
    input_data["OverTime"] = "No"
    input_data["PercentSalaryHike"] = 15
    input_data["PerformanceRating"] = 3
    input_data["RelationshipSatisfaction"] = 3
    input_data["StockOptionLevel"] = 1
    input_data["TotalWorkingYears"] = 10
    input_data["TrainingTimesLastYear"] = 3
    input_data["WorkLifeBalance"] = 3
    input_data["YearsAtCompany"] = 5
    input_data["YearsInCurrentRole"] = 3
    input_data["YearsSinceLastPromotion"] = 1
    input_data["YearsWithCurrManager"] = 3

    # Convert dictionary to DataFrame
    df = pd.DataFrame([input_data])

    # Encode categorical columns
    for column, encoder in label_encoders.items():
        if column in df.columns:
            df[column] = encoder.transform(df[column])

    # Arrange columns in the same order used for training
    df = df[feature_names]

    # Scale features
    df_scaled = scaler.transform(df)

    # Predict
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0]

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("❌ Employee is likely to Leave the Company")
    else:
        st.success("✅ Employee is likely to Stay with the Company")

    st.write(f"**Probability of Staying:** {probability[0]*100:.2f}%")
    st.write(f"**Probability of Leaving:** {probability[1]*100:.2f}%")