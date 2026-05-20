import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="CreditWise AI",
    page_icon="💳",
    layout="wide"
)

# ==========================================
# TITLE
# ==========================================

st.title("💳 CreditWise Loan Approval Prediction")
st.markdown("### AI Based Loan Approval Prediction System")

# ==========================================
# LOAD DATASET
# ==========================================

@st.cache_data

def load_data():
    df = pd.read_csv("loan_approval_data.csv")
    return df


df = load_data()

# ==========================================
# LABEL ENCODING
# ==========================================

le = LabelEncoder()

df["Education_Level"] = le.fit_transform(df["Education_Level"])
df["Loan_Approved"] = le.fit_transform(df["Loan_Approved"])

# ==========================================
# ONE HOT ENCODING
# ==========================================

cols = [
    "Employment_Status",
    "Marital_Status",
    "Loan_Purpose",
    "Property_Area",
    "Gender",
    "Employer_Category"
]


ohe = OneHotEncoder(
    drop="first",
    sparse_output=False,
    handle_unknown="ignore"
)

encoded = ohe.fit_transform(df[cols])

encoded_df = pd.DataFrame(
    encoded,
    columns=ohe.get_feature_names_out(cols),
    index=df.index
)

# Merge

df = pd.concat([
    df.drop(columns=cols),
    encoded_df
], axis=1)

# ==========================================
# FEATURE ENGINEERING
# ==========================================

# Same logic used in notebook

df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
df["Credit_Score_sq"] = df["Credit_Score"] ** 2

# ==========================================
# FEATURES AND TARGET
# ==========================================

X = df.drop(columns=[
    "Loan_Approved",
    "Credit_Score",
    "DTI_Ratio"
])

y = df["Loan_Approved"]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# SCALING
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# BEST MODEL : XGBOOST
# ==========================================

xgb_model = XGBClassifier()

xgb_model.fit(X_train_scaled, y_train)

# Accuracy

y_pred_xgb = xgb_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred_xgb)

# ==========================================
# SIDEBAR INPUTS
# ==========================================

st.sidebar.header("Enter Applicant Details")

Applicant_Income = st.sidebar.number_input(
    "Applicant Income",
    min_value=0,
    value=50000
)

Coapplicant_Income = st.sidebar.number_input(
    "Coapplicant Income",
    min_value=0,
    value=20000
)

Loan_Amount = st.sidebar.number_input(
    "Loan Amount",
    min_value=1000,
    value=150000
)

Loan_Term = st.sidebar.number_input(
    "Loan Term (Months)",
    min_value=1,
    value=36
)

Age = st.sidebar.slider(
    "Age",
    18,
    70,
    30
)

Credit_Score = st.sidebar.slider(
    "Credit Score",
    300,
    900,
    700
)

DTI_Ratio = st.sidebar.slider(
    "DTI Ratio",
    0.0,
    1.0,
    0.3
)

Existing_Loans = st.sidebar.number_input(
    "Existing Loans",
    min_value=0,
    value=1
)

Dependents = st.sidebar.number_input(
    "Dependents",
    min_value=0,
    value=1
)

Savings = st.sidebar.number_input(
    "Savings",
    min_value=0,
    value=100000
)

Collateral_Value = st.sidebar.number_input(
    "Collateral Value",
    min_value=0,
    value=200000
)

Education_Level = st.sidebar.selectbox(
    "Education Level",
    ["Graduate", "Not Graduate"]
)

Employment_Status = st.sidebar.selectbox(
    "Employment Status",
    ["Employed", "Self-Employed", "Unemployed"]
)

Marital_Status = st.sidebar.selectbox(
    "Marital Status",
    ["Single", "Married"]
)

Loan_Purpose = st.sidebar.selectbox(
    "Loan Purpose",
    ["Home", "Business", "Education", "Vehicle"]
)

Property_Area = st.sidebar.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)

Gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

Employer_Category = st.sidebar.selectbox(
    "Employer Category",
    ["Private", "Government", "Business"]
)

# ==========================================
# CREATE INPUT DATAFRAME
# ==========================================

input_data = {
    "Applicant_Income": Applicant_Income,
    "Coapplicant_Income": Coapplicant_Income,
    "Loan_Amount": Loan_Amount,
    "Loan_Term": Loan_Term,
    "Age": Age,
    "Existing_Loans": Existing_Loans,
    "Dependents": Dependents,
    "Savings": Savings,
    "Collateral_Value": Collateral_Value,
    "Education_Level": 1 if Education_Level == "Graduate" else 0,
    "DTI_Ratio_sq": DTI_Ratio ** 2,
    "Credit_Score_sq": Credit_Score ** 2
}

# ==========================================
# HANDLE ONE HOT ENCODING INPUTS
# ==========================================

for col in encoded_df.columns:
    input_data[col] = 0

# Employment Status
if f"Employment_Status_{Employment_Status}" in input_data:
    input_data[f"Employment_Status_{Employment_Status}"] = 1

# Marital Status
if f"Marital_Status_{Marital_Status}" in input_data:
    input_data[f"Marital_Status_{Marital_Status}"] = 1

# Loan Purpose
if f"Loan_Purpose_{Loan_Purpose}" in input_data:
    input_data[f"Loan_Purpose_{Loan_Purpose}"] = 1

# Property Area
if f"Property_Area_{Property_Area}" in input_data:
    input_data[f"Property_Area_{Property_Area}"] = 1

# Gender
if f"Gender_{Gender}" in input_data:
    input_data[f"Gender_{Gender}"] = 1

# Employer Category
if f"Employer_Category_{Employer_Category}" in input_data:
    input_data[f"Employer_Category_{Employer_Category}"] = 1

# ==========================================
# FINAL INPUT DATAFRAME
# ==========================================

input_df = pd.DataFrame([input_data])

# Reorder columns same as training data
input_df = input_df.reindex(columns=X.columns, fill_value=0)

# Scaling
input_scaled = scaler.transform(input_df)

# ==========================================
# DISPLAY MODEL INFO
# ==========================================

st.subheader("📊 Best Model Information")

st.success(f"Best Model: XGBoost Classifier")
st.success(f"Model Accuracy: {round(accuracy * 100, 2)}%")

# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button("Predict Loan Approval"):

    prediction = xgb_model.predict(input_scaled)[0]

    probability = xgb_model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.subheader("Approval Probability")

    st.progress(float(probability))

    st.write(
        f"Probability of Approval: {round(probability * 100, 2)}%"
    )

# ==========================================
# DATA PREVIEW
# ==========================================

st.subheader("📁 Dataset Preview")

st.dataframe(df.head())
