# =========================
# AI LOAN APPROVAL SYSTEM
# End-to-End Streamlit App
# =========================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    roc_auc_score, roc_curve
)

# -------------------------
# STREAMLIT CONFIG
# -------------------------
st.set_page_config(
    page_title="💳 AI Loan Approval System",
    layout="wide"
)

st.title("💳 AI Loan Approval System")
st.caption("End-to-end AI system with analytics, fairness, and explainability")

# -------------------------
# DATA LOADING & CLEANING
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("loan_data.csv")
    df.columns = df.columns.str.strip()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Current_loan_status"])
    df = df.fillna(df.median(numeric_only=True))

    return df

df = load_data()

# -------------------------
# FEATURE / TARGET
# -------------------------
X = df.drop(["Current_loan_status", "home_ownership"], axis=1)
y = df["Current_loan_status"]

# -------------------------
# MODEL TRAINING (Random Forest)
# -------------------------
@st.cache_resource
def train_rf_model(X, y):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    return rf

rf_model = train_rf_model(X, y)

# -------------------------
# FEATURE EXPLAINABILITY (Top 5 Bar Chart)
# -------------------------
st.subheader("🔍 Top 5 Feature Explainability")

fi = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False).head(5)

st.bar_chart(fi.set_index("Feature"))

# -------------------------
# FAIRNESS ANALYSIS
# -------------------------
st.subheader("⚖️ Fairness Check (Historical Default)")

fair_df = X.copy()
fair_df["Prediction"] = rf_model.predict(X)
fair_df["historical_default_label"] = fair_df["historical_default"].map({0: "NO", 1: "Unknown", 2: "YES"})

approval_rate = fair_df.groupby("historical_default_label")["Prediction"].mean()
st.table(approval_rate)

# -------------------------
# LOAN ELIGIBILITY UI
# -------------------------
st.subheader("🧾 Loan Eligibility Checker")

with st.form("loan_form"):
    customer_age = st.slider("Customer Age", 18, 70, 30)
    customer_income = st.number_input("Annual Income", min_value=0, value=0)
    employment_duration = st.text_input("Employment Duration (months)", placeholder="e.g. 60")
    cred_hist_length = st.text_input("Credit History Length (years)", placeholder="e.g. 5")
    historical_default = st.selectbox("Historical Default", ["NO", "Unknown", "YES"])
    loan_amnt = st.number_input("Loan Amount", min_value=0, value=0)
    loan_int_rate = st.slider("Interest Rate (%)", 3.0, 15.0, 7.5)
    term_years = st.selectbox("Loan Term (years)", list(range(1, 11)))

    loan_intent = st.selectbox("Loan Intent", [
        "Debt Consolidation",
        "Education",
        "Home Improvement",
        "Medical",
        "Personal"
    ])

    loan_grade = st.selectbox("Loan Grade", [
        "Grade A: Lowest Risk",
        "Grade B: Low Risk",
        "Grade C: Moderate Risk",
        "Grade D: High Risk",
        "Grade E: Highest Risk"
    ])

    submit = st.form_submit_button("Check Eligibility")

if submit:
    if not employment_duration or not cred_hist_length or customer_income <= 0 or loan_amnt <= 0:
        st.error("⚠️ All fields must be answered correctly to proceed.")
    else:
        try:
            hist_map = {"NO": 0, "Unknown": 1, "YES": 2}
            intent_map = {
                "Debt Consolidation": 0, "Education": 1, "Home Improvement": 2,
                "Medical": 3, "Personal": 4
            }
            grade_map = {
                "Grade A: Lowest Risk": 0, "Grade B: Low Risk": 1, "Grade C: Moderate Risk": 2,
                "Grade D: High Risk": 3, "Grade E: Highest Risk": 4
            }

            user_data_raw = {
                "customer_age": customer_age,
                "customer_income": float(customer_income),
                "employment_duration": float(employment_duration),
                "loan_intent": intent_map[loan_intent],
                "loan_grade": grade_map[loan_grade],
                "loan_amnt": float(loan_amnt),
                "loan_int_rate": loan_int_rate,
                "term_years": term_years,
                "historical_default": hist_map[historical_default],
                "cred_hist_length": float(cred_hist_length)
            }

            user_df = pd.DataFrame([user_data_raw])

            pred = rf_model.predict(user_df)[0]
            prob = rf_model.predict_proba(user_df)[0][1]

            st.markdown("---")
            if pred == 1:
                st.success(f"✅ **Loan Approved** (Probability: {prob:.2%})")
                st.balloons()
            else:
                st.error(f"❌ **Loan Rejected** (Probability: {prob:.2%})")

                st.subheader("💡 Analysis of Rejection")
                st.write("Based on our AI model, the following factors likely influenced this decision:")

                reasons = []
                if user_data_raw["historical_default"] > 0:
                    reasons.append(
                        "• **Historical Default**: A history of default significantly increases lending risk.")
                if user_data_raw["customer_income"] < (user_data_raw["loan_amnt"] / 4):
                    reasons.append(
                        "• **Income-to-Loan Ratio**: Requested loan amount is high relative to annual income.")
                if user_data_raw["loan_int_rate"] > 12.0:
                    reasons.append(
                        "• **Interest Rate**: Higher rates increase repayment burden and default probability.")
                if user_data_raw["employment_duration"] < 12:
                    reasons.append("• **Job Stability**: Employment duration under 12 months suggests higher risk.")

                if not reasons:
                    reasons.append(
                        "• **Overall Risk Profile**: Combined financial attributes do not meet approval thresholds.")

                for r in reasons:
                    st.write(r)
                # -----------------------------------------

        except ValueError:
            st.error("Please enter valid numeric values for Employment Duration and Credit History.")

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.caption(
    "AI Loan Approval System demonstrating model construction, evaluation, "
    "fairness analysis, explainability, and deployment."
)