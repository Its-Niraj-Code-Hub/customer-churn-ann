import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from tensorflow import keras

BASE = Path(__file__).parent

@st.cache_resource
def load_files():
    model = keras.models.load_model(BASE / "models" / "churn_model.keras")
    scaler = joblib.load(BASE / "models" / "scaler.pkl")
    with open(BASE / "models" / "feature_columns.json") as f:
        columns = json.load(f)
    return model, scaler, columns

model, scaler, columns = load_files()

st.title("Customer Churn Predictor")
st.write("Enter a customer's details to estimate how likely they are to leave.")

# ---------- Inputs ----------
c1, c2 = st.columns(2)
with c1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", ["No", "Yes"])
    partner = st.selectbox("Has partner", ["No", "Yes"])
    dependents = st.selectbox("Has dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", ["No", "Yes"])
    payment = st.selectbox("Payment method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.number_input("Monthly charges", 18.0, 120.0, 70.0)

with c2:
    phone = st.selectbox("Phone service", ["Yes", "No"])
    multi = st.selectbox("Multiple lines", ["No", "Yes"]) if phone == "Yes" else "No"
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    if internet != "No":
        security = st.selectbox("Online security", ["No", "Yes"])
        backup = st.selectbox("Online backup", ["No", "Yes"])
        device = st.selectbox("Device protection", ["No", "Yes"])
        tech = st.selectbox("Tech support", ["No", "Yes"])
        stv = st.selectbox("Streaming TV", ["No", "Yes"])
        smov = st.selectbox("Streaming movies", ["No", "Yes"])
    else:
        security = backup = device = tech = stv = smov = "No"

# ---------- Build the row exactly like the training data ----------
yn = lambda v: 1 if v == "Yes" else 0

row = dict.fromkeys(columns, 0)
row.update({
    "gender": 1 if gender == "Female" else 0,
    "SeniorCitizen": yn(senior),
    "Partner": yn(partner),
    "Dependents": yn(dependents),
    "tenure": tenure,
    "PhoneService": yn(phone),
    "MultipleLines": yn(multi),
    "OnlineSecurity": yn(security),
    "OnlineBackup": yn(backup),
    "DeviceProtection": yn(device),
    "TechSupport": yn(tech),
    "StreamingTV": yn(stv),
    "StreamingMovies": yn(smov),
    "PaperlessBilling": yn(paperless),
    "MonthlyCharges": monthly,
    "TotalCharges": tenure * monthly,   # estimate
    "InternetService_Fiber optic": int(internet == "Fiber optic"),
    "InternetService_No": int(internet == "No"),
    "Contract_One year": int(contract == "One year"),
    "Contract_Two year": int(contract == "Two year"),
    "PaymentMethod_Credit card (automatic)": int(payment == "Credit card (automatic)"),
    "PaymentMethod_Electronic check": int(payment == "Electronic check"),
    "PaymentMethod_Mailed check": int(payment == "Mailed check"),
})

X = pd.DataFrame([row])[columns]
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
X[num_cols] = scaler.transform(X[num_cols])

# ---------- Predict ----------
if st.button("Predict"):
    prob = float(model.predict(X.astype("float32"), verbose=0)[0][0])
    st.metric("Churn probability", f"{prob:.1%}")
    st.progress(min(prob, 1.0))

    if prob > 0.5:
        st.error("High risk: this customer is likely to leave.")
    elif prob > 0.3:
        st.warning("Medium risk: worth keeping an eye on this customer.")
    else:
        st.success("Low risk: this customer is likely to stay.")

    st.caption("Total charges is estimated as tenure × monthly charges.")