import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, scaler, encoders
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.title("Employee Salary Prediction")

# User input
age = st.number_input("Age", 18, 100, 30)
workclass = st.selectbox("Workclass", label_encoders['workclass'].classes_)
fnlwgt = st.number_input("Fnlwgt", 10000, 1000000, 200000)
education = st.selectbox("Education", label_encoders['education'].classes_)
education_num = st.number_input("Education Number", 1, 20, 10)
marital_status = st.selectbox("Marital Status", label_encoders['marital-status'].classes_)
occupation = st.selectbox("Occupation", label_encoders['occupation'].classes_)
relationship = st.selectbox("Relationship", label_encoders['relationship'].classes_)
race = st.selectbox("Race", label_encoders['race'].classes_)
sex = st.selectbox("Sex", label_encoders['sex'].classes_)
capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
capital_loss = st.number_input("Capital Loss", 0, 5000, 0)
hours_per_week = st.number_input("Hours per Week", 1, 100, 40)
native_country = st.selectbox("Native Country", label_encoders['native-country'].classes_)

# Prepare input
input_dict = {
    'age': age,
    'workclass': label_encoders['workclass'].transform([workclass])[0],
    'fnlwgt': fnlwgt,
    'education': label_encoders['education'].transform([education])[0],
    'education-num': education_num,
    'marital-status': label_encoders['marital-status'].transform([marital_status])[0],
    'occupation': label_encoders['occupation'].transform([occupation])[0],
    'relationship': label_encoders['relationship'].transform([relationship])[0],
    'race': label_encoders['race'].transform([race])[0],
    'sex': label_encoders['sex'].transform([sex])[0],
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'hours-per-week': hours_per_week,
    'native-country': label_encoders['native-country'].transform([native_country])[0]
}

input_df = pd.DataFrame([input_dict])
input_scaled = scaler.transform(input_df)

# Predict
if st.button("Predict Salary Category"):
    prediction = model.predict(input_scaled)[0]
    salary_label = label_encoders['salary'].inverse_transform([prediction])[0]
    st.success(f"Predicted Salary Category: {salary_label}")