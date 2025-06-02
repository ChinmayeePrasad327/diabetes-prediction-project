import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load model (update filename if needed)
model = pickle.load(open('diabetes_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("🩺 Early Diabetes Prediction App")
st.write("Enter the patient's data below to predict the risk of diabetes.")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=0, max_value=200, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=140, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=79)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
age = st.number_input("Age", min_value=1, max_value=120, value=33)

# Predict button
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ The person is likely to have diabetes.")
    else:
        st.success("✅ The person is unlikely to have diabetes.")

