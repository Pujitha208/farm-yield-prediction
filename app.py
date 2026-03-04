import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("farm_yield_model.pkl")

# Load feature columns
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Farm Yield Prediction", layout="centered")

st.title("🌾 Smart Farm Yield Prediction App")
st.write("Enter farm details below to predict crop yield")

# Create input fields dynamically
input_data = []

for col in feature_columns:
    value = st.number_input(f"Enter {col}", value=0.0)
    input_data.append(value)

# Predict button
if st.button("Predict Yield"):
    
    new_data = pd.DataFrame([input_data], columns=feature_columns)
    
    prediction = model.predict(new_data)
    
    st.success(f"🌱 Predicted Crop Yield: {prediction[0]:.2f}")