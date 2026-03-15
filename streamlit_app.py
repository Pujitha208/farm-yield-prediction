import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Farm Yield Prediction", page_icon="🌾", layout="centered")

# ---------------------------
# Header
# ---------------------------
st.title("🌾 Farm Yield Prediction")

# ---------------------------
# User Inputs
# ---------------------------
st.header("Enter Farm Data")
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)
temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0)
fertilizer = st.number_input("Fertilizer Used (kg/acre)", min_value=0.0, max_value=1000.0, value=50.0)

# ---------------------------
# Sample Linear Regression Model
# ---------------------------
model = LinearRegression()
# Dummy fit to prevent sklearn warnings
X_dummy = np.array([[0,0,0],[1,1,1]])
y_dummy = np.array([2,2.17])
model.fit(X_dummy, y_dummy)
# Hardcoded coefficients for demonstration
model.coef_ = np.array([0.05, 0.1, 0.02])
model.intercept_ = 2

# ---------------------------
# Prediction
# ---------------------------
st.header("Prediction")
if st.button("Predict Yield"):
    input_data = np.array([[rainfall, temperature, fertilizer]])
    prediction = model.predict(input_data)[0]
    
    st.success(f"🌾 Predicted Farm Yield: {prediction:.2f} quintals/acre")
