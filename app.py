import streamlit as st

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
    font-family: 'Segoe UI', sans-serif;
}

/* Title Styling */
h1 {
    color: #2e7d32;
    text-align: center;
    font-size: 40px;
}

/* Input Box Styling */
.stTextInput>div>div>input {
    background-color: #ffffff;
    border-radius: 10px;
    border: 2px solid #81c784;
    padding: 10px;
}

/* Number Input */
.stNumberInput>div>div>input {
    background-color: #ffffff;
    border-radius: 10px;
    border: 2px solid #81c784;
}

/* Selectbox */
.stSelectbox>div>div {
    background-color: #ffffff;
    border-radius: 10px;
    border: 2px solid #81c784;
}

/* Button Styling */
.stButton>button {
    background-color: #43a047;
    color: white;
    font-size: 18px;
    border-radius: 12px;
    padding: 10px 30px;
}

.stButton>button:hover {
    background-color: #2e7d32;
}

/* Prediction Result Box */
.result-box {
    background-color: #a5d6a7;
    color: #1b5e20;
    padding: 20px;
    border-radius: 15px;
    font-size: 24px;
    text-align: center;
    font-weight: bold;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
}

/* Card Style */
.card {
    background-color: #ffffff;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 3px 3px 12px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)
import streamlit as st
import joblib
import requests

# API endpoint (Flask backend)
API_URL = "http://127.0.0.1:5000/predict"

# Load feature names from shared pickle file; must match backend
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Farm Yield Prediction", layout="centered")

st.markdown(
    """
    <style>
    body {
      color: #243f21;
      background: linear-gradient(135deg, #dff5e2 20%, #d6ea96 100%);
    }
    .stApp {
      background: linear-gradient(135deg, rgba(98,182,112,0.25), rgba(172,219,103,0.25));
      padding: 1.5rem;
      border-radius: 1rem;
      box-shadow: 0 2px 22px rgba(0,0,0,0.16);
    }
    .css-1d391kg {
      background: rgba(255,255,255,0.75) !important;
      border-radius: 1rem !important;
    }
    .streamlit-expanderHeader {
      font-weight: 700 !important;
    }
    .stButton>button {
      background: linear-gradient(145deg, #3e8f3f 0%, #6eb36e 100%) !important;
      color: white !important;
      border-radius: 0.7rem !important;
      padding: 0.6rem 1.1rem !important;
      font-weight: 700 !important;
      border: none !important;
      box-shadow: 0 8px 16px rgba(56, 136, 62, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌾 Smart Farm Yield Prediction App")
st.write("Enter farm details below to predict crop yield")

# Create input fields dynamically
input_data = {}
for col in feature_columns:
    input_data[col] = st.number_input(f"Enter {col}", value=0.0)

# Predict button
if st.button("Predict Yield"):
    with st.spinner("Contacting backend..."):
        try:
            response = requests.post(API_URL, json=input_data, timeout=10)
            response.raise_for_status()
            result = response.json()
            prediction = result.get("predicted_yield")
            if prediction is None:
                st.error("Backend response missing prediction value")
            else:
                st.success(f"🌱 Predicted Crop Yield: {prediction:.2f}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error calling backend API: {e}")
            st.write("Make sure flask_api.py is running at http://127.0.0.1:5000")




