
import streamlit as st
import pickle
import pandas as pd

# ============================================
# LOAD MODEL
# ============================================

with open("car_price_model.pkl", "rb") as f:
    model = pickle.load(f)

# ============================================
# LOAD SCALER
# ============================================

with open("car_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ============================================
# LOAD COLUMN NAMES
# ============================================

with open("car_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

# ============================================
# STREAMLIT UI
# ============================================

st.title("🚗 Car Price Prediction App")

st.write("Enter car details below:")

# ============================================
# USER INPUTS
# ============================================

year = st.number_input("Car Year", min_value=1990, max_value=2030)

present_price = st.number_input("Present Price", min_value=0.0)

kms_driven = st.number_input("Kilometers Driven", min_value=0)

owner = st.number_input("Number of Previous Owners", min_value=0, max_value=10)

fuel_type = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "CNG"]
)

seller_type = st.selectbox(
    "Seller Type",
    ["Dealer", "Individual"]
)

transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic"]
)

# ============================================
# CREATE INPUT DATAFRAME
# ============================================

input_data = pd.DataFrame({
    'Year': [year],
    'Present_Price': [present_price],
    'Kms_Driven': [kms_driven],
    'Owner': [owner],
    'Fuel_Type': [fuel_type],
    'Seller_Type': [seller_type],
    'Transmission': [transmission]
})

# ============================================
# ONE HOT ENCODING
# ============================================

input_data = pd.get_dummies(input_data)

# ============================================
# MATCH TRAINING COLUMNS
# ============================================

input_data = input_data.reindex(
    columns=model_columns,
    fill_value=0
)

# ============================================
# SCALE DATA
# ============================================

scaled_data = scaler.transform(input_data)

# ============================================
# PREDICTION
# ============================================

if st.button("Predict Car Price"):

    prediction = model.predict(scaled_data)

    st.success(f"🚘 Predicted Car Price: {prediction[0]:,.2f} Lakhs")
