import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved Random Forest model, label encoder, and feature names
model = joblib.load("random_forest_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
feature_names = joblib.load("feature_names.pkl")

# Streamlit app title and description
st.title("AQI Prediction App")
st.write("Enter the required values to predict the AQI bucket.")

# Generate input fields for each of the first 12 features
input_data = []
for feature in feature_names:
    value = st.number_input(f"Enter value for {feature}", value=0.0)
    input_data.append(value)

# Convert input data into a DataFrame
input_df = pd.DataFrame([input_data], columns=feature_names)

# Prediction
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)
        predicted_bucket = label_encoder.inverse_transform(prediction)
        st.success(f"The predicted AQI bucket is: {predicted_bucket[0]}")
    except ValueError as e:
        st.error(f"Error: {e}")
