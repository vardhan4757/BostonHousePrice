import streamlit as st
import numpy as np
import joblib

model = joblib.load("model.pkl")

st.title("House Price Prediction")

MedInc = st.number_input("Median Income")
HouseAge = st.number_input("House Age")
AveRooms = st.number_input("Average Rooms")
AveBedrms = st.number_input("Bedrooms")
Population = st.number_input("Population")
AveOccup = st.number_input("Occupancy")
Latitude = st.number_input("Latitude")
Longitude = st.number_input("Longitude")

if st.button("Predict"):
    features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
    prediction = model.predict(features)

    st.success(f"Predicted Price: {prediction[0]}")
