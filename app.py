import streamlit as st
import numpy as np
import joblib

model = joblib.load("model.pkl")

st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏠 House Price Prediction System")

st.markdown("### Enter Property Details")

# 🔹 Income & Location
st.subheader("📍 Location & Income")
col1, col2 = st.columns(2)

with col1:
    MedInc = st.slider("Median Income", 0.0, 15.0, 3.0)

with col2:
    Population = st.number_input("Population", min_value=0.0, value=1000.0)

Latitude = st.slider("Latitude", 32.0, 42.0, 34.0)
Longitude = st.slider("Longitude", -124.0, -114.0, -118.0)

# 🔹 House Features
st.subheader("🏡 House Features")
col3, col4 = st.columns(2)

with col3:
    HouseAge = st.slider("House Age", 1, 50, 10)
    AveRooms = st.slider("Average Rooms", 1.0, 10.0, 5.0)

with col4:
    AveBedrms = st.slider("Bedrooms", 0.5, 5.0, 1.0)
    AveOccup = st.slider("Occupancy", 1.0, 10.0, 3.0)

# 🔹 Feature Selection Toggle (NEW 🔥)
st.subheader("⚙️ Feature Selection")

use_all = st.checkbox("Use all features", value=True)

if not use_all:
    selected_features = st.multiselect(
        "Select features to use:",
        ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"],
        default=["MedInc", "HouseAge", "AveRooms"]
    )
else:
    selected_features = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]

# 🔹 Prediction
if st.button("🔍 Predict Price"):
    input_data = {
        "MedInc": MedInc,
        "HouseAge": HouseAge,
        "AveRooms": AveRooms,
        "AveBedrms": AveBedrms,
        "Population": Population,
        "AveOccup": AveOccup,
        "Latitude": Latitude,
        "Longitude": Longitude
    }

    # Ensure correct order
    final_input = [input_data[feature] for feature in selected_features]

    # If fewer features selected → pad with zeros (simple handling)
    while len(final_input) < 8:
        final_input.append(0)

    final_array = np.array([final_input])
    prediction = model.predict(final_array)

    st.success(f"💰 Predicted House Price: {round(prediction[0], 2)}")
