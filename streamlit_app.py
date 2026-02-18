import streamlit as st
import joblib
import pandas as pd

# Load models
model1 = joblib.load("model1_fertilizer_classifier.pkl")
model2 = joblib.load("model2_fertilizer_quantity.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("🌱 Fertilizer Recommendation System")

soil = st.selectbox("Soil Color", ["Black", "Red", "Clayey", "Sandy"])
crop = st.text_input("Crop Name")
nitrogen = st.number_input("Nitrogen", 0, 200)
phosphorus = st.number_input("Phosphorus", 0, 200)
potassium = st.number_input("Potassium", 0, 200)

if st.button("Recommend"):
    input_data = {
        "Soil_color": soil,
        "Nitrogen": nitrogen,
        "Phosphorus": phosphorus,
        "Potassium": potassium,
        "Crop": crop
    }

    df = pd.DataFrame([input_data])[feature_columns]

    fertilizer = model1.predict(df)[0]

    st.success(f"Recommended Fertilizer: {fertilizer}")
