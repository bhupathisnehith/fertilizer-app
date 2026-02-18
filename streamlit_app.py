import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Fertilizer Recommendation", layout="centered")

st.title("🌱 Fertilizer Recommendation System")

# ---- SAFE MODEL LOADING ----
try:
    model1 = joblib.load("model1_fertilizer_classifier.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    st.success("Model Loaded Successfully ✅")
except Exception as e:
    st.error("Model Loading Failed ❌")
    st.stop()

# ---- INPUT SECTION ----
soil = st.selectbox("Soil Color", ["Black", "Red", "Clayey", "Sandy"])
crop = st.text_input("Crop Name")
nitrogen = st.number_input("Nitrogen", min_value=0, max_value=300)
phosphorus = st.number_input("Phosphorus", min_value=0, max_value=300)
potassium = st.number_input("Potassium", min_value=0, max_value=300)

# ---- PREDICTION ----
if st.button("Recommend Fertilizer"):

    if crop == "":
        st.warning("Please enter crop name")
        st.stop()

    input_data = {
        "Soil_color": soil,
        "Nitrogen": nitrogen,
        "Phosphorus": phosphorus,
        "Potassium": potassium,
        "Crop": crop
    }

    df = pd.DataFrame([input_data])[feature_columns]

    prediction = model1.predict(df)[0]
    confidence = max(model1.predict_proba(df)[0])

    st.success(f"Recommended Fertilizer: {prediction}")
    st.info(f"Confidence: {round(confidence*100,2)}%")
