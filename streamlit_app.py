import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Smart Fertilizer Recommendation System",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Smart Fertilizer Recommendation System")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("Crop and fertilizer dataset.csv")
    df = df.drop(columns=["District_Name","Temperature","Rainfall","pH","Link"])
    df["Soil_color"] = df["Soil_color"].str.strip()
    return df

df = load_data()

# ---------------- TRAIN MODEL ----------------
X = df.drop("Fertilizer", axis=1)
y = df["Fertilizer"]

categorical_cols = ["Soil_color","Crop"]
numeric_cols = [c for c in X.columns if c not in categorical_cols]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", "passthrough", numeric_cols)
])

model = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42
    ))
])

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

st.success(f"Model Ready ✅ (Accuracy: {round(accuracy*100,2)}%)")

# ---------------- INPUT SECTION ----------------
st.subheader("🧪 Enter Soil Parameters")

col1, col2 = st.columns(2)

with col1:
    soil = st.selectbox("Soil Color", df["Soil_color"].unique())
    crop = st.selectbox("Crop", df["Crop"].unique())

with col2:
    nitrogen = st.number_input("Nitrogen (N)", 0, 300)
    phosphorus = st.number_input("Phosphorus (P)", 0, 300)
    potassium = st.number_input("Potassium (K)", 0, 300)

# Fertilizer price mapping
fertilizer_prices = {
    "Urea": 6,
    "DAP": 25,
    "MOP": 15,
    "NPK": 20,
    "Compost": 5
}

# ---------------- PREDICTION ----------------
if st.button("Recommend Fertilizer"):

    input_data = pd.DataFrame([{
        "Soil_color": soil,
        "Nitrogen": nitrogen,
        "Phosphorus": phosphorus,
        "Potassium": potassium,
        "Crop": crop
    }])

    prediction = model.predict(input_data)[0]

    # Safe probability
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(input_data).max())
    else:
        probability = 0.75

    st.success(f"🌾 Recommended Fertilizer: {prediction}")
    st.info(f"🔍 Confidence Level: {round(probability*100,2)}%")

    # ---------------- AI EXPLANATION ----------------
    st.subheader("🤖 AI Recommendation Insight")

    reasons = []

    if nitrogen < 40:
        reasons.append("Nitrogen is low → improves leaf growth")

    if phosphorus < 30:
        reasons.append("Phosphorus is low → improves root development")

    if potassium < 30:
        reasons.append("Potassium is low → increases disease resistance")

    if not reasons:
        reasons.append("Soil nutrients are balanced → maintenance fertilizer recommended")

    for r in reasons:
        st.write(f"✔ {r}")

    # ---------------- SOIL HEALTH ----------------
    st.subheader("📊 Soil Health Analysis")

    if nitrogen < 40:
        st.warning("Low Nitrogen")

    if phosphorus < 30:
        st.warning("Low Phosphorus")

    if potassium < 30:
        st.warning("Low Potassium")

    if nitrogen >= 40 and phosphorus >= 30 and potassium >= 30:
        st.success("Soil Nutrient Levels are Balanced ✅")

    # ---------------- QUANTITY ----------------
    st.subheader("📦 Estimated Quantity Recommendation (Per Acre)")

    avg_deficiency = (
        max(0, 100 - nitrogen) +
        max(0, 60 - phosphorus) +
        max(0, 60 - potassium)
    )

    quantity_hectare = avg_deficiency / 3
    quantity_acre = round(quantity_hectare / 2.471, 2)

    # Ensure minimum fertilizer
    if quantity_acre <= 0:
        quantity_acre = 10

    st.info(f"Recommended Quantity: {quantity_acre} kg per acre")

    # ---------------- COST ----------------
    st.subheader("💰 Estimated Cost Per Acre")

    price_per_kg = fertilizer_prices.get(prediction, 20)
    cost = round(quantity_acre * price_per_kg, 2)

    st.info(f"Fertilizer Price: ₹{price_per_kg}/kg")
    st.info(f"Estimated Cost: ₹{cost}")

   
