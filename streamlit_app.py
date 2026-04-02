import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ---------------- PAGE ----------------
st.set_page_config(page_title="Smart Agri AI System", layout="wide")
st.title("🌾 Smart Agriculture System")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("Crop and fertilizer dataset.csv")
    df = df.drop(columns=["District_Name","Temperature","Rainfall","pH","Link"], errors='ignore')
    df["Soil_color"] = df["Soil_color"].str.strip()
    return df

df = load_data()

# ---------------- MODEL ----------------
X = df.drop("Fertilizer", axis=1)
y = df["Fertilizer"]

cat_cols = ["Soil_color","Crop"]
num_cols = [c for c in X.columns if c not in cat_cols]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

model = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=400, max_depth=12, random_state=42))
])

# Train model (no accuracy display)
model.fit(X, y)


# ---------------- INPUT ----------------
st.subheader("🧪 Soil Inputs")

col1, col2 = st.columns(2)

with col1:
    soil = st.selectbox("Soil Color", df["Soil_color"].unique())
    crop = st.selectbox("Crop", df["Crop"].unique())

with col2:
    nitrogen = st.number_input("Nitrogen", 0, 300, 50)
    phosphorus = st.number_input("Phosphorus", 0, 300, 40)
    potassium = st.number_input("Potassium", 0, 300, 40)

# ---------------- WEATHER ----------------
st.subheader("🌦 Weather Condition")

weather = st.selectbox("Weather", [
    "Sunny", "Rainy", "Cloudy", "Overcast", "Windy",
    "Snowy", "Foggy/Mis", "Thunderstorms", "Sandstorms"
])

rain_factor_dict = {
    "Sunny": 0.8,
    "Rainy": 1.2,
    "Cloudy": 1.0,
    "Overcast": 1.1,
    "Windy": 0.9,
    "Snowy": 0.7,
    "Foggy/Mis": 0.85,
    "Thunderstorms": 1.15,
    "Sandstorms": 0.6
}

rain_factor = rain_factor_dict.get(weather, 1.0)

# ---------------- PRICES ----------------
fertilizer_prices = {
    "Urea": 6,
    "DAP": 25,
    "MOP": 15,
    "NPK": 20
}

crop_prices = {
    "Rice": 20, "Wheat": 18, "Maize": 15,
    "Sugarcane": 3, "Jowar": 22, "Cotton": 60,
    "Groundnut": 55, "Tur": 70, "Urad": 65,
    "Moong": 72, "Gram": 50, "Masoor": 55,
    "Soybean": 40, "Ginger": 120,
    "Turmeric": 110, "Grapes": 80
}

# ---------------- RUN MODEL ----------------
if st.button("🚀 Weather Prediction"):

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
        prob = float(model.predict_proba(input_data).max())
    else:
        prob = 0.75

    st.success(f"🌾 Fertilizer: {prediction}")
    st.info(f"Confidence: {round(prob*100,2)}%")

    # ---------------- AI INSIGHT ----------------
    st.subheader("🤖 AI Insight")

    if nitrogen < 40:
        st.write("✔ Low Nitrogen → Boost leaf growth")
    if phosphorus < 30:
        st.write("✔ Low Phosphorus → Better roots")
    if potassium < 30:
        st.write("✔ Low Potassium → Disease resistance")

    # ---------------- QUANTITY ----------------
    deficiency = (
        max(0, 100 - nitrogen) +
        max(0, 60 - phosphorus) +
        max(0, 60 - potassium)
    )

    qty = round((deficiency / 3) / 2.471, 2)

    if qty <= 0:
        qty = 10

    st.info(f"Fertilizer Needed: {qty} kg/acre")

    # ---------------- COST ----------------
    price = fertilizer_prices.get(prediction, 20)
    cost = qty * price

    st.info(f"Cost: ₹{round(cost,2)}")

    # ---------------- YIELD ----------------
    st.subheader("📈 Yield Prediction")

    base_yield = (nitrogen + phosphorus + potassium) / 3
    yield_est = round(base_yield * rain_factor, 2)

    st.success(f"Estimated Yield: {yield_est} quintals/acre")

    # ---------------- PROFIT ----------------
    st.subheader("💰 Profit Estimation")

    crop_price = crop_prices.get(crop, 20)
    revenue = yield_est * crop_price
    profit = revenue - cost

    st.success(f"Revenue: ₹{round(revenue,2)}")
    st.success(f"Profit: ₹{round(profit,2)}")
