import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Smart Fertilizer Recommendation System",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Smart Fertilizer Recommendation System")
st.markdown("AI-based fertilizer recommendation with soil health analysis and cost estimation")

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
    ("clf", RandomForestClassifier(n_estimators=250, random_state=42))
])

model.fit(X, y)

st.success("Model Ready ✅")

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
    probability = model.predict_proba(input_data).max()

    st.success(f"🌾 Recommended Fertilizer: {prediction}")
    st.info(f"🔍 Confidence Level: {round(probability*100,2)}%")

    # ---------------- SOIL HEALTH ANALYSIS ----------------
    st.subheader("📊 Soil Health Analysis")

    if nitrogen < 40:
        st.warning("Low Nitrogen – may affect leaf growth")

    if phosphorus < 30:
        st.warning("Low Phosphorus – root development may be weak")

    if potassium < 30:
        st.warning("Low Potassium – disease resistance reduced")

    if nitrogen >= 40 and phosphorus >= 30 and potassium >= 30:
        st.success("Soil Nutrient Levels are Balanced ✅")

    # ---------------- QUANTITY RECOMMENDATION (PER ACRE) ----------------
st.subheader("📦 Estimated Quantity Recommendation (Per Acre)")

# Ideal nutrient levels
ideal_N = 100
ideal_P = 60
ideal_K = 60

# Calculate deficiency (never negative)
def_N = max(0, ideal_N - nitrogen)
def_P = max(0, ideal_P - phosphorus)
def_K = max(0, ideal_K - potassium)

# Weighted fertilizer need (more realistic)
total_deficiency = (def_N * 0.5) + (def_P * 0.3) + (def_K * 0.2)

# Convert to kg/hectare (scaling factor)
quantity_hectare = total_deficiency * 2

# Convert hectare → acre
quantity_acre = round(quantity_hectare / 2.471, 2)

# Avoid zero fertilizer suggestion
if quantity_acre <= 0:
    quantity_acre = 5  # minimum baseline recommendation

st.info(f"Recommended Quantity: {quantity_acre} kg per acre")

    # ---------------- COST ESTIMATION ----------------
st.subheader("💰 Estimated Cost Per Acre")

# More realistic fertilizer pricing range
price_per_kg = 20 + (5 * probability)  # dynamic price based on confidence

cost = round(quantity_acre * price_per_kg, 2)

st.info(f"Estimated Cost: ₹{cost}")

    # ---------------- GRAPH VISUALIZATION ----------------
    st.subheader("📈 NPK Comparison (Current vs Target)")

    nutrients = ["Nitrogen", "Phosphorus", "Potassium"]
    values = [nitrogen, phosphorus, potassium]
    targets = [100, 60, 60]

    fig, ax = plt.subplots()
    ax.bar(nutrients, values)
    ax.plot(nutrients, targets)
    ax.set_ylabel("Nutrient Value")
    ax.set_title("NPK vs Ideal Target")

    st.pyplot(fig)
