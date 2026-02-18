import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Fertilizer Recommendation", layout="centered")

st.title("🌱 Fertilizer Recommendation System")

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
    ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
])

model.fit(X, y)

st.success("Model Ready ✅")

# ---------------- INPUT ----------------
soil = st.selectbox("Soil Color", df["Soil_color"].unique())
crop = st.selectbox("Crop", df["Crop"].unique())
nitrogen = st.number_input("Nitrogen", 0, 300)
phosphorus = st.number_input("Phosphorus", 0, 300)
potassium = st.number_input("Potassium", 0, 300)

if st.button("Recommend Fertilizer"):

    input_data = pd.DataFrame([{
        "Soil_color": soil,
        "Nitrogen": nitrogen,
        "Phosphorus": phosphorus,
        "Potassium": potassium,
        "Crop": crop
    }])

    prediction = model.predict(input_data)[0]

    st.success(f"🌾 Recommended Fertilizer: {prediction}")
