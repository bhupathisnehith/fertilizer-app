import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score
import warnings
import io
import datetime

warnings.filterwarnings("ignore")

# ─────────────────────────── PAGE CONFIG ───────────────────────────
st.set_page_config(
    page_title="Smart Agriculture AI System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────── CUSTOM CSS ────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    .metric-card {
        background: linear-gradient(135deg, #1a3c2e 0%, #2d6a4f 100%);
        border-radius: 12px; padding: 18px; color: white;
        text-align: center; margin: 6px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    .metric-card h2 { font-size: 2rem; margin: 0; font-weight: 700; }
    .metric-card p  { font-size: 0.85rem; margin: 0; opacity: 0.85; }
    .alert-warn {
        background: #fff3cd; border-left: 4px solid #ffc107;
        padding: 10px 14px; border-radius: 6px; color: #856404; margin: 6px 0;
    }
    .alert-ok {
        background: #d4edda; border-left: 4px solid #28a745;
        padding: 10px 14px; border-radius: 6px; color: #155724; margin: 6px 0;
    }
    .fert-badge {
        display:inline-block; background:#2d6a4f; color:white;
        padding:4px 14px; border-radius:20px; font-weight:600; font-size:0.9rem;
    }
    .section-header {
        font-size: 1.2rem; font-weight: 700; color: #1a3c2e;
        border-bottom: 2px solid #2d6a4f; padding-bottom: 6px; margin: 20px 0 12px 0;
    }
    div[data-testid="stTabs"] button {
        font-family: 'Outfit', sans-serif; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── CONSTANTS ─────────────────────────────

FERTILIZER_PRICES = {
    "Urea": 6, "DAP": 25, "MOP": 15,
    "10:26:26 NPK": 22, "19:19:19 NPK": 24,
    "13:32:26 NPK": 23, "12:32:16 NPK": 21,
    "50:26:26 NPK": 28, "20:20:20 NPK": 26,
    "NPK": 20,
}

CROP_PRICES = {
    "Rice": 20, "Wheat": 18, "Maize": 15, "Sugarcane": 3,
    "Jowar": 22, "Cotton": 60, "Groundnut": 55, "Tur": 70,
    "Urad": 65, "Moong": 72, "Gram": 50, "Masoor": 55,
    "Soybean": 40, "Ginger": 120, "Turmeric": 110, "Grapes": 80,
}

# Agronomic base yields (quintals/acre) per crop
CROP_BASE_YIELDS = {
    "Rice": 20, "Wheat": 18, "Maize": 22, "Sugarcane": 350,
    "Jowar": 12, "Cotton": 8, "Groundnut": 10, "Tur": 7,
    "Urad": 6, "Moong": 6, "Gram": 8, "Masoor": 6,
    "Soybean": 10, "Ginger": 40, "Turmeric": 35, "Grapes": 60,
}

# Optimal NPK targets per crop (N, P, K in kg/ha → approximate)
CROP_NPK_OPTIMAL = {
    "Rice":      (120, 60, 60), "Wheat":     (120, 60, 40),
    "Maize":     (150, 75, 40), "Sugarcane": (250, 115, 115),
    "Jowar":     (80, 40, 40),  "Cotton":    (120, 60, 60),
    "Groundnut": (25, 50, 50),  "Tur":       (20, 50, 0),
    "Urad":      (20, 40, 40),  "Moong":     (20, 40, 40),
    "Gram":      (20, 40, 0),   "Masoor":    (20, 40, 0),
    "Soybean":   (30, 60, 40),  "Ginger":    (80, 50, 120),
    "Turmeric":  (120, 60, 120),"Grapes":    (80, 40, 80),
}

RAIN_FACTOR = {
    "Sunny": 0.80, "Rainy": 1.20, "Cloudy": 1.00,
    "Overcast": 1.10, "Windy": 0.90, "Snowy": 0.70,
    "Foggy/Mist": 0.85, "Thunderstorms": 1.15, "Sandstorms": 0.60,
}

NPK_THRESHOLDS = {
    "Nitrogen":   {"low": 40,  "high": 200},
    "Phosphorus": {"low": 30,  "high": 120},
    "Potassium":  {"low": 30,  "high": 120},
}

# ─────────────────────────── LOAD DATA ─────────────────────────────

@st.cache_data
def load_data():
    df = pd.read_csv("Crop and fertilizer dataset.csv")
    df["Soil_color"] = df["Soil_color"].str.strip()
    df["Crop"]       = df["Crop"].str.strip()
    df["Fertilizer"] = df["Fertilizer"].str.strip()
    df = df.drop(columns=["Link"], errors="ignore")
    df = df.dropna()
    return df

df = load_data()

# ─────────────────────────── BUILD MODEL ───────────────────────────

@st.cache_resource
def build_model(df):
    feature_cols = ["District_Name", "Soil_color", "Nitrogen", "Phosphorus",
                    "Potassium", "pH", "Rainfall", "Temperature", "Crop"]
    X = df[feature_cols]
    y = df["Fertilizer"]

    cat_cols = ["District_Name", "Soil_color", "Crop"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", "passthrough", num_cols),
    ])

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=15,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )

    pipeline = Pipeline([("prep", preprocessor), ("clf", clf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    # Feature importances (from RF)
    ohe_features  = pipeline.named_steps["prep"].transformers_[0][1].get_feature_names_out(cat_cols)
    all_features   = list(ohe_features) + num_cols
    importances    = pipeline.named_steps["clf"].feature_importances_
    feat_imp       = pd.Series(importances, index=all_features).sort_values(ascending=False)

    return pipeline, acc, feat_imp, X_test, y_test

model, model_accuracy, feat_imp, X_test, y_test = build_model(df)

# ─────────────────────────── SESSION STATE ─────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []

# ─────────────────────────── SIDEBAR ───────────────────────────────

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1147/1147809.png", width=70)
    st.title("🌾 Smart Agriculture AI System")
    st.markdown("---")


    st.markdown(f"""
    <div class="metric-card">
        <p>Training Records</p>
        <h2>{len(df):,}</h2>
        <p>{df['Crop'].nunique()} crops · {df['Fertilizer'].nunique()} fertilizers</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-card">
        <p>Predictions Made</p>
        <h2>{len(st.session_state.history)}</h2>
        <p>This session</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Columns used: District, Soil, N, P, K, pH, Rainfall, Temp, Crop")

# ─────────────────────────── TABS ──────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    "🔬 Predict & Analyze",
    "📊 EDA Explorer",
    "📋 Prediction History"
])

# ══════════════════════════════════════════════════════
# TAB 1 – PREDICT & ANALYZE
# ══════════════════════════════════════════════════════
with tab1:
    st.markdown("## 🔬 Fertilizer Prediction & Farm Analysis")

    # ── INPUT FORM ──
    with st.expander("⚙️ Input Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<p class="section-header">📍 Location & Soil</p>', unsafe_allow_html=True)
            district   = st.selectbox("District", sorted(df["District_Name"].unique()))
            soil_color = st.selectbox("Soil Color", sorted(df["Soil_color"].unique()))
            pH_val     = st.number_input("Soil pH", 4.0, 9.0, 6.5, 0.1)

        with col2:
            st.markdown('<p class="section-header">🧪 NPK Levels (kg/ha)</p>', unsafe_allow_html=True)
            nitrogen   = st.number_input("Nitrogen (N)", 0, 300, 80)
            phosphorus = st.number_input("Phosphorus (P)", 0, 300, 50)
            potassium  = st.number_input("Potassium (K)", 0, 300, 50)

        with col3:
            st.markdown('<p class="section-header">🌦️ Environment & Crop</p>', unsafe_allow_html=True)
            crop        = st.selectbox("Crop", sorted(df["Crop"].unique()))
            rainfall    = st.number_input("Rainfall (mm)", 0, 3000, 800)
            temperature = st.number_input("Temperature (°C)", 5, 50, 25)
            weather     = st.selectbox("Current Weather", list(RAIN_FACTOR.keys()))

        # Editable prices
        with st.expander("💰 Edit Market Prices"):
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                fert_price_override = st.number_input(
                    "Override Fertilizer Price (₹/kg, 0 = auto)", 0, 200, 0
                )
            with p_col2:
                crop_price_override = st.number_input(
                    "Override Crop Price (₹/quintal, 0 = auto)", 0, 1000, 0
                )

    # ── INPUT VALIDATION ──
    st.markdown('<p class="section-header">🩺 Soil Health Diagnostics</p>', unsafe_allow_html=True)
    warn_col1, warn_col2, warn_col3 = st.columns(3)

    npk_status = {}
    for nutrient, val, col in [
        ("Nitrogen", nitrogen, warn_col1),
        ("Phosphorus", phosphorus, warn_col2),
        ("Potassium", potassium, warn_col3),
    ]:
        low  = NPK_THRESHOLDS[nutrient]["low"]
        high = NPK_THRESHOLDS[nutrient]["high"]
        with col:
            if val < low:
                st.markdown(f'<div class="alert-warn">⚠️ <b>{nutrient}</b> is LOW ({val})<br><small>Optimal ≥ {low} kg/ha</small></div>', unsafe_allow_html=True)
                npk_status[nutrient] = "low"
            elif val > high:
                st.markdown(f'<div class="alert-warn">⚠️ <b>{nutrient}</b> is HIGH ({val})<br><small>Toxicity risk above {high}</small></div>', unsafe_allow_html=True)
                npk_status[nutrient] = "high"
            else:
                st.markdown(f'<div class="alert-ok">✅ <b>{nutrient}</b> is OK ({val})<br><small>Within optimal range</small></div>', unsafe_allow_html=True)
                npk_status[nutrient] = "ok"

    # pH warning
    if pH_val < 5.5:
        st.warning("⚠️ pH is very acidic — consider lime application before fertilizing.")
    elif pH_val > 8.0:
        st.warning("⚠️ pH is alkaline — phosphorus and micronutrient availability may be reduced.")

    # ── SOIL HEALTH SCORE ──
    def soil_health_score(n, p, k, ph):
        scores = []
        for val, lo, hi in [(n, 40, 150), (p, 30, 80), (k, 30, 100)]:
            if lo <= val <= hi:
                scores.append(1.0)
            elif val < lo:
                scores.append(max(0, val / lo))
            else:
                scores.append(max(0, 1 - (val - hi) / hi))
        ph_score = 1.0 if 6.0 <= ph <= 7.5 else max(0, 1 - abs(ph - 6.75) / 3)
        scores.append(ph_score)
        return round(np.mean(scores) * 100, 1)

    shs = soil_health_score(nitrogen, phosphorus, potassium, pH_val)
    shs_color = "#28a745" if shs >= 70 else "#ffc107" if shs >= 40 else "#dc3545"

    st.markdown(f"""
    <div style="background:#f8f9fa;border-radius:10px;padding:12px;margin:10px 0;border-left:5px solid {shs_color}">
        <b>🌱 Soil Health Score:</b>
        <span style="font-size:1.6rem;font-weight:700;color:{shs_color}"> {shs}/100</span>
        <span style="font-size:0.85rem;color:#666;"> — {'Excellent' if shs>=70 else 'Moderate' if shs>=40 else 'Poor'} soil condition</span>
    </div>
    """, unsafe_allow_html=True)

    # ── PREDICT BUTTON ──
    st.markdown("---")
    run_btn = st.button("🚀 Predict Fertilizer", type="primary", use_container_width=True)

    if run_btn:
        input_df = pd.DataFrame([{
            "District_Name": district,
            "Soil_color":    soil_color,
            "Nitrogen":      nitrogen,
            "Phosphorus":    phosphorus,
            "Potassium":     potassium,
            "pH":            pH_val,
            "Rainfall":      rainfall,
            "Temperature":   temperature,
            "Crop":          crop,
        }])

        prediction  = model.predict(input_df)[0]
        proba       = model.predict_proba(input_df)[0]
        classes     = model.classes_
        top3_idx    = np.argsort(proba)[::-1][:3]
        top3        = [(classes[i], round(proba[i]*100, 1)) for i in top3_idx]
        confidence  = top3[0][1]

        # ── RESULTS ──
        r1, r2 = st.columns([2, 1])

        with r1:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1a3c2e,#2d6a4f);border-radius:14px;
                        padding:24px;color:white;margin:10px 0;">
                <div style="font-size:0.9rem;opacity:0.8">🏆 Primary Recommendation</div>
                <div style="font-size:2.2rem;font-weight:700;margin:6px 0">{prediction}</div>
                <div style="background:rgba(255,255,255,0.2);border-radius:20px;height:10px;margin:8px 0">
                    <div style="background:#74c69d;width:{confidence}%;height:10px;border-radius:20px"></div>
                </div>
                <div style="font-size:0.85rem">Confidence: {confidence}%</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<p class="section-header">🏅 Top 3 Alternatives</p>', unsafe_allow_html=True)
            for rank, (fert, prob) in enumerate(top3):
                color = "#2d6a4f" if rank == 0 else "#52b788" if rank == 1 else "#95d5b2"
                st.markdown(f"""
                <div style="margin:6px 0;display:flex;align-items:center;gap:12px">
                    <span style="background:{color};color:white;width:26px;height:26px;border-radius:50%;
                                 display:inline-flex;align-items:center;justify-content:center;
                                 font-weight:700;font-size:0.8rem;flex-shrink:0">#{rank+1}</span>
                    <span style="flex:1">{fert}</span>
                    <div style="width:160px;background:#eee;border-radius:10px;height:8px">
                        <div style="background:{color};width:{prob}%;height:8px;border-radius:10px"></div>
                    </div>
                    <span style="min-width:42px;text-align:right;font-weight:600">{prob}%</span>
                </div>
                """, unsafe_allow_html=True)

        with r2:
            # NPK Radar Chart
            fig, ax = plt.subplots(figsize=(3.5, 3.5), subplot_kw=dict(polar=True))
            opt = CROP_NPK_OPTIMAL.get(crop, (100, 60, 60))
            categories = ['Nitrogen', 'Phosphorus', 'Potassium']
            actual_vals = [nitrogen / opt[0], phosphorus / opt[1], potassium / opt[2]]
            actual_vals_plot = actual_vals + [actual_vals[0]]
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]
            ax.plot(angles, actual_vals_plot, 'o-', color='#2d6a4f', linewidth=2)
            ax.fill(angles, actual_vals_plot, alpha=0.3, color='#52b788')
            ax.plot(angles, [1]*4, '--', color='#ffc107', linewidth=1.2, alpha=0.7, label='Optimal')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=9)
            ax.set_ylim(0, 2)
            ax.set_title("NPK vs Optimal\n(ratio)", fontsize=9, pad=14)
            ax.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.3, 1.1))
            st.pyplot(fig, use_container_width=True)

        # ── QUANTITY & COST ──
        st.markdown('<p class="section-header">📦 Fertilizer Dosage & Cost</p>', unsafe_allow_html=True)

        # NPK gap-based quantity
        opt = CROP_NPK_OPTIMAL.get(crop, (100, 60, 60))
        gap_n = max(0, opt[0] - nitrogen)
        gap_p = max(0, opt[1] - phosphorus)
        gap_k = max(0, opt[2] - potassium)
        total_gap = gap_n + gap_p + gap_k
        qty = round(max(10, total_gap / 3 / 2.471), 2)  # convert kg/ha to kg/acre

        fert_price = fert_price_override if fert_price_override > 0 else FERTILIZER_PRICES.get(prediction, 20)
        cost = round(qty * fert_price, 2)

        qc1, qc2, qc3 = st.columns(3)
        with qc1:
            st.metric("Fertilizer Needed", f"{qty} kg/acre")
        with qc2:
            st.metric("Price per kg", f"₹{fert_price}")
        with qc3:
            st.metric("Total Cost", f"₹{cost}")

        if total_gap == 0:
            st.info("✅ Soil NPK levels meet crop optimal requirements. Minimum maintenance dose recommended.")

        # ── YIELD & PROFIT ──
        st.markdown('<p class="section-header">📈 Yield & Profit Estimation</p>', unsafe_allow_html=True)

        rain_factor = RAIN_FACTOR.get(weather, 1.0)
        base_yield  = CROP_BASE_YIELDS.get(crop, 20)
        npk_ratio   = min(1.3, ((nitrogen + phosphorus + potassium) / (sum(CROP_NPK_OPTIMAL.get(crop, (100,60,60))))))
        yield_est   = round(base_yield * npk_ratio * rain_factor, 2)

        c_price   = crop_price_override if crop_price_override > 0 else CROP_PRICES.get(crop, 20)
        revenue   = round(yield_est * c_price, 2)
        profit    = round(revenue - cost, 2)
        roi       = round((profit / cost) * 100, 1) if cost > 0 else 0

        yc1, yc2, yc3, yc4 = st.columns(4)
        with yc1:
            st.metric("Est. Yield", f"{yield_est} q/acre")
        with yc2:
            st.metric("Revenue", f"₹{revenue:,}")
        with yc3:
            st.metric("Profit", f"₹{profit:,}", delta=f"ROI {roi}%")
        with yc4:
            st.metric("Crop Price", f"₹{c_price}/quintal")

        # ── AI INSIGHTS ──
        st.markdown('<p class="section-header">🤖 AI Agronomic Insights</p>', unsafe_allow_html=True)
        insights = []

        if nitrogen < NPK_THRESHOLDS["Nitrogen"]["low"]:
            insights.append(("🌿", "Low Nitrogen", "Apply Urea or DAP at early growth stage to boost leaf & shoot development."))
        if phosphorus < NPK_THRESHOLDS["Phosphorus"]["low"]:
            insights.append(("🌱", "Low Phosphorus", "DAP or SSP at planting helps root establishment and early flowering."))
        if potassium < NPK_THRESHOLDS["Potassium"]["low"]:
            insights.append(("🛡️", "Low Potassium", "MOP application improves disease resistance and water uptake."))
        if pH_val < 5.5:
            insights.append(("🪨", "Acidic Soil", "Apply agricultural lime (2–3 t/ha) to raise pH before fertilizing."))
        if pH_val > 8.0:
            insights.append(("🌊", "Alkaline Soil", "Use acidifying fertilizers like Ammonium Sulfate; check micronutrients."))
        if weather == "Rainy":
            insights.append(("🌧️", "Rainy Weather", "Split fertilizer doses to reduce leaching. Apply top-dressing after rain."))
        if weather == "Sunny" and temperature > 38:
            insights.append(("☀️", "Heat Stress", "Irrigate before fertilizing to prevent fertilizer burn under high temperatures."))
        if not insights:
            insights.append(("✅", "Balanced Inputs", "Soil parameters look good. Maintain current nutrient management schedule."))

        for icon, title, desc in insights:
            st.markdown(f"""
            <div style="background:#f0fdf4;border-left:4px solid #2d6a4f;padding:10px 14px;
                        border-radius:6px;margin:6px 0">
                <b>{icon} {title}</b><br>
                <span style="font-size:0.88rem;color:#444">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

        # ── DOWNLOAD REPORT ──
        st.markdown("---")
        report_lines = [
            "=" * 50,
            "   SMART AGRI AI — PREDICTION REPORT",
            f"   Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 50,
            f"District   : {district}",
            f"Soil Color : {soil_color}",
            f"Crop       : {crop}",
            f"Weather    : {weather}",
            "",
            "── Soil Inputs ──",
            f"  N  : {nitrogen} kg/ha",
            f"  P  : {phosphorus} kg/ha",
            f"  K  : {potassium} kg/ha",
            f"  pH : {pH_val}",
            f"  Rainfall    : {rainfall} mm",
            f"  Temperature : {temperature} °C",
            "",
            "── Prediction ──",
            f"  Recommended Fertilizer : {prediction}",
            f"  Confidence             : {confidence}%",
            f"  Soil Health Score      : {shs}/100",
            "",
            "── Alternatives ──",
        ] + [f"  #{r+1} {f} ({p}%)" for r, (f, p) in enumerate(top3)] + [
            "",
            "── Dosage & Economics ──",
            f"  Fertilizer Needed : {qty} kg/acre",
            f"  Fertilizer Cost   : ₹{cost}",
            f"  Estimated Yield   : {yield_est} q/acre",
            f"  Revenue           : ₹{revenue}",
            f"  Profit            : ₹{profit}  (ROI {roi}%)",
            "",
            "── Insights ──",
        ] + [f"  • {t}: {d}" for _, t, d in insights] + ["", "=" * 50]

        report_text = "\n".join(report_lines)
        st.download_button(
            "📥 Download Report (.txt)",
            data=report_text,
            file_name=f"agri_report_{crop}_{datetime.date.today()}.txt",
            mime="text/plain",
        )

        # ── SAVE TO HISTORY ──
        st.session_state.history.append({
            "Time":       datetime.datetime.now().strftime("%H:%M:%S"),
            "District":   district,
            "Crop":       crop,
            "Fertilizer": prediction,
            "Conf%":      confidence,
            "Qty(kg)":    qty,
            "Cost(₹)":    cost,
            "Yield(q)":   yield_est,
            "Profit(₹)":  profit,
        })

# ══════════════════════════════════════════════════════
# TAB 2 – EDA EXPLORER
# ══════════════════════════════════════════════════════
with tab2:
    st.markdown("## 📊 Dataset Exploration")

    eda1, eda2 = st.columns(2)

    with eda1:
        st.markdown("**Fertilizer Distribution**")
        fert_counts = df["Fertilizer"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.barh(fert_counts.index, fert_counts.values,
                       color=plt.cm.YlGn(np.linspace(0.4, 0.9, len(fert_counts))))
        ax.set_xlabel("Count")
        ax.set_title("Fertilizers in Dataset")
        for bar, val in zip(bars, fert_counts.values):
            ax.text(val + 5, bar.get_y() + bar.get_height()/2, str(val), va='center', fontsize=8)
        st.pyplot(fig, use_container_width=True)

    with eda2:
        st.markdown("**Crop Distribution**")
        crop_counts = df["Crop"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.bar(crop_counts.index, crop_counts.values,
                color=plt.cm.BuGn(np.linspace(0.4, 0.9, len(crop_counts))))
        ax2.set_xticklabels(crop_counts.index, rotation=45, ha="right", fontsize=8)
        ax2.set_ylabel("Count")
        ax2.set_title("Crops in Dataset")
        st.pyplot(fig2, use_container_width=True)

    eda3, eda4 = st.columns(2)

    with eda3:
        st.markdown("**NPK Distribution by Fertilizer**")
        nutrient_sel = st.selectbox("Select Nutrient", ["Nitrogen", "Phosphorus", "Potassium"])
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        groups = [df[df["Fertilizer"] == f][nutrient_sel].values
                  for f in df["Fertilizer"].unique()]
        ax3.boxplot(groups, labels=df["Fertilizer"].unique(), patch_artist=True,
                    boxprops=dict(facecolor="#b7e4c7"))
        ax3.set_xticklabels(df["Fertilizer"].unique(), rotation=45, ha="right", fontsize=8)
        ax3.set_title(f"{nutrient_sel} by Fertilizer")
        st.pyplot(fig3, use_container_width=True)

    with eda4:
        st.markdown("**Soil Color vs Fertilizer (Heatmap)**")
        pivot = df.groupby(["Soil_color", "Fertilizer"]).size().unstack(fill_value=0)
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        im = ax4.imshow(pivot.values, cmap="YlGn", aspect="auto")
        ax4.set_xticks(range(len(pivot.columns)))
        ax4.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
        ax4.set_yticks(range(len(pivot.index)))
        ax4.set_yticklabels(pivot.index, fontsize=8)
        plt.colorbar(im, ax=ax4, label="Count")
        ax4.set_title("Soil Color × Fertilizer")
        st.pyplot(fig4, use_container_width=True)

    st.markdown("**Dataset Sample (filtered)**")
    filter_crop = st.selectbox("Filter by crop", ["All"] + sorted(df["Crop"].unique().tolist()))
    display_df  = df if filter_crop == "All" else df[df["Crop"] == filter_crop]
    st.dataframe(display_df.head(100), use_container_width=True)
    st.caption(f"Showing up to 100 of {len(display_df):,} records.")

# ══════════════════════════════════════════════════════
# TAB 3 – HISTORY
# ══════════════════════════════════════════════════════
with tab4:
    st.markdown("## 📋 Prediction History")

    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)

        csv = history_df.to_csv(index=False)
        st.download_button(
            "📥 Download History CSV",
            data=csv,
            file_name=f"prediction_history_{datetime.date.today()}.csv",
            mime="text/csv",
        )

        if len(history_df) > 1:
            st.markdown("**Profit trend across predictions**")
            fig_h, ax_h = plt.subplots(figsize=(8, 3))
            ax_h.plot(range(1, len(history_df)+1), history_df["Profit(₹)"],
                      marker="o", color="#2d6a4f", linewidth=2)
            ax_h.set_xlabel("Prediction #")
            ax_h.set_ylabel("Profit (₹)")
            ax_h.set_title("Profit Across Predictions")
            ax_h.axhline(0, color="red", linewidth=0.8, linestyle="--")
            st.pyplot(fig_h, use_container_width=True)
    else:
        st.info("No predictions yet. Run a prediction in the first tab to see history here.")

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()
