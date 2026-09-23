
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "model" / "mumbai_house_price_lgbm.joblib"
META_PATH = BASE / "model" / "metadata.json"
GEO_PATH = BASE / "model" / "locality_geo.json"

st.set_page_config(
    page_title="Mumbai House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    geo = json.loads(GEO_PATH.read_text(encoding="utf-8"))
    return model, metadata, geo

model, meta, geo = load_artifacts()

def money_inr(value: float) -> str:
    value = float(value)
    if value >= 1e7:
        return f"₹{value / 1e7:.2f} Cr"
    if value >= 1e5:
        return f"₹{value / 1e5:.2f} Lakh"
    return f"₹{value:,.0f}"

def indian_number(value: float) -> str:
    return f"₹{value:,.0f}"

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero {
    padding: 1.5rem 1.7rem;
    border-radius: 18px;
    border: 1px solid rgba(128,128,128,.25);
    margin-bottom: 1.2rem;
}
.result {
    padding: 1.5rem;
    border-radius: 18px;
    border: 1px solid rgba(128,128,128,.25);
    text-align: center;
}
.price {font-size: 2.6rem; font-weight: 800; margin: .3rem 0;}
.muted {opacity: .72;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>🏠 Mumbai House Price Predictor</h1>
<p class="muted">Estimate a property's market price using a LightGBM model trained on Mumbai real-estate listings.</p>
</div>
""", unsafe_allow_html=True)

ranges = meta["ranges"]
localities = meta["localities"]
property_types = meta["property_types"]
furnished_options = meta["furnished_options"]

with st.form("prediction_form"):
    st.subheader("📍 Location")
    locality = st.selectbox("Locality", localities, index=localities.index("Bandra") if "Bandra" in localities else 0)

    st.subheader("🏡 Property specifications")
    c1, c2, c3 = st.columns(3)

    with c1:
        area = st.number_input(
            "Area (sqft)",
            min_value=float(ranges["area"][0]),
            max_value=float(ranges["area"][1]),
            value=1000.0,
            step=50.0,
        )
        bhk = st.slider(
            "Bedrooms (BHK)",
            min_value=max(1, ranges["bedroom_num"][0]),
            max_value=ranges["bedroom_num"][1],
            value=2,
        )

    with c2:
        bathrooms = st.slider(
            "Bathrooms",
            min_value=max(1, ranges["bathroom_num"][0]),
            max_value=ranges["bathroom_num"][1],
            value=2,
        )
        balconies = st.slider(
            "Balconies",
            min_value=max(0, ranges["balcony_num"][0]),
            max_value=ranges["balcony_num"][1],
            value=1,
        )

    with c3:
        age = st.slider(
            "Property age (years)",
            min_value=0,
            max_value=ranges["age"][1],
            value=5,
        )
        total_floors = st.slider(
            "Total floors",
            min_value=1,
            max_value=max(1, ranges["total_floors"][1]),
            value=min(10, max(1, ranges["total_floors"][1])),
        )

    c4, c5 = st.columns(2)
    with c4:
        furnished = st.selectbox("Furnishing", furnished_options)
    with c5:
        property_type = st.selectbox("Property type", property_types)

    submitted = st.form_submit_button("🔮 Predict House Price", use_container_width=True)

if submitted:
    if bathrooms > bhk + 4:
        st.warning("The bathroom count is unusually high compared with the selected BHK. You can still predict, but the estimate may be less reliable.")

    location = geo.get(locality, {"latitude": 19.0760, "longitude": 72.8777})

    row = pd.DataFrame([{
        "area": area,
        "locality": locality,
        "property_type": property_type,
        "bedroom_num": bhk,
        "bathroom_num": bathrooms,
        "balcony_num": balconies,
        "furnished": furnished,
        "age": age,
        "total_floors": total_floors,
        "latitude": location["latitude"],
        "longitude": location["longitude"],
    }])

    for col in meta["categorical_features"]:
        row[col] = pd.Categorical(row[col], categories=meta["categories"][col])

    predicted_log = float(model.predict(row)[0])
    predicted_price = max(0.0, float(np.expm1(predicted_log)))
    price_sqft = predicted_price / area

    st.markdown("---")
    st.subheader("📊 Prediction")

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f'<div class="result"><div class="muted">Estimated price</div><div class="price">{money_inr(predicted_price)}</div></div>', unsafe_allow_html=True)
    with r2:
        st.metric("Estimated price / sqft", money_inr(price_sqft))
    with r3:
        st.metric("Selected locality", locality)

    st.caption(
        "This is a machine-learning estimate, not a property valuation or guaranteed market price. "
        "Actual prices can vary with exact building, road, floor, view, amenities, legal status and market conditions."
    )

with st.expander("ℹ️ About this model"):
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Training records", f"{meta['dataset_rows']:,}")
    with m2:
        st.metric("Holdout log RMSE", f"{meta['metrics']['cv_or_holdout_rmse_log']:.3f}")
    with m3:
        st.metric("Holdout R²", f"{meta['metrics']['holdout_r2']:.3f}")
    st.write(
        "The model uses area, locality, property type, BHK, bathrooms, balconies, "
        "furnishing, property age, total floors and locality-level geographic coordinates. "
        "Price-per-square-foot is intentionally excluded as an input because it is derived from price "
        "and would create target leakage."
    )
