import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from model_utils import predict_sales, get_shap_values

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="Nike Sales Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- DARK THEME ----
st.markdown(
    """
    <style>
    body { background-color: #0E1117; color: #FAFAFA; }
    .stButton>button { background-color: #FF6F00; color: white; border-radius: 10px; }
    .stSidebar { background-color: #1E1E1E; color: white; }
    .stMarkdown { color: #FAFAFA; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Nike Sales Prediction Dashboard ")

# ---- SIDEBAR INPUTS ----
st.sidebar.header("Input Features")
sale_price = st.sidebar.number_input("Sale Price", min_value=0, value=0)
rating = st.sidebar.slider("Rating", 1, 5, value=3)
review_posted = st.sidebar.number_input("Reviews Posted", min_value=0, value=0)
sold_products = st.sidebar.number_input("Sold Products", min_value=0, value=0)

# ---- PREDICTION & SHAP ----
if st.button("Predict Sales"):
    features = np.array([sale_price, rating, review_posted, sold_products]).reshape(1, -1)

    # Predict sales
    prediction = predict_sales(sale_price, rating, review_posted, sold_products)
    st.success(f"Predicted Future Sales: {prediction}")

    # Get SHAP values
    shap_values, explainer = get_shap_values(features)

    st.subheader("Feature Importance (SHAP)")

    # For a single prediction, waterfall plot is better for local explanation
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], feature_names=["Sale Price", "Rating", "Reviews Posted", "Sold Products"], show=False)
    st.pyplot(fig)

# ---- METRICS TABLE ----
st.markdown("---")
st.subheader("Input Metrics")
metrics_df = pd.DataFrame({
    "Feature": ["Sale Price", "Rating", "Reviews Posted", "Sold Products"],
    "Value": [sale_price, rating, review_posted, sold_products]
})
st.table(metrics_df)
