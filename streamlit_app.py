import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

# -------- Dummy Model and Background Data --------

class DummyModel:
    def predict(self, X):
        # Simple dummy prediction: sum of features * 2
        return np.sum(X, axis=1) * 2

# Dummy model instance
model = DummyModel()

# Background data for SHAP (e.g., representative sample of training data)
background = np.array([
    [10, 3, 5, 100],
    [20, 4, 2, 150],
    [15, 5, 10, 80]
])

# -------- Prediction and SHAP functions --------

def predict_sales(sale_price, rating, review_posted, sold_products):
    features = np.array([[sale_price, rating, review_posted, sold_products]])
    pred = model.predict(features)[0]
    return pred

def get_shap_values(features):
    # Use KernelExplainer (works with any model)
    explainer = shap.KernelExplainer(model.predict, background)
    shap_values = explainer.shap_values(features)
    return shap_values, explainer

# -------- Streamlit app --------

st.set_page_config(
    page_title="Nike Sales Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    body { background-color: #0E1117; color: #FAFAFA; }
    .stButton>button { background-color: #FF6F00; color: white; border-radius: 10px; }
    .stSidebar { background-color: #1E1E1E; color: white; }
    .stMarkdown { color: #FAFAFA; }
    </style>
    """, unsafe_allow_html=True)

st.title("Nike Sales Prediction Dashboard")

# Sidebar inputs
st.sidebar.header("Input Features")
sale_price = st.sidebar.number_input("Sale Price", min_value=0.0, value=10.0)
rating = st.sidebar.slider("Rating", 1, 5, value=3)
review_posted = st.sidebar.number_input("Reviews Posted", min_value=0, value=0)
sold_products = st.sidebar.number_input("Sold Products", min_value=0, value=0)

# Predict button
if st.button("Predict Sales"):
    features = np.array([[sale_price, rating, review_posted, sold_products]])

    # Predict sales
    prediction = predict_sales(sale_price, rating, review_posted, sold_products)
    st.success(f"Predicted Future Sales: {prediction:.2f}")

    # Calculate SHAP values
    shap_values, explainer = get_shap_values(features)

    st.subheader("Feature Importance (SHAP)")

    # Plot waterfall for local explanation
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value, shap_values[0],
        feature_names=["Sale Price", "Rating", "Reviews Posted", "Sold Products"],
        show=False
    )
    st.pyplot(fig)

# Display input metrics table
st.markdown("---")
st.subheader("Input Metrics")
metrics_df = pd.DataFrame({
    "Feature": ["Sale Price", "Rating", "Reviews Posted", "Sold Products"],
    "Value": [sale_price, rating, review_posted, sold_products]
})
st.table(metrics_df)
