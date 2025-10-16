import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

# Import your model and background from your model_utils or wherever
from model_utils import model, background  # assuming these are defined

# Predict function using your model
def predict_sales(sale_price, rating, review_posted, sold_products):
    features = np.array([[sale_price, rating, review_posted, sold_products]])
    prediction = model.predict(features)[0]  # adjust if needed for your model
    return prediction

# SHAP values function
def get_shap_values(features):
    # Use the right SHAP explainer for your model:
    # e.g. shap.TreeExplainer(model), shap.KernelExplainer(model.predict, background), or shap.Explainer(model)
    
    # Example for tree-based models:
    explainer = shap.Explainer(model, background)  # change if your model type requires a different explainer
    
    shap_values = explainer(features)
    return shap_values.values, explainer

# Streamlit UI
st.set_page_config(
    page_title="Nike Sales Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark theme CSS
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

st.title("Nike Sales Prediction Dashboard")

# Sidebar inputs
st.sidebar.header("Input Features")
sale_price = st.sidebar.number_input("Sale Price", min_value=0.0, value=10.0)
rating = st.sidebar.slider("Rating", 1, 5, value=3)
review_posted = st.sidebar.number_input("Reviews Posted", min_value=0, value=0)
sold_products = st.sidebar.number_input("Sold Products", min_value=0, value=0)

# Prediction & SHAP on button click
if st.button("Predict Sales"):
    features = np.array([[sale_price, rating, review_posted, sold_products]])

    prediction = predict_sales(sale_price, rating, review_posted, sold_products)
    st.success(f"Predicted Future Sales: {prediction:.2f}")

    shap_values, explainer = get_shap_values(features)

    st.subheader("Feature Importance (SHAP)")

    # Plot waterfall for the single prediction
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value, shap_values[0],
        feature_names=["Sale Price", "Rating", "Reviews Posted", "Sold Products"],
        show=False
    )
    st.pyplot(fig)

# Display input metrics
st.markdown("---")
st.subheader("Input Metrics")
metrics_df = pd.DataFrame({
    "Feature": ["Sale Price", "Rating", "Reviews Posted", "Sold Products"],
    "Value": [sale_price, rating, review_posted, sold_products]
})
st.table(metrics_df)
