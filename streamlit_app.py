import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

# Import your model and background from your model_utils
try:
    from model_utils import model, background  # ensure these exist and are loaded properly
except ImportError:
    st.error("model_utils.py must define 'model' and 'background' variables.")
    st.stop()

FEATURE_NAMES = ["Sale Price", "Rating", "Reviews Posted", "Sold Products"]

# Predict function using your model
def predict_sales(sale_price, rating, review_posted, sold_products):
    features = np.array([[sale_price, rating, review_posted, sold_products]])
    prediction = model.predict(features)[0]  # Adjust if your model returns differently
    return prediction

# Cache the SHAP explainer for faster repeated use
@st.cache_resource
def get_shap_explainer():
    # Use shap.Explainer (auto selects best explainer for your model)
    return shap.Explainer(model, background)

# Calculate SHAP values for given features
def get_shap_values(features, explainer):
    shap_values = explainer(features)
    return shap_values.values, explainer

# --- Streamlit UI ---
st.set_page_config(
    page_title="Nike Sales Prediction Dashboard",
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
sale_price = st.sidebar.number_input("Sale Price", min_value=0.0, value=10.0, format="%.2f")
rating = st.sidebar.slider("Rating", 1, 5, value=3)
review_posted = st.sidebar.number_input("Reviews Posted", min_value=0, value=0, step=1)
sold_products = st.sidebar.number_input("Sold Products", min_value=0, value=0, step=1)

if st.button("Predict Sales"):
    # Validate inputs (optional)
    if sale_price < 0 or rating < 1 or rating > 5 or review_posted < 0 or sold_products < 0:
        st.error("Please enter valid input values.")
    else:
        features = np.array([[sale_price, rating, review_posted, sold_products]])

        # Prediction
        prediction = predict_sales(sale_price, rating, review_posted, sold_products)
        st.success(f"Predicted Future Sales: {prediction:.2f}")

        # SHAP explanation
        explainer = get_shap_explainer()
        shap_values, _ = get_shap_values(features, explainer)

        st.subheader("Feature Importance (SHAP)")

        # Plot SHAP waterfall plot using public API
        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(shap.Explanation(values=shap_values[0],
                                              base_values=explainer.expected_value,
                                              data=features[0],
                                              feature_names=FEATURE_NAMES),
                            max_display=len(FEATURE_NAMES), show=False)
        st.pyplot(fig)

# Display input metrics table
st.markdown("---")
st.subheader("Input Metrics")
metrics_df = pd.DataFrame({
    "Feature": FEATURE_NAMES,
    "Value": [sale_price, rating, review_posted, sold_products]
})
st.table(metrics_df)
