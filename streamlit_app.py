import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib  # for loading model if needed

# ---- MODEL & BACKGROUND SETUP ----
# Replace with your own model loading logic
# model = joblib.load("model.pkl")
# background = pd.read_csv("background_sample.csv").values

# For demo purposes, we'll define dummy model and background:
class DummyModel:
    def predict(self, X):
        # Dummy prediction: sum of features times 2
        return np.sum(X, axis=1) * 2

model = DummyModel()
background = np.array([
    [10.0, 3, 5, 100],
    [20.0, 4, 2, 150],
    [15.0, 5, 10, 80]
])

def predict_sales(sale_price, rating, review_posted, sold_products):
    # Dummy prediction function matching the dummy model
    features = np.array([[sale_price, rating, review_posted, sold_products]])
    return model.predict(features)[0]

def get_shap_values(features, model, background):
    # Use KernelExplainer here, since we don't know your model type
    explainer = shap.KernelExplainer(model.predict, background)
    shap_values = explainer.shap_values(features)
    return shap_values, explainer

# ---- STREAMLIT APP ----
st.set_page_config(
    page_title="Nike Sales Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
sale_price = st.sidebar.number_input("Sale Price", min_value=0.0, value=0.0)
rating = st.sidebar.slider("Rating", 1, 5, value=3)
review_posted = st.sidebar.number_input("Reviews Posted", min_value=0, value=0)
sold_products = st.sidebar.number_input("Sold Products", min_value=0, value=0)

if st.button("Predict Sales"):
    features = np.array([sale_price, rating, review_posted, sold_products]).reshape(1, -1)
    
    prediction = predict_sales(sale_price, rating, review_posted, sold_products)
    st.success(f"Predicted Future Sales: {prediction:.2f}")

    try:
        shap_values, explainer = get_shap_values(features, model, background)
        st.subheader("Feature Importance (SHAP)")
        st.markdown("This chart shows how each feature contributed to the prediction.")

        fig, ax = plt.subplots()
        shap.summary_plot(
            shap_values,
            features=features,
            feature_names=["Sale Price", "Rating", "Reviews Posted", "Sold Products"],
            plot_type="bar",
            show=False,
        )
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error calculating SHAP values: {e}")

# Metrics table
st.markdown("---")
st.subheader("Input Metrics")
metrics_df = pd.DataFrame({
    "Feature": ["Predicted Sales", "Rating", "Reviews Posted", "Sold Products"],
    "Value": [sale_price, rating, review_posted, sold_products]
})
st.table(metrics_df)

