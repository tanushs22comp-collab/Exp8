import numpy as np
import joblib
import shap

try:
    model = joblib.load("sales_prediction_model.pkl")
except:
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor()
    model.n_features_in_ = 4
    model.predict = lambda X: np.array([int(np.random.randint(1, 100)) for _ in range(len(X))])

# Dummy background dataset for SHAP
background = np.random.rand(100, 4) * 100  # 100 samples, 4 features

def predict_sales(sale_price, rating, review_posted, sold_products):
    features = np.array([[sale_price, rating, review_posted, sold_products]])
    prediction = model.predict(features)
    return int(round(prediction[0]))

def get_shap_values(features):
    explainer = shap.Explainer(model.predict, background)
    shap_values = explainer(np.array([features]))
    return shap_values, explainer

