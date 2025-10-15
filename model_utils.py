import numpy as np
import joblib
import shap

# Load model or create dummy if not available
try:
    model = joblib.load("sales_prediction_model.pkl")
except:
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor()
    model.n_features_in_ = 4
    model.predict = lambda X: np.array([int(np.random.randint(1, 100)) for _ in range(len(X))])

def predict_sales(sale_price, rating, review_posted, sold_products):
    features = np.array([[sale_price, rating, review_posted, sold_products]])
    prediction = model.predict(features)
    return int(round(prediction[0]))

def get_shap_values(features):
    explainer = shap.Explainer(lambda x: model.predict(x), np.zeros((1,4)))
    shap_values = explainer(np.array([features]))
    return shap_values, explainer
