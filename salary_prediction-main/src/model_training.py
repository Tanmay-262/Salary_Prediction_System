import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

def train_and_evaluate(X_train, y_train, X_val, y_val, model_dir='models'):
    """Train multiple models and evaluate them. Save the best model."""
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42, verbosity=0)
    }
    results = {}
    best_score = -np.inf
    best_model = None
    best_name = ''
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        rmse = mean_squared_error(y_val, preds, squared=False)
        r2 = r2_score(y_val, preds)
        results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        if r2 > best_score:
            best_score = r2
            best_model = model
            best_name = name
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, 'salary_model.pkl')
    joblib.dump(best_model, model_path)
    return results, best_name, model_path 