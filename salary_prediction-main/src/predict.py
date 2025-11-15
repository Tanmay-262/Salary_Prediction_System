import joblib
import pandas as pd
import os

def load_model(model_path):
    """Load the trained model from disk."""
    return joblib.load(model_path)

def predict_salary(model, input_df):
    """Predict salary given a model and input DataFrame (single row)."""
    return model.predict(input_df)[0] 