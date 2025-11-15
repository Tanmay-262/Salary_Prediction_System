import os
import pytest
import pandas as pd
import joblib
from src import predict

def test_model_file_exists():
    assert os.path.exists('models/salary_model.pkl'), "Model file does not exist!"

def test_prediction_shape():
    if not os.path.exists('models/salary_model.pkl'):
        pytest.skip('Model file not found, skipping prediction test.')
    model = predict.load_model('models/salary_model.pkl')
    # Example input, adjust columns as per your feature engineering
    input_dict = {
        'education_bachelor': [1],
        'education_master': [0],
        'education_phd': [0],
        'job_role_developer': [1],
        'job_role_analyst': [0],
        'job_role_manager': [0],
        'location_new york': [1],
        'location_san francisco': [0],
        'location_bangalore': [0],
        'company_size_small': [1],
        'company_size_medium': [0],
        'company_size_large': [0],
        'years_experience': [5]
    }
    input_df = pd.DataFrame(input_dict)
    pred = predict.predict_salary(model, input_df)
    assert isinstance(pred, (int, float)), "Prediction is not a number!" 