# Salary Prediction ML Project

## Project Overview
This project predicts salaries based on user input features such as education, experience, job role, location, and company size using machine learning. It includes data preprocessing, feature engineering, model training, evaluation, and a Streamlit web app frontend.

## Folder Structure
```
salary_prediction_project/
│
├── data/
│   └── salaries.csv                # <- Place your dataset here
├── notebooks/
│   └── EDA.ipynb                   # Exploratory data analysis
├── src/
│   ├── data_cleaning.py            # Data loading, missing value handling
│   ├── feature_engineering.py      # Encoding, scaling, transformations
│   ├── model_training.py           # Train and save model
│   ├── predict.py                  # Load model and make predictions
│   └── utils.py                    # Helper functions
├── models/
│   └── salary_model.pkl            # Trained ML model
├── app/
│   ├── app.py                      # Streamlit app interface
│   └── requirements.txt            # All dependencies
├── tests/
│   └── test_model.py               # Pytest unit tests
├── README.md
└── .gitignore
```

## Setup Instructions
1. Clone the repo and navigate to the project folder.
2. Place your `salaries.csv` file in the `data/` directory.
3. Install dependencies:
   ```bash
   pip install -r app/requirements.txt
   ```
4. Run EDA and preprocess data as needed.
5. Train the model using scripts in `src/`.
6. Launch the Streamlit app:
   ```bash
   streamlit run app/app.py
   ```

## Usage
- Use the Streamlit app to input features and get salary predictions.
- Modify the code as needed for your dataset.

## Testing
- Run unit tests with:
  ```bash
  pytest tests/
  ``` 