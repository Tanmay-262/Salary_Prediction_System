import os
from src import data_cleaning, feature_engineering, utils, model_training
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def main():
    # Paths
    data_path = os.path.join('data', 'salaries.csv')
    model_dir = 'models'
    encoder_path = os.path.join(model_dir, 'encoder.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    label_enc_path = os.path.join(model_dir, 'label_encoder.pkl')

    # Load and clean data
    df = data_cleaning.load_data(data_path)
    df = data_cleaning.handle_missing_values(df)
    df = data_cleaning.clean_category_names(df)

    # Use actual columns from CSV
    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
    numeric_cols = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    target_col = 'income'

    # Encode target
    label_encoder = LabelEncoder()
    df[target_col] = label_encoder.fit_transform(df[target_col])
    joblib.dump(label_encoder, label_enc_path)

    # Feature engineering (fit and save encoder/scaler)
    df_fe = feature_engineering.encode_and_scale(
        df, categorical_cols, numeric_cols, fit=True,
        enc_path=encoder_path, scaler_path=scaler_path
    )

    # Split and train
    from sklearn.model_selection import train_test_split
    X = df_fe.drop(columns=[target_col])
    y = df_fe[target_col]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print('Validation Accuracy:', acc)
    print(classification_report(y_val, preds))
    model_path = os.path.join(model_dir, 'salary_model.pkl')
    joblib.dump(clf, model_path)
    print('Model, encoder, scaler, and label encoder saved to models/')

if __name__ == '__main__':
    main() 