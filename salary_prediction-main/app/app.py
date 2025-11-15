import streamlit as st
import pandas as pd
import joblib
import os

MODEL_PATH = os.path.join('models', 'salary_model.pkl')
ENCODER_PATH = os.path.join('models', 'encoder.pkl')
SCALER_PATH = os.path.join('models', 'scaler.pkl')
LABEL_ENCODER_PATH = os.path.join('models', 'label_encoder.pkl')

st.title('Income Category Prediction App')
st.write('Enter your details to predict if your income is >50K or <=50K:')

# User input fields based on actual columns
def user_input():
    age = st.number_input('Age', min_value=17, max_value=90, value=30)
    workclass = st.selectbox('Workclass', ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', '?'])
    fnlwgt = st.number_input('Final Weight (fnlwgt)', min_value=0, value=100000)
    education = st.selectbox('Education', ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'])
    educational_num = st.number_input('Education Number', min_value=1, max_value=16, value=9)
    marital_status = st.selectbox('Marital Status', ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent'])
    occupation = st.selectbox('Occupation', ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces', '?'])
    relationship = st.selectbox('Relationship', ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
    race = st.selectbox('Race', ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
    gender = st.selectbox('Gender', ['Female', 'Male'])
    capital_gain = st.number_input('Capital Gain', min_value=0, value=0)
    capital_loss = st.number_input('Capital Loss', min_value=0, value=0)
    hours_per_week = st.number_input('Hours per Week', min_value=1, max_value=99, value=40)
    native_country = st.selectbox('Native Country', ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands', '?'])
    data = {
        'age': [age],
        'workclass': [workclass.lower()],
        'fnlwgt': [fnlwgt],
        'education': [education.lower()],
        'educational-num': [educational_num],
        'marital-status': [marital_status.lower()],
        'occupation': [occupation.lower()],
        'relationship': [relationship.lower()],
        'race': [race.lower()],
        'gender': [gender.lower()],
        'capital-gain': [capital_gain],
        'capital-loss': [capital_loss],
        'hours-per-week': [hours_per_week],
        'native-country': [native_country.lower()]
    }
    return pd.DataFrame(data)

input_df = user_input()

if st.button('Predict Income Category'):
    # Load encoders and model
    encoder = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
    numeric_cols = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    # Clean and preprocess input
    for col in categorical_cols:
        input_df[col] = input_df[col].str.strip().str.lower()
    # Encode and scale
    encoded = encoder.transform(input_df[categorical_cols])
    scaled = scaler.transform(input_df[numeric_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
    scaled_df = pd.DataFrame(scaled, columns=numeric_cols)
    final_df = pd.concat([encoded_df, scaled_df], axis=1)
    # Load model and predict
    model = joblib.load(MODEL_PATH)
    pred = model.predict(final_df)[0]
    pred_label = label_encoder.inverse_transform([pred])[0]
    st.success(f'Predicted Income Category: {pred_label}') 