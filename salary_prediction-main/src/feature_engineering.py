import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib


def encode_and_scale(df, categorical_cols, numeric_cols, fit=True, enc_path=None, scaler_path=None):
    """Encode categorical columns and scale numeric columns. Save or load encoders/scalers as needed."""
    if fit:
        # For scikit-learn >=1.2, use sparse_output. For older, use sparse.
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        scaler = StandardScaler()
        encoded = encoder.fit_transform(df[categorical_cols])
        scaled = scaler.fit_transform(df[numeric_cols])
        if enc_path:
            joblib.dump(encoder, enc_path)
        if scaler_path:
            joblib.dump(scaler, scaler_path)
    else:
        encoder = joblib.load(enc_path)
        scaler = joblib.load(scaler_path)
        encoded = encoder.transform(df[categorical_cols])
        scaled = scaler.transform(df[numeric_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)
    scaled_df = pd.DataFrame(scaled, columns=numeric_cols, index=df.index)
    df = df.drop(columns=categorical_cols+numeric_cols)
    df = pd.concat([df, encoded_df, scaled_df], axis=1)
    return df 