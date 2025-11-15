import pandas as pd

def load_data(filepath):
    """Load the salary dataset from a CSV file."""
    return pd.read_csv(filepath)

def handle_missing_values(df):
    """Handle missing values by filling or dropping as appropriate."""
    # Example: fill numeric with median, categorical with mode
    for col in df.select_dtypes(include=['number']).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df

def clean_category_names(df):
    """Clean and standardize categorical column names."""
    # Example: strip whitespace, lower case
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip().str.lower()
    return df 