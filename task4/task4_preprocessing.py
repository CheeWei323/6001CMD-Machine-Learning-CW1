import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
dataset_path = project_root / "data" / "mudah-apartment-kl-selangor.csv"

df = pd.read_csv(dataset_path)

# --------------------------------------------------
# 4.1 Data Cleaning
# --------------------------------------------------

clean_df = df.drop_duplicates().copy()

# Convert monthly rent to numeric RM
clean_df["rent_numeric"] = pd.to_numeric(
    clean_df["monthly_rent"]
    .astype(str)
    .str.replace(r"[^0-9]", "", regex=True),
    errors="coerce"
)

# Convert property size to numeric square feet
clean_df["size_numeric"] = pd.to_numeric(
    clean_df["size"]
    .astype(str)
    .str.extract(r"([0-9][0-9,]*)")[0]
    .str.replace(",", "", regex=False),
    errors="coerce"
)

# Remove records without the regression target
model_df = clean_df[
    clean_df["rent_numeric"].notna()
].copy()

print("TASK 4.1 - DATA CLEANING")
print(f"Original records: {len(df):,}")
print(f"After duplicate removal: {len(clean_df):,}")
print(f"Records available for regression: {len(model_df):,}")
print(f"Missing parking: {model_df['parking'].isna().sum():,}")
print(
    f"Missing completion year: "
    f"{model_df['completion_year'].isna().sum():,}"
)

# --------------------------------------------------
# 4.2 Data Transformation
# --------------------------------------------------

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Numerical predictors used as an example
numeric_features = model_df[
    ["size_numeric", "parking", "bathroom"]
].copy()

# Standardisation
standard_scaler = StandardScaler()
standardised = standard_scaler.fit_transform(
    numeric_features
)

# Min-max normalisation
minmax_scaler = MinMaxScaler()
normalised = minmax_scaler.fit_transform(
    numeric_features
)

# Log transformation of the target
model_df["log_rent"] = np.log1p(
    model_df["rent_numeric"]
)

# --------------------------------------------------
# 4.3 Feature Engineering
# --------------------------------------------------

from sklearn.preprocessing import OneHotEncoder

# Remove identifier from predictive features
feature_df = model_df.drop(
    columns=["ads_id", "monthly_rent"]
).copy()

# Selected categorical features
categorical_features = [
    "region",
    "property_type",
    "furnished",
    "location"
]

# One-hot encode nominal categories
encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=True
)

encoded_categories = encoder.fit_transform(
    feature_df[categorical_features]
)