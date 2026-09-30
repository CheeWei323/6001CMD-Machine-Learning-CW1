import pandas as pd
from pathlib import Path

# Locate and load the dataset
project_root = Path(__file__).resolve().parent.parent
dataset_path = project_root / "data" / "mudah-apartment-kl-selangor.csv"

df = pd.read_csv(dataset_path)

# Basic dataset characteristics
print("Dataset shape:", df.shape)
print("Kuala Lumpur listings:",
      (df["region"] == "Kuala Lumpur").sum())
print("Selangor listings:",
      (df["region"] == "Selangor").sum())

# Target availability
print("Recorded monthly rent:",
      df["monthly_rent"].notna().sum())
print("Missing monthly rent:",
      df["monthly_rent"].isna().sum())

# Original storage formats
print("monthly_rent data type:", df["monthly_rent"].dtype)
print("size data type:", df["size"].dtype)
print("Rent example:",
      df["monthly_rent"].dropna().iloc[0])
print("Size example:",
      df["size"].dropna().iloc[0])