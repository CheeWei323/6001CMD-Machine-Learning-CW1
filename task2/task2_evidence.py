import pandas as pd
from pathlib import Path

# Locate and load the dataset
project_root = Path(__file__).resolve().parent.parent
dataset_path = project_root / "data" / "mudah-apartment-kl-selangor.csv"

df = pd.read_csv(dataset_path)

# Task 2.1 - Label availability
total = len(df)
with_rent = df["monthly_rent"].notna().sum()
without_rent = df["monthly_rent"].isna().sum()
label_coverage = (with_rent / total) * 100

print("TASK 2.1 - LABEL AVAILABILITY")
print(f"Total advertisements: {total:,}")
print(f"With monthly rent: {with_rent:,}")
print(f"Without monthly rent: {without_rent:,}")
print(f"Label coverage: {label_coverage:.2f}%")

# Task 2.2 - Data requirements
missing_year = df["completion_year"].isna().sum()
missing_parking = df["parking"].isna().sum()

print("\nTASK 2.2 - DATA REQUIREMENTS")
print(f"Missing completion year: {missing_year:,}")
print(f"Missing parking: {missing_parking:,}")
print(f"monthly_rent type: {df['monthly_rent'].dtype}")
print(f"size type: {df['size'].dtype}")
print(f"location type: {df['location'].dtype}")
print(f"furnished type: {df['furnished'].dtype}")

