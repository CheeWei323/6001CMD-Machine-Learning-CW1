import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Locate dataset and output folder
project_root = Path(__file__).resolve().parent.parent
dataset_path = project_root / "data" / "mudah-apartment-kl-selangor.csv"
output_path = project_root / "outputs" / "task3"

output_path.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(dataset_path)

# --------------------------------------------------
# 3.1 Missing Values
# --------------------------------------------------

missing_count = df.isna().sum()
missing_percentage = (missing_count / len(df)) * 100

missing_summary = pd.DataFrame({
    "Missing Values": missing_count,
    "Missing (%)": missing_percentage
})

missing_summary = missing_summary[
    missing_summary["Missing Values"] > 0
].sort_values("Missing Values", ascending=False)

print("TASK 3.1 - MISSING VALUES")
print(missing_summary.round(2))

# Plot the five attributes with the most missing values
top_missing = missing_summary.head(5)

plt.figure(figsize=(8, 5))
plt.bar(top_missing.index, top_missing["Missing (%)"])

plt.title("Attributes with the Highest Missing Values")
plt.xlabel("Attribute")
plt.ylabel("Missing Values (%)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig(output_path / "missing_values.png", dpi=300)
plt.show()

# --------------------------------------------------
# 3.2 Duplicate and Inconsistent Records
# --------------------------------------------------

# Find advertisement IDs that appear more than once
id_counts = df["ads_id"].value_counts()
repeated_ids = id_counts[id_counts > 1]

# Count exact duplicate rows
exact_duplicate_rows = df.duplicated().sum()

# Separate repeated IDs into identical and conflicting records
identical_ids = 0
conflicting_ids = 0

for ads_id in repeated_ids.index:
    records = df[df["ads_id"] == ads_id]

    if len(records.drop_duplicates()) == 1:
        identical_ids += 1
    else:
        conflicting_ids += 1

# Check selected inconsistent room-value formats
room_formats = ["3", "3.0", "More than 10"]

print("\nTASK 3.2 - DUPLICATES AND INCONSISTENCIES")
print(f"Repeated advertisement IDs: {len(repeated_ids)}")
print(f"Rows involving repeated IDs: {repeated_ids.sum()}")
print(f"Exact duplicate rows: {exact_duplicate_rows}")
print(f"Repeated IDs with identical records: {identical_ids}")
print(f"Repeated IDs with conflicting records: {conflicting_ids}")

print("\nExample room-format inconsistencies:")
for value in room_formats:
    count = (df["rooms"] == value).sum()
    print(f"{value}: {count}")

# --------------------------------------------------
# 3.3 Distribution, Skewness, Outliers and Noise
# --------------------------------------------------

from scipy.stats import skew

# Remove exact duplicates
prepared_df = df.drop_duplicates().copy()

# Convert rent and size into numerical values
prepared_df["rent_numeric"] = pd.to_numeric(
    prepared_df["monthly_rent"]
    .astype(str)
    .str.replace(r"[^0-9]", "", regex=True),
    errors="coerce"
)

prepared_df["size_numeric"] = pd.to_numeric(
    prepared_df["size"]
    .astype(str)
    .str.extract(r"([0-9][0-9,]*)")[0]
    .str.replace(",", "", regex=False),
    errors="coerce"
)

# Retain records with a known rent target
prepared_df = prepared_df[
    prepared_df["rent_numeric"].notna()
].copy()

# Calculate descriptive statistics
rent = prepared_df["rent_numeric"]
size = prepared_df["size_numeric"]

print("\nTASK 3.3 - DISTRIBUTION AND SKEWNESS")
print(f"Prepared records: {len(prepared_df):,}")

print("\nMonthly rent:")
print(f"Median: RM{rent.median():,.0f}")
print(f"Mean: RM{rent.mean():,.0f}")
print(f"Maximum: RM{rent.max():,.0f}")
print(f"Skewness: {skew(rent, bias=False):.2f}")

print("\nProperty size:")
print(f"Median: {size.median():,.0f} sq.ft.")
print(f"Mean: {size.mean():,.0f} sq.ft.")
print(f"Maximum: {size.max():,.0f} sq.ft.")
print(f"Skewness: {skew(size, bias=False):.2f}")

# Monthly rent distribution
rent_limit = rent.quantile(0.99)

plt.figure(figsize=(8, 5))
plt.hist(rent[rent <= rent_limit], bins=40)

plt.title("Distribution of Monthly Rent")
plt.xlabel("Monthly Rent (RM)")
plt.ylabel("Number of Advertisements")
plt.tight_layout()

plt.savefig(
    output_path / "rent_distribution.png",
    dpi=300
)
plt.show()

# Property size distribution
size_limit = size.quantile(0.99)

plt.figure(figsize=(8, 5))
plt.hist(size[size <= size_limit], bins=40)

plt.title("Distribution of Property Size")
plt.xlabel("Property Size (sq.ft.)")
plt.ylabel("Number of Advertisements")
plt.tight_layout()

plt.savefig(
    output_path / "size_distribution.png",
    dpi=300
)
plt.show()

# --------------------------------------------------
# 3.4 Correlation and Target Imbalance
# --------------------------------------------------

# Convert selected variables to numeric
prepared_df["rooms_numeric"] = pd.to_numeric(
    prepared_df["rooms"], errors="coerce"
)

prepared_df["parking_numeric"] = pd.to_numeric(
    prepared_df["parking"], errors="coerce"
)

prepared_df["bathroom_numeric"] = pd.to_numeric(
    prepared_df["bathroom"], errors="coerce"
)

# Variables used for Spearman correlation
pairs = [
    ("rent_numeric", "size_numeric", "Rent and size"),
    ("rent_numeric", "parking_numeric", "Rent and parking"),
    ("size_numeric", "rooms_numeric", "Size and rooms"),
    ("rooms_numeric", "bathroom_numeric", "Rooms and bathrooms")
]

print("\nTASK 3.4 - CORRELATION")

for first, second, label in pairs:
    valid = prepared_df[[first, second]].dropna()

    correlation = valid[first].corr(
        valid[second], method="spearman"
    )

    print(
        f"{label}: "
        f"{len(valid):,} pairs, "
        f"correlation = {correlation:.2f}"
    )

# Examine sparsely represented high-rent listings
high_rent = (prepared_df["rent_numeric"] >= 5000).sum()
high_rent_percentage = (
    high_rent / len(prepared_df)
) * 100

print("\nHigh-rent listings:")
print(f"RM5,000 and above: {high_rent:,}")
print(f"Percentage: {high_rent_percentage:.2f}%")
