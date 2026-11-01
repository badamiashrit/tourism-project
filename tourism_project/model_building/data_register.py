import os
import sys
import pandas as pd

def register_dataset():
    csv_path = os.path.join("tourism_project", "data", "tourism.csv")
    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at '{csv_path}'.")
        sys.exit(1)

    print(f"Loading dataset from: {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        sys.exit(1)

    expected_columns = [
        "CustomerID",
        "ProdTaken",
        "Age",
        "TypeofContact",
        "CityTier",
        "DurationOfPitch",
        "Occupation",
        "Gender",
        "MaritalStatus"
    ]

    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        print(f"Validation failed. Missing columns: {missing_columns}")
        sys.exit(1)
    else:
        print("Validation success. All expected columns are present.")

if __name__ == "__main__":
    register_dataset()
