import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def main():
    # Load registered dataset
    csv_path = os.path.join("tourism_project", "data", "tourism.csv")
    df = pd.read_csv(csv_path)
    
    # Treat error in Gender column
    df['Gender'] = df['Gender'].replace("Fe Male", "Female")
    
    # Define features and target
    X = df.drop(["CustomerID", "ProdTaken", "PitchSatisfactionScore", "ProductPitched", "NumberOfFollowups", "DurationOfPitch"], axis=1, errors="ignore")
    y = df["ProdTaken"]
    
    # Create dummy variables for categorical features
    X = pd.get_dummies(X, drop_first=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=25, stratify=y)
    
    # Create output directory for processed files
    output_dir = os.path.join("tourism_project", "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save splits
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    pd.DataFrame(y_train).to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    pd.DataFrame(y_test).to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    
    print("Data preparation successfully completed. Splits saved to:", output_dir)

if __name__ == '__main__':
    main()
