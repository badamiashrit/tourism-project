import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

def train_model():
    # Define dataset path
    csv_path = os.path.join("tourism_project", "data", "tourism.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    # 1. Load Data
    df = pd.read_csv(csv_path)
    
    # Data Preprocessing
    df['Gender'] = df['Gender'].replace("Fe Male", "Female")

    # Split the columns
    X = df.drop(["CustomerID", "ProdTaken", "PitchSatisfactionScore", "ProductPitched", "NumberOfFollowups", "DurationOfPitch"], axis=1, errors='ignore')
    if 'Unnamed: 0' in X.columns:
        X = X.drop(['Unnamed: 0'], axis=1)
    y = df["ProdTaken"]

    # Create dummy variables
    X = pd.get_dummies(X, drop_first=True)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=25, stratify=y)

    # Set up MLflow tracking configuration
    mlflow.set_experiment("Tourism_Demand_Forecasting")

    with mlflow.start_run(run_name="Production_Random_Forest") as run:
        # Initialize and fit the best selected classifier
        model = RandomForestClassifier(class_weight={0: 0.15, 1: 0.85}, max_features='log2', max_samples=0.3, min_samples_leaf=4, n_estimators=110, random_state=1)
        model.fit(X_train, y_train)
        
        # Evaluate
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        
        # Log parameters & metrics to MLflow
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        
        # Log Model artifact
        mlflow.sklearn.log_model(model, artifact_path="model")
        
        # Save locally for App Deployment
        deploy_dir = "tourism_project/deployment"
        os.makedirs(deploy_dir, exist_ok=True)
        with open(os.path.join(deploy_dir, "best_model.pkl"), "wb") as f:
            pickle.dump(model, f)
            
        print("Model training completed and model saved to deployment directory!")

if __name__ == "__main__":
    train_model()
