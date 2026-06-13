"""
Western States 2025 Baseline Model Training Pipeline

Overview:
This script loads the integrated feature matrix, handles categorical encoding 
and missing values, splits the data into train/test sets, and trains a baseline 
Linear Regression model to predict runner velocity. It outputs evaluation metrics 
to establish our Version 1.0 performance floor.

Inputs:
    - data/processed/final_training_matrix.csv

Outputs:
    - Printed Performance Metrics (MAE, R-squared)
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MATRIX_CSV = PROCESSED_DATA_DIR / "final_training_matrix.csv"

def train_baseline_model():
    if not MATRIX_CSV.exists():
        raise FileNotFoundError(f"Missing training matrix! Please run feature integration first: {MATRIX_CSV}")
        
    print("Loading integrated training matrix...")
    df = pd.read_csv(MATRIX_CSV)
    
    # --- DATA CLEANING & PREPROCESSING ---
    initial_rows = len(df)
    
    # Drop rows where critical features or targets are missing (e.g., unrecorded age or DNF gaps)
    df = df.dropna(subset=['age', 'velocity_mph', 'mean_grade'])
    
    # Filter out extreme outliers (e.g., people resting at an aid station for 3 hours 
    # making their speed look like 0.1 mph, or data errors)
    df = df[(df['velocity_mph'] > 1.5) & (df['velocity_mph'] < 15.0)]
    
    print(f"Cleaned dataset: Retained {len(df)} of {initial_rows} rows after filtering outliers.")
    
    # Convert Categorical Gender to Binary Feature (Male = 1, Female = 0)
    # Handling variations in data formatting gracefully
    df['is_male'] = df['gender'].astype(str).str.upper().apply(lambda x: 1 if x.startswith('M') else 0)
    
    # --- FEATURE SELECTION ---
    # Define features (X) and target variable (y)
    feature_cols = [
        'age', 
        'is_male', 
        'elevation_gain_feet', 
        'elevation_loss_feet', 
        'mean_grade', 
        'cumulative_miles_completed', 
        'elapsed_hours_at_segment_start'
    ]
    
    X = df[feature_cols]
    y = df['velocity_mph']
    
    # --- TRAIN/TEST SPLIT ---
    # Split: 80% to train the model, 20% held back to test its real-world generalization
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {X_train.shape[0]} rows")
    print(f"Testing set size: {X_test.shape[0]} rows")
    
    # --- MODEL TRAINING ---
    print("\nTraining Baseline Ridge Regression Model...")
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    # --- EVALUATION ---
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print("\n=========================================")
    print("      VERSION 1.0 BASELINE METRICS       ")
    print("=========================================")
    print(f"Mean Absolute Error (MAE): {mae:.2f} mph")
    print(f"R-squared Score (R2):      {r2:.4f}")
    print("=========================================")
    
    # Quick insight look at the mathematical weights assigned to features
    print("\nFeature Impact (Model Coefficients):")
    for col, coef in zip(feature_cols, model.coef_):
        print(f" - {col:<32}: {coef:.5f}")

if __name__ == "__main__":
    train_baseline_model()