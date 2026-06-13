"""
Western States 2025 Advanced Production Pipeline

Overview:
This script constructs an end-to-end scikit-learn Pipeline. It applies automatic 
scaling to numerical features, encodes categorical variables, and trains a 
non-linear Random Forest Regressor. Finally, it serializes (saves) the entire 
pipeline as a deployment-ready .joblib artifact.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Directory setup
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
MATRIX_CSV = PROCESSED_DATA_DIR / "final_training_matrix.csv"
MODEL_OUTPUT_PATH = MODELS_DIR / "wser_pacing_pipeline.joblib"

def build_and_train_pipeline():
    if not MATRIX_CSV.exists():
        raise FileNotFoundError(f"Missing training matrix at: {MATRIX_CSV}")
        
    print("Loading training matrix...")
    df = pd.read_csv(MATRIX_CSV)
    
    # Basic cleanup (dropping NAs and extreme outliers)
    df = df.dropna(subset=['age', 'velocity_mph', 'mean_grade', 'gender'])
    df = df[(df['velocity_mph'] > 1.5) & (df['velocity_mph'] < 15.0)]
    
    # Define features (X) and target (y)
    # Notice we pass the raw 'gender' string now, not a manually coded binary
    feature_cols = [
        'age', 
        'gender', 
        'elevation_gain_feet', 
        'elevation_loss_feet', 
        'mean_grade', 
        'cumulative_miles_completed', 
        'elapsed_hours_at_segment_start'
    ]
    
    X = df[feature_cols]
    y = df['velocity_mph']
    
    # Split the data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nConstructing Preprocessing Pipeline...")
    # 1. Define which columns get which mathematical treatment
    numeric_features = ['age', 'elevation_gain_feet', 'elevation_loss_feet', 'mean_grade', 'cumulative_miles_completed', 'elapsed_hours_at_segment_start']
    categorical_features = ['gender']
    
    # 2. Build the transformer gate
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # 3. Build the final Master Pipeline
    print("Initializing Random Forest Regressor...")
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    # Train the entire pipeline (scaling + encoding + modeling) in one line
    print("Training Pipeline on 2025 WSER data (This may take a few seconds)...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate performance
    predictions = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print("\n=========================================")
    print("      VERSION 2.0 ADVANCED METRICS       ")
    print("=========================================")
    print(f"Mean Absolute Error (MAE): {mae:.2f} mph")
    print(f"R-squared Score (R2):      {r2:.4f}")
    print("=========================================")
    
    # Serialize and save the model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_OUTPUT_PATH)
    print(f"\nSuccess! Production pipeline saved to: {MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    build_and_train_pipeline()