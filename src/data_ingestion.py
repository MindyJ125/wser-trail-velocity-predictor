"""
Western States 2025 Runner Splits Ingestion Pipeline

Overview:
This script ingests the official 2025 WSER runner checkpoint splits. 
It establishes the target layer (y) for our machine learning model by reading 
the actual checkpoint times, isolating the time-of-day data, and preparing it 
to map directly onto our 2025 terrain GPX profile.

Inputs:
    - Local Cache: data/raw/WSER_2025_Splits.xlsx

Outputs:
    - Processed Table: data/processed/wser_2025_cleaned_splits.csv
"""

from pathlib import Path
import pandas as pd

# Establish absolute paths
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

LOCAL_EXCEL_PATH = RAW_DATA_DIR / "WSER_2025_Splits.xlsx"
OUTPUT_CSV_PATH = PROCESSED_DATA_DIR / "wser_2025_cleaned_splits.csv"

def process_2025_splits():
    """Reads the official 2025 Excel splits and standardizes the columns."""
    if not LOCAL_EXCEL_PATH.exists():
        raise FileNotFoundError(f"Missing 2025 Splits file! Please download to: {LOCAL_EXCEL_PATH}")
        
    print(f"Reading 2025 runner data from {LOCAL_EXCEL_PATH}...")
    
    # Read the Excel file (skipping any weird header rows WSER usually puts at the top)
    # Note: You may need to adjust `header=1` depending on the exact Excel layout
    df = pd.read_excel(LOCAL_EXCEL_PATH, header = 1, engine='openpyxl')
    
    # Drop rows where the runner DNF'd (Did Not Finish) before the first major checkpoint
    # Or clean empty rows
    df.dropna(how='all', inplace=True)
    
    print(f"Successfully loaded {len(df)} runner profiles from 2025.")
    
    # Save the cleaned intermediate version
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Cleaned 2025 splits saved to: {OUTPUT_CSV_PATH}\n")

if __name__ == "__main__":
    print("Starting 2025 Runner Data Ingestion...")
    process_2025_splits()