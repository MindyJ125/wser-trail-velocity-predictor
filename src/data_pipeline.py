"""
Western States 100 (WSER) Terrain Profiling Pipeline

Overview:
This script establishes the geographical terrain foundation for the runner velocity 
prediction model. Because official course maps contain precise spatial layout data 
but lack real-time runner performance logs, this script isolates the terrain features 
as an independent data layer.

The pipeline automatically streams the official WSER 2025 GPX route, extracts all 
coordinate points, and chronologically calculates the step-by-step distance deltas, 
cumulative mile markers, and absolute elevations. It then calculates the raw incline 
grade (slope) for every segment of the trail. 

To fix consumer-grade GPS noise and erratic spikes in altitude data, the pipeline 
applies a centered rolling average window to generate a clean, smoothed trail grade metric.

Inputs:
    - Remote URL: Official WSER 2025 GPX course route with elevation markers.
    - Local Cache: data/raw/WSER2025welev.gpx

Outputs:
    - Processed Table: data/processed/wser_course_profile.csv
      Contains columns: lat, lon, ele, dist_delta_meters, cumulative_dist_meters, 
      cumulative_dist_miles, ele_delta_meters, raw_grade, and smoothed_grade.
"""

import os
from pathlib import Path
import requests
import gpxpy
import pandas as pd
import numpy as np

# Establish absolute paths relative to this script's location
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

# Ensure target directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

GPX_URL = "https://www.wser.org/wp-content/uploads/gpx/WSER2025welev.gpx"
LOCAL_GPX_PATH = RAW_DATA_DIR / "WSER2025welev.gpx"
OUTPUT_CSV_PATH = PROCESSED_DATA_DIR / "wser_course_profile.csv"

def download_gpx_if_missing():
    """Checks for local data; downloads from the WSER server if missing."""
    if LOCAL_GPX_PATH.exists():
        print(f"Local GPX file found at: {LOCAL_GPX_PATH}")
        return
        
    print(f"Downloading remote WSER GPX profile from {GPX_URL}...")
    response = requests.get(GPX_URL, timeout=15)
    if response.status_code == 200:
        with open(LOCAL_GPX_PATH, 'wb') as f:
            f.write(response.content)
        print("Download completed successfully.")
    else:
        raise ConnectionError(f"Failed to fetch GPX file. Status code: {response.status_code}")

def extract_trackpoints() -> pd.DataFrame:
    """Parses XML track elements into a standard pandas DataFrame."""
    print("Parsing spatial trackpoints from file...")
    with open(LOCAL_GPX_PATH, 'r', encoding='utf-8') as f:
        gpx = gpxpy.parse(f)
        
    data = []
    
    # Handle both standard track logs and route layouts
    if gpx.tracks:
        for track in gpx.tracks:
            for segment in track.segments:
                for pt in segment.points:
                    data.append({'lat': pt.latitude, 'lon': pt.longitude, 'ele': pt.elevation})
    elif gpx.routes:
        for route in gpx.routes:
            for pt in route.points:
                data.append({'lat': pt.latitude, 'lon': pt.longitude, 'ele': pt.elevation})
                
    if not data:
        raise ValueError("No spatial coordinates found. Check the GPX file structure.")
        
    return pd.DataFrame(data)

def compute_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates distances, mile markers, and smoothed slope grades."""
    print("Calculating terrain slope and mileage features...")
    
    # Compute 3D distance between consecutive points
    distances = [0.0]
    for i in range(1, len(df)):
        p1 = gpxpy.gpx.GPXTrackPoint(df.iloc[i-1]['lat'], df.iloc[i-1]['lon'], elevation=df.iloc[i-1]['ele'])
        p2 = gpxpy.gpx.GPXTrackPoint(df.iloc[i]['lat'], df.iloc[i]['lon'], elevation=df.iloc[i]['ele'])
        distances.append(p1.distance_3d(p2))
        
    df['dist_delta_meters'] = distances
    df['cumulative_dist_meters'] = df['dist_delta_meters'].cumsum()
    df['cumulative_dist_miles'] = df['cumulative_dist_meters'] * 0.000621371
    
    # Compute change in elevation (meters)
    df['ele_delta_meters'] = df['ele'].diff().fillna(0)
    
    # Calculate raw grade (rise over run), avoiding division by zero
    df['raw_grade'] = np.where(df['dist_delta_meters'] > 0, df['ele_delta_meters'] / df['dist_delta_meters'], 0)
    
    # Use a rolling average window to smooth out minor GPS elevation errors
    df['smoothed_grade'] = df['raw_grade'].rolling(window=7, min_periods=1, center=True).mean()
    
    return df

def run_pipeline():
    """Runs the full download, extraction, and feature engineering pipeline."""
    print("Starting WSER Data Pipeline...")
    
    download_gpx_if_missing()
    df_raw = extract_trackpoints()
    df_processed = compute_spatial_features(df_raw)
    
    # Export to data/processed/ directory
    df_processed.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
    print(f"Pipeline complete. Processed data saved to: {OUTPUT_CSV_PATH}\n")

if __name__ == "__main__":
    run_pipeline()