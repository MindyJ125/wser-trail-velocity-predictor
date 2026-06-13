"""
Western States 2025 Feature Integration & Speed Processor

Overview:
This script acts as the final data joiner for our ML dataset. It iterates through 
every 2025 runner's time checkpoints, maps those checkpoints to exact mileage windows 
in our GPX terrain matrix, and calculates the segment features: total elevation gain, 
total elevation loss, average slope grade, and the runner's actual velocity (mph).

Inputs:
    - data/processed/wser_course_profile.csv
    - data/processed/wser_2025_cleaned_splits.csv

Outputs:
    - data/processed/final_training_matrix.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

TERRAIN_CSV = PROCESSED_DATA_DIR / "wser_course_profile.csv"
SPLITS_CSV = PROCESSED_DATA_DIR / "wser_2025_cleaned_splits.csv"
OUTPUT_CSV = PROCESSED_DATA_DIR / "final_training_matrix.csv"

# Exact 2025 Official Checkpoint Layout Mapping
CHECKPOINTS = {
    'Start': 0.0,
    'Lyon Ridge': 10.3,
    'Red Star Ridge': 15.8,
    'Robinson Flat': 30.3,
    'Dusty Corners': 38.0,
    'Devil\'s Thumb': 47.8,
    'Michigan Bluff': 55.7,
    'Foresthill': 62.0,
    'Rucky Chucky': 78.0,
    'Green Gate': 79.8,
    'Auburn Lake Trails': 85.2,
    'Pointed Rocks': 94.3,
    'Finish': 100.2
}

def parse_time_to_hours(time_str):
    """Converts a standard split string (HH:MM:SS or elapsed) into float hours."""
    try:
        if pd.isna(time_str) or str(time_str).strip() in ['', 'DNF', 'DNS']:
            return None
        parts = list(map(int, str(time_str).split(':')))
        if len(parts) == 3:
            return parts[0] + parts[1]/60.0 + parts[2]/3600.0
        elif len(parts) == 2:
            return parts[0]/60.0 + parts[1]/3600.0
        return None
    except Exception:
        return None

def extract_segment_terrain_metrics(start_mile, end_mile, df_terrain):
    """Isolates a slice of the GPX terrain and computes net environmental metrics."""
    # Slice the terrain df for rows falling inside the distance bounds
    mask = (df_terrain['cumulative_dist_miles'] >= start_mile) & (df_terrain['cumulative_dist_miles'] <= end_mile)
    segment_df = df_terrain[mask]
    
    if segment_df.empty:
        return 0.0, 0.0, 0.0
        
    elevations = segment_df['ele_delta_meters'].values
    
    # Calculate cumulative ascents and descents (converted to feet for intuitive running analysis)
    gain_feet = np.sum(elevations[elevations > 0]) * 3.28084
    loss_feet = np.abs(np.sum(elevations[elevations < 0])) * 3.28084
    
    # Compute the average smoothed grade of this chunk
    avg_grade = segment_df['smoothed_grade'].mean()
    
    return gain_feet, loss_feet, avg_grade

def build_training_matrix():
    print("Loading datasets...")
    df_terrain = pd.read_csv(TERRAIN_CSV)
    df_splits = pd.read_csv(SPLITS_CSV)
    
    # Identify checkpoint order sequence
    checkpoint_sequence = list(CHECKPOINTS.keys())
    
    rows = []
    
    print("Processing matching matrix rows across runner profiles...")
    for idx, runner in df_splits.iterrows():
        # Extrapolate biological features
        runner_id = runner.get('Bib', idx)
        gender = runner.get('Gender', 'Unknown')
        age = runner.get('Age', np.nan)
        
        # Track previous checkpoint values inside the sequence loop
        prev_cp = checkpoint_sequence[0]
        prev_time = 0.0 # Time at start is 0.0 hours elapsed
        
        for current_cp in checkpoint_sequence[1:]:
            # Cleanly get time value for current checkpoint header
            raw_time = runner.get(current_cp)
            current_time = parse_time_to_hours(raw_time)
            
            # If runner missed a checkpoint or dropped out (DNF), skip this and future segments
            if current_time is None or current_time <= prev_time:
                break
                
            # Segment math
            dist_start = CHECKPOINTS[prev_cp]
            dist_end = CHECKPOINTS[current_cp]
            segment_distance = dist_end - dist_start
            
            elapsed_segment_hours = current_time - prev_time
            
            # Velocity Calculation (y-target)
            velocity_mph = segment_distance / elapsed_segment_hours
            
            # Pull down physical terrain features (X-features)
            gain_ft, loss_ft, mean_grade = extract_segment_terrain_metrics(dist_start, dist_end, df_terrain)
            
            # Track overall progress to capture systemic fatigue development
            cumulative_miles_completed = dist_start
            
            rows.append({
                'runner_id': runner_id,
                'gender': gender,
                'age': age,
                'segment_name': f"{prev_cp}_to_{current_cp}",
                'segment_distance_miles': segment_distance,
                'elevation_gain_feet': gain_ft,
                'elevation_loss_feet': loss_ft,
                'mean_grade': mean_grade,
                'cumulative_miles_completed': cumulative_miles_completed,
                'elapsed_hours_at_segment_start': prev_time,
                'velocity_mph': velocity_mph  # The machine learning prediction goal
            })
            
            # Advance structural pointers forward
            prev_cp = current_cp
            prev_time = current_time

    # Build final integrated dataframe
    final_df = pd.DataFrame(rows)
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Success! Integrated matrix contains {len(final_df)} structured data arrays.")
    print(f"File saved cleanly to: {OUTPUT_CSV}\n")

if __name__ == "__main__":
    build_training_matrix()