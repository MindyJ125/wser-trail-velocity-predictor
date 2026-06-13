"""
Western States 2025 - Inference Engine

Overview:
Loads the serialized production ML pipeline to predict a runner's pace over a 
custom sequence of trail segments. It dynamically updates cumulative fatigue 
(elapsed hours and miles) as the simulation progresses.
"""

from pathlib import Path
import pandas as pd
import joblib

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
MODEL_PATH = ROOT_DIR / "models" / "wser_pacing_pipeline.joblib"

def simulate_trail_run():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Cannot find trained model at {MODEL_PATH}")
        
    print("Loading Production AI Pipeline...")
    pipeline = joblib.load(MODEL_PATH)
    
    # 1. Define our Runner Profile
    runner_age = 32
    runner_gender = 'M'
    print(f"Runner Profile: {runner_age}-year-old {runner_gender}")
    
    # 2. Define a custom 20-mile trail (Four 5-mile segments)
    # Segment 1: Flat and fast
    # Segment 2: Brutal climb
    # Segment 3: Steep downhill
    # Segment 4: Rolling hills to the finish
    trail_segments = [
        {"segment_name": "The Flat Start", "miles": 5.0, "gain_ft": 200, "loss_ft": 200, "mean_grade": 0.0},
        {"segment_name": "The Widowmaker", "miles": 5.0, "gain_ft": 2500, "loss_ft": 50, "mean_grade": 9.5},
        {"segment_name": "Knee Crusher Descent", "miles": 5.0, "gain_ft": 100, "loss_ft": 2800, "mean_grade": -10.2},
        {"segment_name": "The Exhausted Finish", "miles": 5.0, "gain_ft": 600, "loss_ft": 600, "mean_grade": 1.5}
    ]
    
    # 3. Initialize Tracking Variables
    cumulative_miles = 0.0
    elapsed_hours = 0.0
    
    print("\nStarting Simulation...\n")
    print(f"{'Segment Name':<25} | {'Grade':<6} | {'Predicted Speed':<15} | {'Segment Time':<15} | {'Total Race Time'}")
    print("-" * 85)
    
    # 4. Run the Simulation Loop
    for seg in trail_segments:
        # Create a single-row DataFrame formatted exactly like our training data
        current_state = pd.DataFrame([{
            'age': runner_age,
            'gender': runner_gender,
            'elevation_gain_feet': seg["gain_ft"],
            'elevation_loss_feet': seg["loss_ft"],
            'mean_grade': seg["mean_grade"],
            'cumulative_miles_completed': cumulative_miles,
            'elapsed_hours_at_segment_start': elapsed_hours
        }])
        
        # Ask the model for the predicted speed for this specific segment
        predicted_mph = pipeline.predict(current_state)[0]
        
        # Calculate how long this segment will take (Time = Distance / Speed)
        segment_hours = seg["miles"] / predicted_mph
        
        # Update our trackers for the NEXT loop iteration
        cumulative_miles += seg["miles"]
        elapsed_hours += segment_hours
        
        # Convert decimal hours to Minutes for easier reading
        seg_mins = int(segment_hours * 60)
        total_hrs = int(elapsed_hours)
        total_mins = int((elapsed_hours - total_hrs) * 60)
        
        print(f"{seg['segment_name']:<25} | {seg['mean_grade']:>4.1f}% | {predicted_mph:>4.2f} mph       | {seg_mins:>3} minutes     | {total_hrs}h {total_mins:02d}m")

if __name__ == "__main__":
    simulate_trail_run()