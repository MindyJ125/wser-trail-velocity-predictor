"""
Western States 2025 - Interactive Pace Dashboard
Run this app in your terminal using: streamlit run src/app.py
"""

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --- PAGE SETUP ---
st.set_page_config(page_title="WSER AI Pacer", layout="wide", page_icon="⛰️")

# --- LOAD MODEL ---
# The @st.cache_resource decorator tells Streamlit to only load the model 
# once, keeping the app lightning fast when the user tweaks inputs.
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent.parent / "models" / "wser_pacing_pipeline.joblib"
    if not model_path.exists():
        st.error(f"Model not found at {model_path}. Please train the advanced pipeline first.")
        st.stop()
    return joblib.load(model_path)

pipeline = load_model()

# --- UI FRONTEND ---
st.title("🏃‍♂️ Western States 100: AI Pace Predictor")
st.markdown("Adjust the runner profile on the left to see how the neural ensemble predicts speed decay across varying mountain terrain.")

# --- SIDEBAR INPUTS ---
st.sidebar.header("⚙️ Runner Profile")
runner_age = st.sidebar.slider("Age", min_value=18, max_value=80, value=32, step=1)
runner_gender = st.sidebar.radio("Gender", options=["M", "F"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Simulation Parameters:**")
st.sidebar.text("Total Distance: 20 miles")
st.sidebar.text("Elevation Gain: +3,400 ft")

# --- TRAIL DEFINITION ---
trail_segments = [
    {"Segment": "1. The Flat Start", "miles": 5.0, "gain_ft": 200, "loss_ft": 200, "mean_grade": 0.0},
    {"Segment": "2. The Widowmaker", "miles": 5.0, "gain_ft": 2500, "loss_ft": 50, "mean_grade": 9.5},
    {"Segment": "3. Knee Crusher Descent", "miles": 5.0, "gain_ft": 100, "loss_ft": 2800, "mean_grade": -10.2},
    {"Segment": "4. The Exhausted Finish", "miles": 5.0, "gain_ft": 600, "loss_ft": 600, "mean_grade": 1.5}
]

# --- SIMULATION ENGINE ---
cumulative_miles = 0.0
elapsed_hours = 0.0
results = []

for seg in trail_segments:
    # 1. Build the data row exactly as the pipeline expects it
    current_state = pd.DataFrame([{
        'age': runner_age,
        'gender': runner_gender,
        'elevation_gain_feet': seg["gain_ft"],
        'elevation_loss_feet': seg["loss_ft"],
        'mean_grade': seg["mean_grade"],
        'cumulative_miles_completed': cumulative_miles,
        'elapsed_hours_at_segment_start': elapsed_hours
    }])
    
    # 2. Predict
    predicted_mph = pipeline.predict(current_state)[0]
    
    # 3. Time Math
    segment_hours = seg["miles"] / predicted_mph
    cumulative_miles += seg["miles"]
    elapsed_hours += segment_hours
    
    total_hrs = int(elapsed_hours)
    total_mins = int((elapsed_hours - total_hrs) * 60)
    time_string = f"{total_hrs}h {total_mins:02d}m"
    
    # 4. Save to results
    results.append({
        "Segment": seg["Segment"],
        "Grade (%)": seg["mean_grade"],
        "Predicted Speed (mph)": round(predicted_mph, 2),
        "Cumulative Time": time_string
    })

# Convert results to a clean DataFrame
results_df = pd.DataFrame(results)

# --- VISUALIZATION ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Speed Decay Profile")
    # Set the segment name as the index so the chart labels nicely
    chart_data = results_df.set_index("Segment")[["Predicted Speed (mph)"]]
    st.line_chart(chart_data)

with col2:
    st.subheader("Turn-by-Turn Metrics")
    st.dataframe(results_df, hide_index=True)