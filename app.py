import streamlit as st
import pandas as pd
import joblib

# --- Page Config ---
st.set_page_config(page_title="Fitness Predictor", page_icon="💪", layout="centered")

# --- Load Model & Columns ---
@st.cache_resource
def load_artifacts():
    model = joblib.load('calories_burn_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
    return model, model_columns

try:
    model, model_columns = load_artifacts()
except FileNotFoundError:
    st.error("Please run 'train_model.py' first to generate the model files!")
    st.stop()

# --- Title & Header ---
st.title("💪 Workout Calories Predictor")
st.markdown("Enter your stats below to see how many calories you'll burn!")

# --- Input Form ---
with st.form("prediction_form"):
    st.header("User Details")
    c1, c2 = st.columns(2)
    
    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", 10, 100, 25)
        weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0)
        height = st.number_input("Height (m)", 1.2, 2.5, 1.75)
    
    with c2:
        experience = st.slider("Experience Level", 1, 3, 2, help="1=Beginner, 2=Intermediate, 3=Expert")
        workout_freq = st.slider("Workout Frequency (days/week)", 1, 7, 3)
        water_intake = st.number_input("Water Intake (Liters)", 0.5, 10.0, 2.0)

    st.header("Workout Details")
    c3, c4 = st.columns(2)
    
    with c3:
        workout_type = st.selectbox("Workout Type", ["Cardio", "HIIT", "Strength", "Yoga"])
        duration = st.number_input("Duration (hours)", 0.1, 5.0, 1.0, step=0.1)
    
    with c4:
        avg_bpm = st.number_input("Avg BPM", 60, 200, 140)
        max_bpm = st.number_input("Max BPM", 80, 220, 180)
        resting_bpm = st.number_input("Resting BPM", 40, 100, 60)

    # Submit Button
    submit = st.form_submit_button("🔥 Predict Calories")

if submit:
    # 1. Calculate Derived Metrics (BMI, Fat %)
    bmi = weight / (height ** 2)
    # Simple estimation formula for Fat %
    fat_percentage = (1.20 * bmi) + (0.23 * age) - 16.2 
    if gender == 'Male':
        fat_percentage -= 10.8 # Correction for males

    # 2. Prepare Data Frame
    input_data = pd.DataFrame({
        'Session_Duration (hours)': [duration],
        'Experience_Level': [experience],
        'Workout_Frequency (days/week)': [workout_freq],
        'Water_Intake (liters)': [water_intake],
        'Age': [age],
        'Weight (kg)': [weight],
        'Height (m)': [height],
        'Max_BPM': [max_bpm],
        'Avg_BPM': [avg_bpm],
        'Resting_BPM': [resting_bpm],
        'Fat_Percentage': [fat_percentage],
        'BMI': [bmi],
        'Gender': [gender],
        'Workout_Type': [workout_type]
    })

    # 3. Encoding (Must match training data exactly)
    # We use pd.get_dummies and then align with saved model columns
    input_data_encoded = pd.get_dummies(input_data, columns=['Gender', 'Workout_Type'], drop_first=True)
    
    # Reindex ensures all columns from training exist here (filling missing with 0)
    input_data_ready = input_data_encoded.reindex(columns=model_columns, fill_value=0)

    # 4. Predict
    prediction = model.predict(input_data_ready)
    
    # 5. Display Result
    st.balloons()
    st.success(f"### You burned approximately **{int(prediction[0])} Calories**! 🚀")
    
    st.info(f"**Stats Used:** BMI: {bmi:.1f} | Est. Fat %: {fat_percentage:.1f}%")
    