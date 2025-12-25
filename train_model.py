import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def process_and_train():
    print("⏳ Loading dataset...")
    df = pd.read_csv('Final_data.csv')

    # --- 1. Data Cleaning ---
    # Drop irrelevant diet and leakage columns
    cols_to_drop = [
        'meal_name', 'meal_type', 'diet_type', 'sugar_g', 'sodium_mg', 
        'cholesterol_mg', 'serving_size_g', 'cooking_method', 'prep_time_min', 
        'cook_time_min', 'rating', 'Carbs', 'Proteins', 'Fats', 
        'Calories', 'Daily meals frequency', 'cal_from_macros', 
        'pct_carbs', 'protein_per_kg', 'cal_balance',
        'expected_burn', 'Burns Calories (per 30 min)_bc', 'BMI_calc', 'Name of Exercise'
    ]
    
    # Only drop columns that actually exist in the CSV
    existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df_cleaned = df.drop(columns=existing_cols_to_drop)
    
    print(f"✅ Data cleaned. Shape: {df_cleaned.shape}")

    # --- 2. Feature Selection ---
    features = [
        'Session_Duration (hours)', 'Experience_Level', 'Workout_Frequency (days/week)', 
        'Water_Intake (liters)', 'Age', 'Weight (kg)', 'Height (m)', 
        'Max_BPM', 'Avg_BPM', 'Resting_BPM', 'Fat_Percentage', 'BMI',
        'Gender', 'Workout_Type'
    ]
    
    target = 'Calories_Burned'
    
    X = df_cleaned[features]
    y = df_cleaned[target]

    # --- 3. Preprocessing (One-Hot Encoding) ---
    X = pd.get_dummies(X, columns=['Gender', 'Workout_Type'], drop_first=True)
    
    # Save the column names! We need this for the web app to match inputs exactly.
    model_columns = list(X.columns)
    joblib.dump(model_columns, 'model_columns.pkl')
    print("✅ Model columns saved.")

    # --- 4. Model Training ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("⏳ Training Random Forest Model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # --- 5. Evaluation ---
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(" Training Complete!")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   R² Score: {r2:.4f}")

    # --- 6. Save Model ---
    #joblib.dump(model, 'calories_burn_model.pkl')
    #print("💾 Model saved as 'calories_burn_model.pkl'")
    joblib.dump(model, 'calories_burn_model.pkl', compress=3)
    print("💾 Model saved compressed as 'calories_burn_model.pkl'")

if __name__ == "__main__":
    process_and_train()