# STEP 9: Real-time Prediction Module (Inference Script).
#         This module serves as the final production interface for the House Price Prediction system.

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Importing your feature engineering logic from Step 2
from src.step2_feature_engineering import add_features

# Path configuration - Updated to match your champion model name
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "house_price_hybrid_champion.joblib"

# =================================================================
# REQUIRED FEATURES FROM USER
# =================================================================
# These 9 features represent the inputs in your future web form.
REQUIRED_INPUTS = [
    'Bedrooms', 
    'Bathrooms', 
    'Sqft_Living', 
    'Sqft_lot', 
    'Floors', 
    'Waterfront', 
    'view', 
    'Condition', 
    'Sqft_Basement'
]

# Global variable to hold the model (Performance boost: load it once)
_MODEL = None

def load_trained_model():
    """Loads the saved Pipeline (Processor + Model) once."""
    global _MODEL
    if _MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"❌ Error: Model file not found at {MODEL_PATH}")
        _MODEL = joblib.load(MODEL_PATH)
    return _MODEL

def predict_price(input_data: dict):
    """
    Receives house data, processes it, and returns the estimated market price.
    """
    # 1. Conversion: Dictionary -> DataFrame
    # Using [input_data] to create a single-row DataFrame
    df_raw = pd.DataFrame([input_data])

    # 2. Validation: Ensure all required columns are present
    for feature in REQUIRED_INPUTS:
        if feature not in df_raw.columns:
            raise KeyError(f"❌ ERROR: Missing column '{feature}' in input data!")

    # 3. Feature Engineering: Create interactions (Living_Quality_Index, etc.)
    # This matches the data structure the model was trained on.
    df_processed = add_features(df_raw)

    # 4. Load the Champion Model
    model_pipeline = load_trained_model()

    # 5. Prediction
    # IMPORTANT: Our Champion uses TransformedTargetRegressor.
    # .predict() already returns the result in REAL DOLLARS (expm1 is automatic).
    prediction = model_pipeline.predict(df_processed)

    # Returning as a float, rounded to 2 decimal places
    return round(float(prediction[0]), 2)

# =================================================================
# EXAMPLE USAGE (For testing purposes)
# =================================================================
if __name__ == "__main__":
    # Sample house data (exactly as it will come from your Web Form)
    test_house = {
        'Bedrooms': 3,
        'Bathrooms': 2.5,
        'Sqft_Living': 2100,
        'Sqft_lot': 5000,
        'Floors': 2.0,
        'Waterfront': 0,
        'view': 0,
        'Condition': 3, # Fixed typo from 'Condtion'
        'Sqft_Basement': 800
    }

    print("🏠 --- HOUSE PRICE PREDICTOR ---")
    try:
        price = predict_price(test_house)
        print(f"✅ Success! Predicted Market Value: ${price:,.2f}")
    except Exception as e:
        print(f"⚠️  An error occurred: {e}")