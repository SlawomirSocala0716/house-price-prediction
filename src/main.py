# Main: The Production Pipeline Orchestrator
"""
House Price Prediction: Production Pipeline Orchestrator
Process: Load -> Clean -> Feature Engineering -> Train Champion -> Evaluate -> Export
"""

# Import standard libraries
import pandas as pd
from sklearn.model_selection import train_test_split

# Import project-specific steps
from step1_load_data import load_processed_data
from step2_feature_engineering import add_features
from step3_data_cleaning import clean_data
from step4_split_train_test import split_data
from step6_model_training import train_hybrid_champion
from step7_evaluate import evaluate_model
from step8_save_model import save_model

def main():
    print("STARTING: Final Production Model Training")
    print("-" * 60)

    # ---------------------------------------------------------
    # STEP 1, 2 & 3: Data Preparation
    # ---------------------------------------------------------
    # Load the processed CSV
    df_raw = load_processed_data()
    
    # Clean records (remove price outliers and invalid houses)
    df_cleaned = clean_data(df_raw)
    
    # Apply Feature Engineering (Interaction terms and groupings)
    df_final = add_features(df_cleaned)
    
    print(f"Data Preparation Complete. Final features count: {df_final.shape[1] - 1}")

    # ---------------------------------------------------------
    #  STEP 4: Train/Test Split ---
    X_train, X_test, y_train, y_test = split_data(df_final)
    print(f"Data split complete: {len(X_train)} train / {len(X_test)} test records.")

    # ---------------------------------------------------------
    # STEP 5 & 6: Training the Champion
    # ---------------------------------------------------------
    print("\nBuilding Hybrid Ensemble (Linear Poly + Tuned XGBoost)...")
    # This function uses logic from step5 and parameters from laboratory tuning
    champion_model = train_hybrid_champion(X_train, y_train)
    print("Model training finished.")

    # ---------------------------------------------------------
    # STEP 7: Evaluation
    # ---------------------------------------------------------
    print("\nRunning Final Evaluation...")
    metrics = evaluate_model(champion_model, X_test, y_test)
    
    print("\n" + "="*45)
    print("FINAL PRODUCTION METRICS")
    print("-" * 45)
    print(f"R2 Score:      {metrics['R2']:.4f}")
    print(f"MAPE:          {metrics['MAPE (%)']:.2f}%")
    print(f"MAE (Error):   ${metrics['MAE']:,.2f}")
    print(f"RMSE:          ${metrics['RMSE']:,.2f}")
    print("="*45)

    # ---------------------------------------------------------
    # STEP 8: Exporting the Model
    # ---------------------------------------------------------
    # Saving the 'Champion' object to /models folder for the web backend
    model_filename = "house_price_hybrid_champion.joblib"
    save_model(champion_model, model_filename)
    
    print(f"\nSUCCESS: '{model_filename}' is ready for deployment!")

if __name__ == "__main__":
    main()