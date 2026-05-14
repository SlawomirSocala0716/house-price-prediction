import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add current directory to sys.path to ensure local imports work correctly
sys.path.append(str(Path(__file__).resolve().parent))

from model_selection.a_data_utils import load_and_clean_data
from model_selection.c_training import get_models
from model_selection.d_cross_validation import evaluate_stability
from model_selection.f_model_comparison import show_results
from model_selection.e_hyperparameter_search import tune_xgboost



def run_model_selection():
    print("START: Initiating Final Model Battle")
    print("-" * 60)

    # 1. DATA PREPARATION (Now includes Interaction Features!)
    try:
        df = load_and_clean_data()
        X = df.drop(columns=["Rates or Price"])
        y = df["Rates or Price"]
        print(f"Data ready with New Features: {X.shape[1]} features total.")
    except Exception as e:
        print(f"Data Error: {e}")
        return

    # 2. SYSTEM COMPARISON
    print("\n--- STEP 1: Comparing Linear, Tree, and Hybrid Systems ---")
    models = get_models()
    results = []
    
    for name, model in models.items():
        print(f"Testing {name}...")
        res = evaluate_stability(name, model, X, y)
        results.append(res)
    
    show_results(results)

    # 3. FINAL TUNING (Focusing on the Champion)
    # We choose Hybrid_Ensemble because it's usually the most stable
    print("\n--- STEP 2: Hyperparameter Tuning (XGBoost Component) ---")
    try:
        # Tuning only the XGBoost part of the hybrid or standalone
        best_p, best_s = tune_xgboost(models["XGBoost"], X, y)
        print(f"\nBest XGBoost Params: {best_p}")
        print(f"Best CV R2: {best_s:.4f}")
    except Exception as e:
        print(f"Tuning Error: {e}")

if __name__ == "__main__":
    run_model_selection()