# Step 8: Exports the trained model object to a .joblib file in the /models directory.
#         This file can later be loaded by the web backend for real-time predictions.

from pathlib import Path
import joblib

# Set up paths relative to this script's location (src folder)
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

def save_model(model, filename="house_price_hybrid_champion.joblib"):

    # 1. Ensure the /models/ directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Define the full save path
    save_path = MODELS_DIR / filename

    # 3. Serialize and save the model
    joblib.dump(model, save_path)

    print(f"Step 8: Model saved successfully to: {save_path}")