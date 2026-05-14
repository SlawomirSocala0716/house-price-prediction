# Step 1: Loading dataset

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

def load_processed_data(filename="House_Rate_Data_processed.csv"):
    path = DATA_DIR / filename
    print(f"Dataset has been loaded from: {path}")
    return pd.read_csv(path)
