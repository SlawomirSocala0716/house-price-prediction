import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "House_Rate_Data_processed.csv"

def load_and_clean_data():
    df = pd.read_csv(DATA_PATH)

    # Cleaning
    df = df[(df['Rates or Price'] >= 50000) & (df['Rates or Price'] <= 5000000)]
    df = df[(df['Bedrooms'] > 0) & (df['Bathrooms'] > 0)]
    df = df[df['Sqft_Living'] > 0] # Every house must have some living space
    df = df.drop(columns=['Date', 'Sqft_Above'], errors='ignore') 

    # a)Feature Engineering
    df_eng = df.copy()
    # Explicitly cast to int for Linear Regression compatibility
    df_eng['Has_Basement'] = (df_eng['Sqft_Basement'] > 0).astype(int)   
    df_eng['Sqft_per_Floor'] = df_eng['Sqft_Living'] / df_eng['Floors']

    view_map = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2}
    df_eng['View_Type'] = df_eng['view'].map(view_map)
    
    cond_map = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2}
    df_eng['Condition_Status'] = df_eng['Condition'].map(cond_map)
    
    df_eng['Bathrooms_grouped'] = df_eng['Bathrooms'].clip(upper=4.75)
    df_eng['Floors_grouped'] = df_eng['Floors'].replace({3.5: 3.0})

    # b) NEW FE
    # Living Space Quality Index
    df_eng['Living_Quality_Index'] = df_eng['Sqft_Living'] * df_eng['Condition']

    # Bed_Bath_Ratio
    df_eng['Bed_Bath_Ratio'] = df_eng['Bedrooms'] / (df_eng['Bathrooms'] + 0.1)

    #Total_Rooms_Area
    df_eng['Avg_Room_Size'] = df_eng['Sqft_Living'] / (df_eng['Bedrooms'] + df_eng['Bathrooms'] + 0.1)

    
    print(f"Data ready. Records: {len(df_eng)}")
    return df_eng