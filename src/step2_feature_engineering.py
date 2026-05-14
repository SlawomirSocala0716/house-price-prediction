# Step 2: Feature Engineering - creates new features, simplifies categorical variables and handles rare category values based on EDA insights.
#         Also help models understand non-linear relationships.

import pandas as pd

def add_features(df):
   
    df_eng = df.copy()

    # 1. Basic engineering
    df_eng['Has_Basement'] = (df_eng['Sqft_Basement'] > 0).astype(int)
    df_eng['Sqft_per_Floor'] = df_eng['Sqft_Living'] / df_eng['Floors']

    # 2. Advanced Interaction Features (The secret sauce)
    # Living Space Quality: Combine area with the house condition
    df_eng['Living_Quality_Index'] = df_eng['Sqft_Living'] * df_eng['Condition']
    
    # Bed/Bath Balance: Ratio of bedrooms to bathrooms
    df_eng['Bed_Bath_Ratio'] = df_eng['Bedrooms'] / (df_eng['Bathrooms'] + 0.1)
    
    # Average Room Size: Distribution of living space among rooms
    df_eng['Avg_Room_Size'] = df_eng['Sqft_Living'] / (df_eng['Bedrooms'] + df_eng['Bathrooms'] + 0.1)

    # 3. Mappings and Groupings
    view_map = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2}
    df_eng['View_Type'] = df_eng['view'].map(view_map)

    cond_map = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2}
    df_eng['Condition_Status'] = df_eng['Condition'].map(cond_map)
    
    # Capping bathrooms at 4.75 to handle extreme outliers
    df_eng['Bathrooms_grouped'] = df_eng['Bathrooms'].clip(upper=4.75)
    
    # Treating 3.5 floors as 3.0 to simplify numerical distribution
    df_eng['Floors_grouped'] = df_eng['Floors'].replace({3.5: 3.0})

    print("✅ Step 2: Features engineered (including Interaction Terms).")
    return df_eng