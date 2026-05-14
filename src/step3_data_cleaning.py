# Step 3: Refines the dataset by removing extreme outliers, potential data entry errors, and dropping columns that do not contribute to the model's predictive power.

import pandas as pd

def clean_data(df):

    df_cleaned = df.copy()

    # 1. PRICE FILTERING (The "MAPE-Fixer")
    # Removing homes above $5M (extreme luxury outliers) 
    # and below $50k (likely errors or non-habitable structures)
    # This range ensures the model focuses on the core residential market.
    initial_count = len(df_cleaned)
    df_cleaned = df_cleaned[
        (df_cleaned['Rates or Price'] >= 50000) & 
        (df_cleaned['Rates or Price'] <= 5000000)
    ]

    # 2. VALIDITY CHECKS
    # Removing records with 0 bedrooms, 0 bathrooms, or 0 living area.
    # A functional residential house must have at least one of each.
    df_cleaned = df_cleaned[
        (df_cleaned['Bedrooms'] > 0) & 
        (df_cleaned['Bathrooms'] > 0) & 
        (df_cleaned['Sqft_Living'] > 0)
    ]

    # 3. COLUMN REDUCTION
    # Dropping columns that are redundant (Sqft_Above is highly correlated with Sqft_Living)
    # or time-based (Date), which we are not using for this specific regression.
    columns_to_drop = ['Date', 'Sqft_Above']
    df_cleaned = df_cleaned.drop(columns_to_drop, errors='ignore')

    # 4. HANDLING MISSING VALUES
    # Final safety check to ensure no null values reach the training step
    df_cleaned = df_cleaned.dropna()

    removed_count = initial_count - len(df_cleaned)
    print(f"✅ Step 3: Cleaning complete. Removed {removed_count} invalid records.")
    print(f"📊 Final production dataset size: {len(df_cleaned)} records.")
    
    return df_cleaned