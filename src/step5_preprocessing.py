# Step 5: Preprocessing Pipeline (Linear Regression & Tree-Based Models)

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    RobustScaler, 
    OrdinalEncoder, 
    FunctionTransformer,
    PolynomialFeatures  # Added for the Linear component
)

# =================================================================
# 1. FEATURE GROUP DEFINITIONS
# =================================================================

# Updated to include your "Super-Features" from Step 2
continuous_features = [
    "Sqft_Living",
    "Sqft_lot",
    "Sqft_per_Floor",
    "Living_Quality_Index", # New interaction
    "Bed_Bath_Ratio",       # New ratio
    "Avg_Room_Size"         # New distribution
]

numeric_features_linear = ["Bedrooms"] 

ordinal_features_linear = [
    "View_Type", 
    "Condition_Status", 
    "Bathrooms_grouped", 
    "Floors_grouped"
]

binary_features = [
    "Has_Basement",
    "Waterfront"
]

# Raw categorical features for XGBoost
categorical_features_tree = [
    "Condition", 
    "view"
]

# All numerical features for XGBoost (including engineered ones)
tree_numeric_features = (
    continuous_features + 
    ["Bedrooms", "Bathrooms", "Floors"] + 
    binary_features
)

# =================================================================
# 2. PREPROCESSOR FUNCTIONS
# =================================================================

def create_linear_preprocessor():
    """
    Preprocessing for Linear Regression component:
    1. Log transform area-based features
    2. Add Polynomial Features (degree=2) to capture non-linear trends
    3. Scale using RobustScaler
    """
    
    # Advanced pipeline for continuous data
    log_poly_pipe = Pipeline([
        ("log", FunctionTransformer(np.log1p)),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)), # This boosted our R2!
        ("scale", RobustScaler())
    ])

    # Standard pipeline for other numerical data
    robust_pipe = Pipeline([
        ("scale", RobustScaler())
    ])

    return ColumnTransformer(
        transformers=[
            ("cont_poly", log_poly_pipe, continuous_features),
            ("num_ord_bin", robust_pipe, numeric_features_linear + ordinal_features_linear + binary_features),
        ],
        remainder="drop" 
    )


def create_tree_preprocessor():
    """
    Preprocessing for XGBoost component:
    - Pass numerical features through (trees handle raw scales well)
    - Ordinal Encoding for categories (prevents feature explosion)
    """
    
    ordinal_pipe = Pipeline([
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    return ColumnTransformer(
        transformers=[
            ("pass", "passthrough", tree_numeric_features),
            ("cat_tree", ordinal_pipe, categorical_features_tree)
        ],
        remainder="drop"
    )