import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    RobustScaler, 
    OrdinalEncoder, 
    FunctionTransformer, 
    PolynomialFeatures
)

# --- Feature Group Definitions ---

# Continuous features (including our new "Super-Features")
continuous_features = [
    "Sqft_Living", 
    "Sqft_lot", 
    "Sqft_per_Floor", 
    "Living_Quality_Index", 
    "Bed_Bath_Ratio", 
    "Avg_Room_Size"
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

categorical_features_tree = [
    "Condition", 
    "view"
]

# For Trees: All numeric/engineered features + original categorical mapped via Ordinal
tree_numeric_features = continuous_features + ["Bedrooms", "Bathrooms", "Floors"] + binary_features

# --- Preprocessor Creators ---

def create_linear_preprocessor():
    """
    Logic for Linear Models:
    1. Log1p to handle skewness.
    2. PolynomialFeatures to capture non-linear relationships.
    3. RobustScaler to keep everything in check.
    """
    log_poly_pipe = Pipeline([
        ("log", FunctionTransformer(np.log1p)),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scale", RobustScaler())
    ])
    
    robust_pipe = Pipeline([("scale", RobustScaler())])

    return ColumnTransformer(transformers=[
        ("cont", log_poly_pipe, continuous_features),
        ("num_ord_bin", robust_pipe, numeric_features_linear + ordinal_features_linear + binary_features),
    ], remainder="drop")

def create_tree_preprocessor():
    """
    Logic for Tree Models (XGBoost):
    Simple passthrough for numbers and Ordinal Encoding for categories.
    """
    ordinal_pipe = Pipeline([
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])
    
    return ColumnTransformer(transformers=[
        ("pass", "passthrough", tree_numeric_features),
        ("cat_tree", ordinal_pipe, categorical_features_tree)
    ], remainder="drop")
