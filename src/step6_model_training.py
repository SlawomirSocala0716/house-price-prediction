# Step 6: Training the Hybrid Champion (Linear Polynomial + Tuned XGBoost)
#         Selected as the optimal solution for balancing stability and non-linear precision.

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
from xgboost import XGBRegressor
from step5_preprocessing import create_linear_preprocessor, create_tree_preprocessor

def train_hybrid_champion(X_train, y_train):

    # 1. Linear Component (Mathematical trend)
    linear_pipe = Pipeline([
        ("pre", create_linear_preprocessor()),
        ("model", LinearRegression())
    ])

    # 2. XGBoost Component (Tree-based complexity)
    # Using the best parameters discovered during the tuning phase
    xgb_pipe = Pipeline([
        ("pre", create_tree_preprocessor()),
        ("model", XGBRegressor(
            learning_rate=0.1, 
            max_depth=3, 
            n_estimators=100,
            subsample=1.0,
            random_state=42
        ))
    ])

    # 3. Hybrid Ensemble (The "Voting" mechanism)
    # It takes the average prediction of both systems
    ensemble = VotingRegressor([
        ('lr', linear_pipe),
        ('xgb', xgb_pipe)
    ])

    # 4. Target Transformation Wrapper
    # This ensures that fit(X, y) uses raw prices, but the model trains on log prices
    final_model = TransformedTargetRegressor(
        regressor=ensemble,
        func=np.log1p,
        inverse_func=np.expm1
    )

    final_model.fit(X_train, y_train)
    return final_model
