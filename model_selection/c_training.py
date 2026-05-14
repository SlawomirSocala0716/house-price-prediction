import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor # NEW

from model_selection.b_preprocessing import create_linear_preprocessor, create_tree_preprocessor

def get_models():
    # 1. Linear System (with Polynomials)
    linear_pipe = Pipeline([
        ("pre", create_linear_preprocessor()),
        ("model", LinearRegression())
    ])

    # 2. XGBoost System (with Tuning from last run)
    xgb_pipe = Pipeline([
        ("pre", create_tree_preprocessor()),
        ("model", XGBRegressor(
            n_estimators=1000,
            max_depth=3,
            learning_rate=0.01,
            subsample=0.8,
            random_state=42
        ))
    ])

    # 3. Hybrid System (The "Ensemble")
    # This averages the predictions of both models
    ensemble_model = VotingRegressor([
        ('lr', linear_pipe),
        ('xgb', xgb_pipe)
    ])

    return {
        "Linear_Regression": TransformedTargetRegressor(
            regressor=linear_pipe, func=np.log1p, inverse_func=np.expm1
        ),
        "XGBoost": TransformedTargetRegressor(
            regressor=xgb_pipe, func=np.log1p, inverse_func=np.expm1
        ),
        "Hybrid_Ensemble": TransformedTargetRegressor(
            regressor=ensemble_model, func=np.log1p, inverse_func=np.expm1
        )
    }