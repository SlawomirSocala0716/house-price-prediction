# Step 7: Model Evaluation - evaluates the model performance on the test set (log transformation handle by .predict())

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model performance on the test set.
    Since we use TransformedTargetRegressor, the model already handles 
    the inverse log transformation internally.
    """

    # 1. Get predictions (already in real currency thanks to TransformedTargetRegressor)
    y_pred = model.predict(X_test)

    # 2. Ensure y_test is also in real prices (if passed as raw prices from main.py)
    # y_test and y_pred are now directly comparable
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # 3. MAPE calculation using sklearn's built-in function
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE (%)": mape
    }
