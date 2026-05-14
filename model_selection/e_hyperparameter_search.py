from sklearn.model_selection import GridSearchCV

def tune_xgboost(transformed_model, X, y):
    """
    Searches for the best hyperparameters for the XGBoost model.
    transformed_model: a TransformedTargetRegressor instance created in training.py
    """

    
        # Parameters:
        # regressor__  -> enters the Pipeline inside the TransformedTargetRegressor
        # model__      -> enters the 'model' step (XGBRegressor) inside the Pipelin
    param_grid = {
        'regressor__model__n_estimators': [100, 500, 1000],
        'regressor__model__max_depth': [3, 5, 7],
        'regressor__model__learning_rate': [0.01, 0.05, 0.1],
        'regressor__model__subsample': [0.8, 1.0]
    }
    
    print("Starting hyperparameter search (GridSearch)...")
    
    # We use 3-fold cross-validation for faster hyperparameter search
    grid_search = GridSearchCV(
        transformed_model, 
        param_grid, 
        cv=3, 
        scoring='r2', 
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    return grid_search.best_params_, grid_search.best_score_