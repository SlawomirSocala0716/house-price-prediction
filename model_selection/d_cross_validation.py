from sklearn.model_selection import cross_validate

def evaluate_stability(name, model, X, y):
    scoring = {
        'rmse': 'neg_root_mean_squared_error',
        'mae': 'neg_mean_absolute_error',
        'mape': 'neg_mean_absolute_percentage_error',
        'r2': 'r2'
    }

    scores = cross_validate(model, X, y, cv=5, scoring=scoring, n_jobs=-1)

    rmse_scores = -scores['test_rmse']
    mae_scores = -scores['test_mae']
    mape_scores = -scores['test_mape']
    r2_scores = scores['test_r2']

    return {
        "Model": name,
        "R2_mean": r2_scores.mean(),
        "MAE_mean": mae_scores.mean(),
        "RMSE_mean": rmse_scores.mean(),
        "RMSE_std": rmse_scores.std(), # Added this line!
        "MAPE_mean (%)": mape_scores.mean() * 100
    }
