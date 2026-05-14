import pandas as pd

def show_results(results):
    """
    # Presents the results of the model comparison in a clear, readable table
    """
    df = pd.DataFrame(results)
    
    # Sort models by R2 score from highest to lowest (best-performing model first)
    df = df.sort_values(by="R2_mean", ascending=False)

    print("\n" + "="*95)
    print("                     FINAL REPORT: MACHINE LEARNING SYSTEM COMPARISON")
    print("="*95)
    
    # Headlines for better readability
    headers = {
        "Model": "Model Name",
        "R2_mean": "R2 Score",
        "MAE_mean": "MAE ($)",
        "RMSE_mean": "RMSE ($)",
        "RMSE_std": "RMSE Std ($)",
        "MAPE_mean (%)": "MAPE (%)"
    }
    
    # Copy the DataFrame for display with clean, nicely formatted output
    display_df = df.rename(columns=headers).copy()
    
    # Number formatting
    # Prices with commas and two decimal places, percentages with a % sign
    format_map = {
        "R2 Score": "{:.4f}",
        "MAE ($)": "{:,.2f}",
        "RMSE ($)": "{:,.2f}",
        "RMSE Std ($)": "{:,.2f}",
        "MAPE (%)": "{:.2f}%"
    }
    
    for col, fmt in format_map.items():
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: fmt.format(x))

    print(display_df.to_string(index=False))
    print("="*95)
    
    best_model = df.iloc[0]['Model']
    best_mape = df.iloc[0]['MAPE_mean (%)']
    
    print(f" RESULT: Achieves the highest stability and precision (MAPE: {best_mape:.2f}%) revealed {best_model}.")
    print("="*95 + "\n")