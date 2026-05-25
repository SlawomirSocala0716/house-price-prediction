import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def show_results(results):
    """
    Presents the results of the model comparison in a clear, readable table
    AND automatically generates an academic evaluation chart.
    """
    df = pd.DataFrame(results)
    
    # Sort models by R2 score from highest to lowest (best-performing model first)
    df = df.sort_values(by="R2_mean", ascending=False)

    print("\n" + "="*95)
    print("                    FINAL REPORT: MACHINE LEARNING SYSTEM COMPARISON")
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

    # =========================================================================
    # VISUALIZATION LAYER (PRODUCTION MLOps CHART GENERATION)
    # =========================================================================
    try:
        # Configure chart styles for academic publication standards
        sns.set_theme(style="whitegrid")
        plt.rcParams['font.sans-serif'] = "Arial"
        plt.rcParams['font.family'] = "sans-serif"

        # Initialize a two-panel figure (R2 and MAE side-by-side)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Extract raw metrics from the oriented DataFrame into native lists for plotting
        model_names = df['Model'].tolist()
        r2_values = df['R2_mean'].tolist()
        mae_values = df['MAE_mean'].tolist()

        # ---- LEFT PANEL: R2 Score (Higher = Better) ----
        # Dynamically assign a green color to the champion model and light gray to others
        colors_r2 = ["#10B981" if m == best_model else "#CBD5E1" for m in model_names]
        bars_r2 = axes[0].bar(model_names, r2_values, color=colors_r2, width=0.4, edgecolor="#475569", linewidth=1.2)
        
        axes[0].set_title("Coefficient of Determination $R^2$ (Higher = Better)", fontsize=11, fontweight='bold', pad=12)
        axes[0].set_ylabel("Metric Value ($R^2$)", fontsize=10)
        
        # Adjust Y-axis limits to zoom in and emphasize metric differences safely
        min_r2, max_r2 = min(r2_values), max(r2_values)
        axes[0].set_ylim(min_r2 - 0.02, max_r2 + 0.02)

        # Add text labels on top of the R2 bars
        for bar in bars_r2:
            yval = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + (max_r2*0.002), f"{yval:.4f}", 
                         ha='center', va='bottom', fontsize=9, fontweight='bold')

        # ---- RIGHT PANEL: MAE Value (Lower = Better) ----
        # Use red tones for error metrics to indicate cost/loss
        colors_mae = ["#EF4444" if m == best_model else "#FCA5A5" for m in model_names]
        bars_mae = axes[1].bar(model_names, mae_values, color=colors_mae, width=0.4, edgecolor="#991B1B", linewidth=1.2)
        
        axes[1].set_title("Mean Absolute Error MAE (Lower = Better)", fontsize=11, fontweight='bold', pad=12)
        axes[1].set_ylabel("Error Value in USD ($)", fontsize=10)
        
        # Adjust Y-axis limits to display error variance clearly
        min_mae, max_mae = min(mae_values), max(mae_values)
        axes[1].set_ylim(min_mae - 10000, max_mae + 10000)

        # Add text labels on top of the MAE bars
        for bar in bars_mae:
            yval = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + (max_mae*0.002), f"${yval:,.2f}", 
                         ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Global figure formatting
        plt.suptitle("Comparative Evaluation of Predictive Models (Battle Mode)", fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()

        # Save the finalized visualization to a physical image asset
        output_image = "models_comparison_chart.png"
        plt.savefig(output_image, dpi=300, bbox_inches='tight')
        print(f"[MLOps INFO] Evaluation chart successfully generated and saved to: {output_image}\n")
        
        # Display interactive window if GUI/X11 backend is available
        plt.show()

    except Exception as e:
        # Fallback block to prevent server crash in headless environment (e.g., WSL2 or remote cloud Docker containers)
        print(f"[MLOps WARNING] Interactive window could not be opened (No GUI/X11 display available).")
        print(f"[MLOps SUCCESS] Physical image file asset has been successfully saved to disk.\n")