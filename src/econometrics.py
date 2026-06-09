import pandas as pd
import numpy as np
import logging
import warnings
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

# Silence statsmodels frequency and deprecation warnings for clean notebook output
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")
try:
    from statsmodels.tools.sm_exceptions import ValueWarning
    warnings.filterwarnings("ignore", category=ValueWarning)
except ImportError:
    pass

logger = logging.getLogger(__name__)

def prepare_var_data(df: pd.DataFrame, target_vol_col: str, macro_cols: list) -> pd.DataFrame:
    """
    Prepares the data for VAR modeling.
    CRITICAL: The order of columns dictates the Cholesky decomposition for IRFs.
    Macro variables must come BEFORE the financial asset volatility.
    """
    # Isolate the exact columns we need
    var_cols = macro_cols + [target_vol_col]
    var_data = df[var_cols].copy()
    
    # VAR cannot handle NaNs. Drop any remaining missing data.
    var_data = var_data.dropna()
    
    logger.info(f"VAR Data Prepared. Shape: {var_data.shape}")
    logger.info(f"Cholesky Ordering: {', '.join(var_cols)}")
    
    return var_data

def test_granger_causality(var_data: pd.DataFrame, max_lag: int = 5):
    """
    Tests if Macro Surprises Granger-cause Sector Volatility.
    Prints the p-values for the statistical tests.
    """
    logger.info("Running Granger Causality Tests...")
    
    target = var_data.columns[-1] # The volatility column
    predictors = var_data.columns[:-1] # The macro columns
    
    for predictor in predictors:
        print(f"\n--- Does {predictor} Granger-Cause {target}? ---")
        # We test if Predictor causes Target. The data array must be [Target, Predictor]
        test_data = var_data[[target, predictor]]
        
        # We run the test and suppress the verbose output, printing a clean summary instead
        try:
            gc_res = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
            for lag in range(1, max_lag + 1):
                # Using the SSR F-test p-value
                p_value = gc_res[lag][0]['ssr_ftest'][1]
                significance = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""
                print(f"Lag {lag}: p-value = {p_value:.4f} {significance}")
        except Exception as e:
            print(f"Could not compute Granger test for {predictor}: {e}")


def fit_var_and_plot_irf(var_data: pd.DataFrame, max_lags: int = 5, irf_periods: int = 15):
    """
    Fits the Vector Autoregression model and generates the Impulse Response Functions.
    Updated to expose the Lag Selection criteria and internal VAR mechanics.
    """
    logger.info("Fitting VAR Model...")
    model = VAR(var_data)
    
    # 1. Select the optimal number of lags and PRINT the intuitive table
    lag_order_results = model.select_order(maxlags=max_lags)
    print("\n" + "="*60)
    print("LAG SELECTION CRITERIA (The '*' indicates the optimal lag)")
    print("="*60)
    print(lag_order_results.summary())
    
    optimal_lag = lag_order_results.aic
    if optimal_lag == 0:
        logger.warning("AIC selected 0 lags. Forcing 1 lag to allow for dynamic plotting.")
        optimal_lag = 1
        
    # 2. Fit the model
    results = model.fit(optimal_lag)
    
    print("\n" + "="*60)
    print(f"VAR MODEL EQUATION SUMMARY (Lags Used: {optimal_lag})")
    print("="*60)
    
    target_vol_col = var_data.columns[-1]
    
    # Build a clean DataFrame specifically for the target volatility equation
    equation_stats = pd.DataFrame({
        'Coefficient': results.params[target_vol_col],
        'P-Value': results.pvalues[target_vol_col]
    })
    
    # Add an intuitive significance flag (*** = 1%, ** = 5%, * = 10%)
    equation_stats['Sig.'] = equation_stats['P-Value'].apply(
        lambda p: '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
    )
    
    # Print the clean table
    print(f"Target Equation: {target_vol_col}\n")
    print(equation_stats.round(4).to_string())    

    # 3. Generate Impulse Response Functions 
    irf = results.irf(irf_periods)
    macro_cols = var_data.columns[:-1]
    
    for macro_col in macro_cols:
        fig = irf.plot(impulse=macro_col, response=target_vol_col, orth=True, signif=0.05)
        fig.set_size_inches(10, 5)
        fig.suptitle(f"Impulse Response: Shock to {macro_col} -> Effect on {target_vol_col}", fontsize=14, fontweight='bold')
        
        ax = fig.axes[0]
        ax.set_xlabel("Days After Macro Shock")
        ax.set_ylabel("Change in Sector Volatility (Delta)")
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    return results, irf

def plot_historical_fit(results, var_data: pd.DataFrame):
    """
    Overlays the VAR model's predicted volatility (fitted values) against 
    the actual historical market volatility. Excellent for spotting crises.
    """
    target_vol_col = var_data.columns[-1]
    
    # Extract actual and predicted data
    actual = var_data[target_vol_col].iloc[results.k_ar:] # Shift by lag order
    predicted = results.fittedvalues[target_vol_col]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plotting
    ax.plot(actual.index, actual, color='black', alpha=0.5, label='Actual Volatility (Squared Returns)', linewidth=1)
    ax.plot(predicted.index, predicted, color='red', alpha=0.8, label='VAR Model Prediction', linewidth=1.5)
    
    ax.set_title(f"Historical Shock Alignment: Actual vs. VAR Predicted ({target_vol_col})", fontsize=14, fontweight='bold')
    ax.set_ylabel("Volatility")
    ax.legend(loc='upper left')
    ax.grid(axis='both', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
