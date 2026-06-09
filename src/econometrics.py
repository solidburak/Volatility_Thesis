import pandas as pd
import numpy as np
import logging
import warnings
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

# Silence statsmodels frequency and deprecation warnings
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
    Prepares the data for VAR modeling with strict Cholesky Ordering.
    """
    var_cols = macro_cols + [target_vol_col]
    var_data = df[var_cols].copy().dropna()
    return var_data

def test_granger_causality(var_data: pd.DataFrame, sector_name: str, max_lag: int = 5):
    """
    Tests if Macro Surprises Granger-cause Sector Volatility.
    """
    target = var_data.columns[-1]
    predictors = var_data.columns[:-1]
    
    print(f"\n--- Granger Causality: Macro -> {sector_name} ---")
    for predictor in predictors:
        test_data = var_data[[target, predictor]]
        try:
            gc_res = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
            print(f"[{predictor}]")
            for lag in range(1, max_lag + 1):
                p_value = gc_res[lag][0]['ssr_ftest'][1]
                significance = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""
                print(f"  Lag {lag}: p-value = {p_value:.4f} {significance}")
        except Exception as e:
            logger.error(f"Granger test failed for {predictor}: {e}")

def fit_var_model(var_data: pd.DataFrame, max_lags: int = 5, verbose: bool = False):
    """
    Strictly fits the VAR model and returns the results. No plotting.
    """
    model = VAR(var_data)
    lag_order_results = model.select_order(maxlags=max_lags)
    
    if verbose:
        print("\n" + "="*50)
        print("LAG SELECTION CRITERIA")
        print("="*50)
        print(lag_order_results.summary())
        
    optimal_lag = lag_order_results.aic
    if optimal_lag == 0:
        optimal_lag = 1
        
    results = model.fit(optimal_lag)
    return results, optimal_lag

def get_equation_stats(results, target_vol_col: str) -> pd.DataFrame:
    """
    Extracts the algebraic coefficients and p-values into a clean DataFrame.
    This allows the notebook to stitch multiple sectors side-by-side.
    """
    equation_stats = pd.DataFrame({
        'Coef': results.params[target_vol_col],
        'P-Val': results.pvalues[target_vol_col]
    })
    
    equation_stats['Sig'] = equation_stats['P-Val'].apply(
        lambda p: '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
    )
    
    return equation_stats.round(4)

def plot_irfs(results, var_data: pd.DataFrame, sector_name: str, irf_periods: int = 15):
    """
    Draws the Impulse Response Functions for a specific sector.
    """
    target_vol_col = var_data.columns[-1]
    macro_cols = var_data.columns[:-1]
    
    irf = results.irf(irf_periods)
    
    for macro_col in macro_cols:
        fig = irf.plot(impulse=macro_col, response=target_vol_col, orth=True, signif=0.05)
        fig.set_size_inches(10, 4)
        fig.suptitle(f"[{sector_name}] Shock to {macro_col} -> Effect on {target_vol_col}", fontsize=12, fontweight='bold')
        
        ax = fig.axes[0]
        ax.set_xlabel("Days After Macro Shock")
        ax.set_ylabel("Delta Volatility")
        ax.grid(axis='both', alpha=0.3)
        plt.tight_layout()
        plt.show()

def plot_historical_fit(results, var_data: pd.DataFrame, sector_name: str):
    """
    Plots the VAR predicted path vs actual path.
    """
    target_vol_col = var_data.columns[-1]
    actual = var_data[target_vol_col].iloc[results.k_ar:] 
    predicted = results.fittedvalues[target_vol_col]
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(actual.index, actual, color='black', alpha=0.5, label='Actual Volatility', linewidth=1)
    ax.plot(predicted.index, predicted, color='red', alpha=0.8, label='VAR Prediction', linewidth=1.5)
    
    ax.set_title(f"[{sector_name}] Historical Alignment: Actual vs Predicted", fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(axis='both', alpha=0.3)
    plt.tight_layout()
    plt.show()
