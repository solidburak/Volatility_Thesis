import pandas as pd
import numpy as np
import warnings
import logging
from arch import arch_model

# --- Setup Module Logger ---
logger = logging.getLogger(__name__)

# Suppress arch convergence warnings to keep the logs clean during expanding windows
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def fit_static_models_for_research(returns_series: pd.Series) -> pd.DataFrame:
    """
    Fits GARCH(1,1), EGARCH, and GJR-GARCH on the entire dataset.
    Strictly for AIC/BIC comparison in EDA. Do not use outputs as ML features.
    """
    logger.info("Fitting static GARCH models for AIC/BIC comparison...")
    returns_scaled = returns_series.dropna() * 100
    results = {}
    
    # Standard GARCH
    garch = arch_model(returns_scaled, vol='GARCH', p=1, q=1, mean='Constant')
    garch_fit = garch.fit(disp='off', show_warning=False)
    results['GARCH(1,1)'] = {'AIC': garch_fit.aic, 'BIC': garch_fit.bic}
    
    # EGARCH (Exponential)
    egarch = arch_model(returns_scaled, vol='EGARCH', p=1, o=1, q=1, mean='Constant')
    egarch_fit = egarch.fit(disp='off', show_warning=False)
    results['EGARCH(1,1)'] = {'AIC': egarch_fit.aic, 'BIC': egarch_fit.bic}
    
    # GJR-GARCH (Threshold)
    gjr = arch_model(returns_scaled, vol='GARCH', p=1, o=1, q=1, mean='Constant')
    gjr_fit = gjr.fit(disp='off', show_warning=False)
    results['GJR-GARCH(1,1)'] = {'AIC': gjr_fit.aic, 'BIC': gjr_fit.bic}
    
    comparison_df = pd.DataFrame(results).T.sort_values(by='AIC')
    return comparison_df

def generate_expanding_garch(
    df: pd.DataFrame, 
    return_col: str = 'Log_Return', 
    window_size: int = 252, 
    refit_freq: int = 21
) -> pd.DataFrame:
    """
    Industry-Standard ML-Safe GARCH & EGARCH.
    Balances strict out-of-sample ML safety with computational speed by 
    re-fitting the optimizer periodically.
    """
    logger.info(f"Generating Expanding GARCH & EGARCH (Window: {window_size}, Refit: {refit_freq} days)...")
    df_local = df.copy()
    
    # Scale returns by 100 for optimizer convergence stability
    returns = df_local[return_col] * 100
    n_obs = len(returns)
    
    # Pre-allocate output series with NaNs
    vol_garch = pd.Series(np.nan, index=returns.index)
    vol_egarch = pd.Series(np.nan, index=returns.index)
    
    # Loop through the timeline in chunks of 'refit_freq'
    for i in range(window_size, n_obs, refit_freq):
        
        train_window = returns.iloc[i - window_size : i]
        chunk_end = min(i + refit_freq, n_obs)
        eval_data = returns.iloc[i - window_size : chunk_end]
        
        # -------------------------------------------------------------
        # A. STANDARD GARCH(1,1)
        # -------------------------------------------------------------
        train_mod_g = arch_model(train_window, vol='GARCH', p=1, q=1, mean='Constant', rescale=False)
        try:
            train_res_g = train_mod_g.fit(disp='off', show_warning=False)
            eval_mod_g = arch_model(eval_data, vol='GARCH', p=1, q=1, mean='Constant', rescale=False)
            fixed_res_g = eval_mod_g.fix(train_res_g.params)
            
            chunk_vol_g = np.asarray(fixed_res_g.conditional_volatility)[window_size:]
            vol_garch.iloc[i : chunk_end] = chunk_vol_g
        except Exception as e:
            logger.debug(f"GARCH optimizer failed at index {i}: {e}")
            pass
            
        # -------------------------------------------------------------
        # B. EGARCH(1,1) - Critical for the Leverage Effect
        # -------------------------------------------------------------
        train_mod_eg = arch_model(train_window, vol='EGARCH', p=1, o=1, q=1, mean='Constant', rescale=False)
        try:
            train_res_eg = train_mod_eg.fit(disp='off', show_warning=False)
            eval_mod_eg = arch_model(eval_data, vol='EGARCH', p=1, o=1, q=1, mean='Constant', rescale=False)
            fixed_res_eg = eval_mod_eg.fix(train_res_eg.params)
            
            chunk_vol_eg = np.asarray(fixed_res_eg.conditional_volatility)[window_size:]
            vol_egarch.iloc[i : chunk_end] = chunk_vol_eg
        except Exception as e:
            logger.debug(f"EGARCH optimizer failed at index {i}: {e}")
            pass

    # Unscale the final volatility to match original return decimals
    df_local['Vol_GARCH'] = vol_garch / 100
    df_local['Vol_EGARCH'] = vol_egarch / 100

    # -------------------------------------------------------------
    # C. DYNAMIC WINSORIZATION (Replaces the hard 0.20 cap)
    # -------------------------------------------------------------
    # Cap mathematical explosions at 3x the maximum absolute daily return 
    # seen in the trailing window. This handles crashes natively.
    trailing_max_return = df_local[return_col].abs().rolling(window=window_size, min_periods=window_size//2).max()
    dynamic_cap = trailing_max_return * 3.0
    
    df_local['Vol_GARCH'] = df_local['Vol_GARCH'].clip(upper=dynamic_cap)
    df_local['Vol_EGARCH'] = df_local['Vol_EGARCH'].clip(upper=dynamic_cap)
    
    final_df = df_local.dropna(subset=['Vol_GARCH', 'Vol_EGARCH'])
    logger.info(f"Out-of-sample generation complete. Shape: {final_df.shape}")
    
    return final_df
