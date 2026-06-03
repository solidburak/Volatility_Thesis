# src/volatility.py
import pandas as pd
import numpy as np
import warnings
from arch import arch_model

# Suppress convergence warnings to keep the console clean during expanding windows
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def fit_static_models_for_research(returns_series):
    """
    Fits GARCH(1,1), EGARCH, and GJR-GARCH on the entire dataset.
    DO NOT use these outputs for ML features (Data Leakage). 
    Use this strictly for AIC/BIC comparison in your EDA notebook.
    """
    print("Fitting static models for AIC/BIC comparison...")
    returns_scaled = returns_series.dropna() * 100
    results = {}
    
    # Standard GARCH
    garch = arch_model(returns_scaled, vol='GARCH', p=1, q=1, mean='Constant').fit(disp='off')
    results['GARCH(1,1)'] = {'AIC': garch.aic, 'BIC': garch.bic}
    
    # EGARCH (Exponential)
    egarch = arch_model(returns_scaled, vol='EGARCH', p=1, o=1, q=1, mean='Constant').fit(disp='off')
    results['EGARCH(1,1)'] = {'AIC': egarch.aic, 'BIC': egarch.bic}
    
    # GJR-GARCH (Threshold)
    gjr = arch_model(returns_scaled, vol='GARCH', p=1, o=1, q=1, mean='Constant').fit(disp='off')
    results['GJR-GARCH(1,1)'] = {'AIC': gjr.aic, 'BIC': gjr.bic}
    
    comparison_df = pd.DataFrame(results).T.sort_values(by='AIC')
    return comparison_df

def generate_expanding_garch(df, return_col='Log_Return', window_size=252, refit_freq=21):
    """
    INDUSTRY STANDARD ML-SAFE GARCH & EGARCH.
    Balances strict out-of-sample ML safety with computational speed by 
    re-fitting the optimizer periodically (e.g., every 21 days) rather than daily.
    """
    print(f"Generating Industry Standard GARCH & EGARCH (Window: {window_size}, Refit: {refit_freq} days)...")
    df = df.copy()
    
    # Scale returns by 100 for optimizer convergence
    returns = df[return_col] * 100
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
            
            # FIX= Force to numpy array, safely slice, and assign
            chunk_vol_g = np.asarray(fixed_res_g.conditional_volatility)[window_size:]
            vol_garch.iloc[i : chunk_end] = chunk_vol_g
        except Exception:
            pass
            
        # -------------------------------------------------------------
        # B. EGARCH(1,1) - Critical for the Leverage Effect
        # -------------------------------------------------------------
        train_mod_eg = arch_model(train_window, vol='EGARCH', p=1, o=1, q=1, mean='Constant', rescale=False)
        try:
            train_res_eg = train_mod_eg.fit(disp='off', show_warning=False)
            
            eval_mod_eg = arch_model(eval_data, vol='EGARCH', p=1, o=1, q=1, mean='Constant', rescale=False)
            fixed_res_eg = eval_mod_eg.fix(train_res_eg.params)
            
            # FIX= Force to numpy array, safely slice, and assign
            chunk_vol_eg = np.asarray(fixed_res_eg.conditional_volatility)[window_size:]
            vol_egarch.iloc[i : chunk_end] = chunk_vol_eg
        except Exception:
            pass

    # Unscale the final volatility to match original return decimals
    vol_garch_unscaled = vol_garch / 100
    vol_egarch_unscaled = vol_egarch / 100

    # Cap OOS mathematical explosions at a realistic historical maximum (20% daily vol)
    df['Vol_GARCH'] = vol_garch_unscaled.clip(upper=0.20)
    df['Vol_EGARCH'] = vol_egarch_unscaled.clip(upper=0.20)
    
    print("Out-of-sample GARCH/EGARCH generation complete.")
    
    return df.dropna(subset=['Vol_GARCH', 'Vol_EGARCH'])
