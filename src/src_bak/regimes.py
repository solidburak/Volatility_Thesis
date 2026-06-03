# src/regimes.py
import pandas as pd
import numpy as np
from hmmlearn import hmm
import warnings

# Suppress the tiny convergence warning from hmmlearn to keep console clean
warnings.filterwarnings("ignore", category=UserWarning, module='hmmlearn')

def generate_smoothed_targets(df, lower_quant=0.86, upper_quant=0.95, window=252, floor=0.0, shock_smooth=3, elevated_smooth=4):
    """
    Independent version of the heuristic 10-day smoothed target generator.
    """
    print("Generating rolling regimes and smoothed targets...")
    df_local = df.copy()
    
    # 1. Target Variable: Tomorrow's Volatility
    df_local['Future_Sq_Return'] = df_local['Sq_Log_Return'].shift(-1)
    
    # 2. Dynamic Thresholds
    df_local['Rolling_Thresh_Low'] = df_local['Sq_Log_Return'].rolling(window=window).quantile(lower_quant)
    df_local['Rolling_Thresh_High'] = df_local['Sq_Log_Return'].rolling(window=window).quantile(upper_quant)
    
    # Drop rows missing threshold baselines or future returns
    df_local = df_local.dropna(subset=['Rolling_Thresh_Low', 'Rolling_Thresh_High', 'Future_Sq_Return'])
    
    # 3. Vectorized Classification
    conditions = [
        (df_local['Future_Sq_Return'] <= df_local['Rolling_Thresh_Low']) | (df_local["Future_Sq_Return"] <= floor),
        (df_local['Future_Sq_Return'] > df_local['Rolling_Thresh_Low']) & (df_local['Future_Sq_Return'] <= df_local['Rolling_Thresh_High']),
        df_local['Future_Sq_Return'] > df_local['Rolling_Thresh_High']
    ]
    
    choices = [0, 1, 2] 
    df_local['Target_Regime'] = np.select(conditions, choices, default=np.nan)
    df_local = df_local.dropna(subset=['Target_Regime'])
    df_local['Target_Regime'] = df_local['Target_Regime'].astype(int)

    # 4. Target Smoothing (Your Custom Density Logic)
    is_shock = (df_local['Target_Regime'] == 2).astype(int)
    is_elevated = (df_local['Target_Regime'] == 1).astype(int)

    shock_density = is_shock.rolling(10).sum()
    elevated_density = is_elevated.rolling(10).sum()

    df_local['Target_Smooth_10d'] = 0
    df_local.loc[elevated_density >= elevated_smooth, 'Target_Smooth_10d'] = 1
    df_local.loc[shock_density >= shock_smooth, 'Target_Smooth_10d'] = 2

    # Drop early warmup rows for density calculation
    df_local = df_local.dropna(subset=['Target_Smooth_10d'])
    
    # Return ONLY the final target column mapped to the date index
    return df_local[['Target_Smooth_10d']]


def generate_hmm_targets(df, feature_cols=['Log_Return', 'Vol_EGARCH'], n_components=3):
    """
    Independent statistical HMM target generator with safe type-casting.
    """
    print(f"\n--- HMM Generation Process ---")
    df_local = df.copy()
    
    # 1. Isolate clean, contiguous historical sequence
    hmm_data = df_local.dropna(subset=feature_cols).copy()
    X = hmm_data[feature_cols].values
    
    # 2. Fit Gaussian HMM
    model = hmm.GaussianHMM(
        n_components=n_components, 
        covariance_type="full", 
        n_iter=1000, 
        random_state=42
    )
    
    model.fit(X)
    hidden_states = model.predict(X)
    
    # Force numpy output into standard python integers
    hmm_data['Temp_State'] = [int(x) for x in hidden_states]
    
    # 3. Dynamic State Sorting (0=Calm, 1=Elevated, 2=Shock)
    state_vols = {}
    for i in range(n_components):
        # Calculate mean vol, default to -1 if state is somehow empty
        mean_vol = hmm_data[hmm_data['Temp_State'] == i]['Vol_EGARCH'].mean()
        state_vols[i] = mean_vol if pd.notna(mean_vol) else -1.0
        print(f"HMM State {i} Average Volatility: {state_vols[i]:.4f}")
        
    # Sort states by their mean volatility
    sorted_states = sorted(list(state_vols.keys()), key=lambda k: state_vols[k])
    print(f"Mapping Order (Lowest Vol to Highest Vol): {sorted_states}")
    
    # Create strictly typed mapping dictionary
    mapping = {int(old_label): int(new_label) for new_label, old_label in enumerate(sorted_states)}
    
    # Apply mapping using a safe lambda function
    hmm_data['Target_HMM'] = hmm_data['Temp_State'].apply(lambda x: mapping.get(x, np.nan))
    
    # 4. Shift Target Backward (Predicting Tomorrow's State)
    hmm_data['Target_HMM'] = hmm_data['Target_HMM'].shift(-1)
    hmm_data = hmm_data.dropna(subset=['Target_HMM'])
    hmm_data['Target_HMM'] = hmm_data['Target_HMM'].astype(int)
    
    print(f"HMM Target generation complete. Shape: {hmm_data[['Target_HMM']].shape}\n")
    
    return hmm_data[['Target_HMM']]
