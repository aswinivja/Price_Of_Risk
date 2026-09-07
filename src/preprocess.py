import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

PROTECTED = ['fico_range_low', 'dti', 'revol_util', 'income_to_loan']

def winsorize_series(s, limits=(0.01, 0.99)):
    lower = s.quantile(limits[0])
    upper = s.quantile(limits[1])
    return s.clip(lower, upper)

def vif_select(df, target='int_rate', threshold=8.0, protected=None):
    if protected is None:
        protected = PROTECTED.copy()
    
    df = df.copy()
    num_cols = [c for c in df.select_dtypes('number').columns if c != target]
    dropped = []
    
    while True:
        X_vif = df[num_cols].dropna()
        if X_vif.shape[1] < 2:
            break
            
        vif_data = {}
        for i, col in enumerate(num_cols):
            try:
                vif_data[col] = variance_inflation_factor(X_vif.values, i)
            except:
                vif_data[col] = 1.0
        
        vif_series = pd.Series(vif_data).sort_values(ascending=False)
        droppable = vif_series[[c for c in vif_series.index if c not in protected]]
        if droppable.empty or droppable.max() <= threshold:
            break
        
        worst = droppable.idxmax()
        print(f'  Dropped {worst} (VIF={droppable[worst]:.1f})')
        dropped.append(worst)
        num_cols.remove(worst)
    
    keep = num_cols + [target] + list(df.select_dtypes('object').columns)
    keep = [c for c in keep if c in df.columns]
    return df[keep]

def preprocess_pipeline(df_raw, target='int_rate', sample_size=50000):
    df = df_raw.copy()
    df.columns = df.columns.str.strip()
    
    # Filter
    df = df[df['int_rate'].notna()]
    
    # Convert percentage strings to numbers
    df['revol_util'] = df['revol_util'].str.replace('%', '').str.strip().astype(float)
    df[target] = df[target].str.replace('%', '').str.strip().astype(float)
    
    # Create ratio
    df['income_to_loan'] = df['annual_inc'] / df['loan_amnt'].replace(0, np.nan)
    df['income_to_loan'] = df['income_to_loan'].replace([np.inf, -np.inf], np.nan)
    
    # Core features
    core_features = [
        'int_rate', 'loan_amnt', 'annual_inc', 'dti', 'fico_range_low',
        'revol_util', 'open_acc', 'delinq_2yrs', 'pub_rec', 'installment',
        'term', 'home_ownership', 'purpose', 'emp_length'
    ]
    keep_cols = [c for c in core_features if c in df.columns]
    keep_cols.append('income_to_loan')
    df = df[keep_cols]
    
    # Winsorize
    for col in df.select_dtypes('number').columns:
        if col != target:
            df[col] = winsorize_series(df[col])
    
    # Drop missing criticals
    df = df.dropna(subset=[target, 'fico_range_low', 'dti'])
    
    # VIF
    df = vif_select(df, target=target, threshold=8.0)
    
    # Final sample
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    print(f"Sample created: {df.shape}")
    return df
