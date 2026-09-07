#quick_sample.py 
import pandas as pd
import sys
sys.path.append('src')
from preprocess import preprocess_pipeline

print("Loading raw data...")
df_raw = pd.read_csv('data/raw/Loan_status_2007-2020Q3.gzip', low_memory=False)

print(f"Raw data shape: {df_raw.shape}")

df_processed = preprocess_pipeline(df_raw, target='int_rate', sample_size=50000)

df_processed.to_csv('data/processed/lc_sample_50k_processed.csv', index=False)
print("\n✓ Saved to data/processed/lc_sample_50k_processed.csv")
print(f"\nFinal shape: {df_processed.shape}")
print(f"Columns: {list(df_processed.columns)}")