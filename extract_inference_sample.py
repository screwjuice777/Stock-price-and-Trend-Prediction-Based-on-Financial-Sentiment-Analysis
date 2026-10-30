"""
Extract 50,000 unseen rows from stock_with_all_features.csv
starting AFTER the 1,500,000 rows used for training and testing.

These rows have never been seen by XGBoost or LSTM.
The output file is ready to feed directly into pipeline_2_batch_inference.py (Option 2).
"""
import pandas as pd
import numpy as np

SKIP_ROWS     = 1_500_000   # Rows used for training/testing - skip all of these
EXTRACT_ROWS  = 10_000      # How many fresh unseen rows to extract
OUTPUT_FILE   = 'inference_sample.csv'
SOURCE_FILE   = 'stock_with_all_features.csv'

print(f"Loading {EXTRACT_ROWS} unseen rows from beyond row {SKIP_ROWS:,} in {SOURCE_FILE}...")

# Read the header first to know column names
header = pd.read_csv(SOURCE_FILE, nrows=0)
col_names = list(header.columns)

# Now skip the first SKIP_ROWS data rows and read next EXTRACT_ROWS rows
df_unseen = pd.read_csv(
    SOURCE_FILE,
    skiprows=range(1, SKIP_ROWS + 1),   # Skip header=0 is kept, skip rows 1..1500000
    nrows=EXTRACT_ROWS,
    header=0,                            # Use the original header
    parse_dates=['Date']
)

print(f"Raw extracted shape: {df_unseen.shape}")
print(f"Date range: {df_unseen['Date'].min()} to {df_unseen['Date'].max()}")
print(f"Unique stocks: {df_unseen['Stock'].nunique()}")

# Compute the Target column (3-day forward return) so pipeline_2 can measure accuracy
df_unseen = df_unseen.sort_values(['Stock', 'Date']).reset_index(drop=True)
df_unseen['forward_return_3d'] = df_unseen.groupby('Stock')['Daily_Return'].shift(-1).rolling(3, min_periods=1).sum()
df_unseen['Target'] = (df_unseen['forward_return_3d'] > 0).astype(int)

# Drop the intermediate forward return column (keep Target)
df_unseen = df_unseen.drop(columns=['forward_return_3d'])

# Drop rows where Target could not be computed
df_unseen = df_unseen.dropna(subset=['Target'])
df_unseen['Target'] = df_unseen['Target'].astype(int)

# Keep only stocks with enough data to form at least one 60-day LSTM sequence
valid_stocks = df_unseen.groupby('Stock').filter(lambda x: len(x) > 60)['Stock'].unique()
df_unseen = df_unseen[df_unseen['Stock'].isin(valid_stocks)].copy()

print(f"\nFinal inference-ready shape: {df_unseen.shape}")
print(f"Stocks with enough data (> 60 rows): {df_unseen['Stock'].nunique()}")
print(f"Target distribution: UP = {(df_unseen['Target']==1).mean()*100:.1f}%, DOWN = {(df_unseen['Target']==0).mean()*100:.1f}%")

df_unseen.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved: {OUTPUT_FILE}")
print("This file is ready to use with:")
print(" -> python pipeline_2_batch_inference.py   (select Option 2)")
