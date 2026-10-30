import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

print("\n     DATA PREPARATION      ")

# Load data - Huge subset exclusively to feed XGBoost deeper intelligence
df = pd.read_csv('stock_with_all_features.csv', nrows=1500000, parse_dates=['Date'])

# Sort by Stock then by Date chronologically
df = df.sort_values(['Stock', 'Date']).reset_index(drop=True)

# Keep the first 600 unique stocks for XGBoost (Massive Data)
unique_stocks = df['Stock'].unique()[:600]
df = df[df['Stock'].isin(unique_stocks)].copy()

print(f"Loaded {len(df)} rows across {len(unique_stocks)} stocks for XGBoost Phase.")

# Create binary target 3-day forward return
df['forward_return_3d'] = df.groupby('Stock')['Daily_Return'].shift(-1).rolling(3, min_periods=1).sum()
df['Target'] = (df['forward_return_3d'] > 0).astype(int)  # 1=UP, 0=DOWN
df = df.dropna(subset=['Target'])

# Define features to drop 
# 1. Base info cols
drop_cols = ['Date', 'Stock', 'Open', 'High', 'Low', 'Volume', 'Change Pct', 'Daily_Return', 'forward_return_3d']

# 2. Noisy lag features (they overlap too much and cause overfitting)
noisy_lags = [f'return_lag_{i}' for i in [1, 2, 3, 5]] + \
             [f'close_lag_{i}' for i in [1, 5, 10]] + \
             [f'sentiment_compound_lag_{i}' for i in [1, 2, 3]] + \
             [f'sentiment_positive_lag_{i}' for i in [1, 2, 3]] + \
             [f'sentiment_negative_lag_{i}' for i in [1, 2, 3]]

drop_cols.extend(noisy_lags)

# Select actual feature columns
feature_cols = [col for col in df.columns if col not in drop_cols and col != 'Target']

print(f"Total clean features selected: {len(feature_cols)}")
print(f"Features are:\n{feature_cols}")

# Chronological train/test split (e.g., first 80% of days for train, last 20% for test per stock)
# To prevent data leakage across stocks, we split based on a global date cutoff
split_idx = int(len(df) * 0.8)
split_date = df['Date'].sort_values().iloc[split_idx]

train_df = df[df['Date'] <= split_date].copy()
test_df = df[df['Date'] > split_date].copy()

print(f"\nTime-based Split -> Cutoff Date: {split_date.date()}")
print(f"Train dataset: {len(train_df)} rows")
print(f"Test dataset: {len(test_df)} rows")

# Scale features
scaler = MinMaxScaler()
# Fit only on training data!
train_scaled = scaler.fit_transform(train_df[feature_cols])
test_scaled = scaler.transform(test_df[feature_cols])

# Store scaled data back into dataframes temporarily to group them by stock for LSTM
train_df_scaled = train_df[['Stock', 'Date', 'Target']].copy()
test_df_scaled = test_df[['Stock', 'Date', 'Target']].copy()

for i, col in enumerate(feature_cols):
    train_df_scaled[col] = train_scaled[:, i]
    test_df_scaled[col] = test_scaled[:, i]

# Save XGBoost data (which doesn't need sequences)
X_train_xgb = train_df_scaled[feature_cols].values
y_train_xgb = train_df_scaled['Target'].values
X_test_xgb = test_df_scaled[feature_cols].values
y_test_xgb = test_df_scaled['Target'].values

np.save('xgb_X_train.npy', X_train_xgb)
np.save('xgb_X_test.npy', X_test_xgb)
np.save('xgb_y_train.npy', y_train_xgb)
np.save('xgb_y_test.npy', y_test_xgb)


# Prepare LSTM sequences properly grouped by stock!
sequence_length = 60

def create_stock_sequences(df_scaled, feature_cols, seq_length):
    X_seq, y_seq = [], []
    # Group by stock to ensure we never cross time-series boundaries!
    for stock, group in df_scaled.groupby('Stock'):
        group = group.sort_values('Date').reset_index(drop=True)
        if len(group) <= seq_length: continue
        
        features = group[feature_cols].values
        targets = group['Target'].values
        
        for i in range(len(features) - seq_length):
            X_seq.append(features[i:i + seq_length])
            y_seq.append(targets[i + seq_length])
            
    return np.array(X_seq), np.array(y_seq)

print("\nFiltering strictly to 100 stocks for LSTM memory safety before parsing sequences...")
unique_stocks_lstm = unique_stocks[:100]
train_df_lstm_scaled = train_df_scaled[train_df_scaled['Stock'].isin(unique_stocks_lstm)].copy()
test_df_lstm_scaled = test_df_scaled[test_df_scaled['Stock'].isin(unique_stocks_lstm)].copy()

print("Generating clean, contiguous 60-day sequences for LSTM...")
X_train_lstm, y_train_lstm = create_stock_sequences(train_df_lstm_scaled, feature_cols, sequence_length)
X_test_lstm, y_test_lstm = create_stock_sequences(test_df_lstm_scaled, feature_cols, sequence_length)

np.save('lstm_X_train.npy', X_train_lstm)
np.save('lstm_X_test.npy', X_test_lstm)
np.save('lstm_y_train.npy', y_train_lstm)
np.save('lstm_y_test.npy', y_test_lstm)

print(f"LSTM Train sequences: {X_train_lstm.shape}")
print(f"LSTM Test sequences: {X_test_lstm.shape}")

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\nTarget Class Imbalance check:")
print("XGB Train Target distribution: UP = {:.2f}%, DOWN = {:.2f}%".format(
    np.sum(y_train_xgb == 1) / len(y_train_xgb) * 100, 
    np.sum(y_train_xgb == 0) / len(y_train_xgb) * 100))
print("LSTM Train Target distribution: UP = {:.2f}%, DOWN = {:.2f}%".format(
    np.sum(y_train_lstm == 1) / len(y_train_lstm) * 100, 
    np.sum(y_train_lstm == 0) / len(y_train_lstm) * 100))

print("\n Data preparation completed successfully.")

