import pandas as pd
import numpy as np

# load dataset
df = pd.read_csv('stock_with_sentiment_features.csv', parse_dates=['Date'])
# sort by stock and date

df = df.sort_values(['Stock', 'Date']).reset_index(drop=True)
# technical features
# simple moving averages of close price
df['sma_5'] = df.groupby('Stock')['Close'].transform(lambda x: x.rolling(window=5).mean())
df['sma_10'] = df.groupby('Stock')['Close'].transform(lambda x: x.rolling(window=10).mean())
df['sma_20'] = df.groupby('Stock')['Close'].transform(lambda x: x.rolling(window=20).mean())

# exponential moving average
df['ema_5'] = df.groupby('Stock')['Close'].transform(lambda x: x.ewm(span=5).mean())
df['ema_20'] = df.groupby('Stock')['Close'].transform(lambda x: x.ewm(span=20).mean())
# rolling volatility
df['volatility_5'] = df.groupby('Stock')['Close'].transform(lambda x: x.rolling(window=5).std())
df['volatility_20'] = df.groupby('Stock')['Close'].transform(lambda x: x.rolling(window=20).std())

# relative strength index
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['rsi_14'] = df.groupby('Stock')['Close'].transform(lambda x: calculate_rsi(x, 14))

# macd and signal line
df['ema_12'] = df.groupby('Stock')['Close'].transform(lambda x: x.ewm(span=12).mean())
df['ema_26'] = df.groupby('Stock')['Close'].transform(lambda x: x.ewm(span=26).mean())
df['macd'] = df['ema_12'] - df['ema_26']
df['signal_line'] = df.groupby('Stock')['macd'].transform(lambda x: x.ewm(span=9).mean())
df['macd_histogram'] = df['macd'] - df['signal_line']

# bollinger bands
df['bb_middle'] = df.groupby('Stock')['Close'].transform(lambda x: x.rolling(window=20).mean())
df['bb_std'] = df.groupby('Stock')['Close'].transform(lambda x: x.rolling(window=20).std())
df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)

# lagged features

# lagged daily returns
for lag in [1, 2, 3, 5]:
    df[f'return_lag_{lag}'] = df.groupby('Stock')['Daily_Return'].shift(lag)

# lagged close price
for lag in [1, 5, 10]:
    df[f'close_lag_{lag}'] = df.groupby('Stock')['Close'].shift(lag)

# lagged sentiment features
for lag in [1, 2, 3]:
    df[f'sentiment_compound_lag_{lag}'] = df.groupby('Stock')['sentiment_compound'].shift(lag)
    df[f'sentiment_positive_lag_{lag}'] = df.groupby('Stock')['sentiment_positive'].shift(lag)
    df[f'sentiment_negative_lag_{lag}'] = df.groupby('Stock')['sentiment_negative'].shift(lag)

# rolling window features
df['sentiment_compound_rolling_5'] = df.groupby('Stock')['sentiment_compound'].transform(lambda x: x.rolling(window=5).mean())
df['sentiment_compound_rolling_10'] = df.groupby('Stock')['sentiment_compound'].transform(lambda x: x.rolling(window=10).mean())

df['sentiment_compound_std_5'] = df.groupby('Stock')['sentiment_compound'].transform(lambda x: x.rolling(window=5).std())

# drop intermediate columns not needed further
df = df.drop(columns=['ema_12', 'ema_26', 'bb_middle', 'bb_std'])

# fill missing values using forward then backward fill
df = df.fillna(method='ffill').fillna(method='bfill')

print("feature engineering complete")
print(df.head(10))
print("shape:", df.shape)
print("columns:", list(df.columns))

# save  csv
df.to_csv('stock_with_all_features.csv', index=False)
print("feature-engineered dataset 'stock_with_all_features.csv'")
