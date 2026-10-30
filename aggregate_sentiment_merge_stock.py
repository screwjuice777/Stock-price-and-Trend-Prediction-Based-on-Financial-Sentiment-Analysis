import pandas as pd

# load data
stock_df = pd.read_csv('stock_cleaned.csv', parse_dates=['Date'])
news_df = pd.read_csv('news_with_finbert_sentiment_and_logits.csv', parse_dates=['Date'])
# by Date
sentiment_agg = news_df.groupby('Date').agg({
    'finbert_compound': 'mean',
    'finbert_positive': 'mean',
    'finbert_negative': 'mean'
}).reset_index()

sentiment_agg.rename(columns={
    'finbert_compound': 'sentiment_compound',
    'finbert_positive': 'sentiment_positive',
    'finbert_negative': 'sentiment_negative'
}, inplace=True)

# merge sentiment w stock data
merged_df = pd.merge(stock_df, sentiment_agg, on='Date', how='left')

# handle missing senti values
merged_df['sentiment_compound'] = merged_df['sentiment_compound'].fillna(0)
merged_df['sentiment_positive'] = merged_df['sentiment_positive'].fillna(0.33)
merged_df['sentiment_negative'] = merged_df['sentiment_negative'].fillna(0.33)

# Save to CSV
merged_df.to_csv('stock_with_sentiment_features.csv', index=False)

# output
print("Merged stock data with sentiment features:")
print(merged_df.head(5))                       
print("\nShape:", merged_df.shape)            
print("Columns:", list(merged_df.columns))    

print("\n aggregation and merge complete.")
print("merged dataset 'stock_with_sentiment_features.csv'.")
