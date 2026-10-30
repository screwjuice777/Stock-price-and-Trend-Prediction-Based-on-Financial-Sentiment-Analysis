import pandas as pd
import re

stock_df = pd.read_csv('stock.csv', parse_dates=['Date'])
stock_df.drop_duplicates(inplace=True)

cleaned_stocks = []
for stock_name, group in stock_df.groupby('Stock'):
    group = group.set_index('Date').sort_index()
    bdays = pd.date_range(start=group.index.min(), end=group.index.max(), freq='B')
    group = group.reindex(bdays)
    group['Stock'] = stock_name

    price_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change Pct']
    group[price_cols] = group[price_cols].ffill()
    group['Daily_Return'] = group['Close'].pct_change()
    group = group.dropna(subset=['Close', 'Daily_Return'])
    cleaned_stocks.append(group)

stock_cleaned = pd.concat(cleaned_stocks)

print("Stock price data cleaned.")
print(stock_cleaned.head())

news_df = pd.read_csv('indianfinancenews.csv', parse_dates=['Date'])
news_df.drop_duplicates(inplace=True)

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

headline_col = 'Title'
if headline_col in news_df.columns:
    news_df['cleaned_headline'] = news_df[headline_col].apply(clean_text)
    news_df = news_df[news_df['cleaned_headline'] != ""]
else:
    print(f"Column '{headline_col}' not found. Use one from: {list(news_df.columns)}")

if 'Description' in news_df.columns:
    news_df['cleaned_description'] = news_df['Description'].apply(clean_text)


print("News data cleaned.")
print(news_df.head())

stock_cleaned = stock_cleaned.reset_index()
stock_cleaned.rename(columns={'index': 'Date'}, inplace=True)
stock_cleaned.to_csv('stock_cleaned.csv', index=False)
news_df.to_csv('news_cleaned.csv', index=False)
