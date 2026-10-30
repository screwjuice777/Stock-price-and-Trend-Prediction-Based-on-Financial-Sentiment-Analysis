import os
import subprocess
import sys

def run_script(script_name):
    print(f"\n{'='*60}")
    print(f"Executing: {script_name}")
    print(f"{'='*60}\n")
    try:
        subprocess.run([sys.executable, script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Pipeline 0 failed during '{script_name}'.")
        print(f"Error details: {e}")
        sys.exit(1)

def main():
    print("""
=========================================================
   PIPELINE 0: DATA PREPROCESSING & FEATURE ENGINEERING
=========================================================
     Pipeline order:

 Step 1. Data Cleaning
         - Load raw stock price data (stock.csv)
         - Remove duplicates, fill missing trading days
         - Forward-fill prices, compute daily returns
         - Load raw news data (indianfinancenews.csv)
         - Clean and normalize all headline text
         - Output: stock_cleaned.csv, news_cleaned.csv

 Step 2. FinBERT Sentiment Analysis
         - Load cleaned news (news_cleaned.csv)
         - Run FinBERT deep learning model on each headline
         - Extract compound score, positive, negative probabilities
         - Output: news_with_finbert_sentiment_and_logits.csv

 Step 3. Sentiment Aggregation & Stock Merge
         - Aggregate daily sentiment scores across all news
         - Merge sentiment scores with cleaned stock data by Date
         - Fill neutral sentiment (0) for days with no news
         - Output: stock_with_sentiment_features.csv

 Step 4. Technical Feature Engineering
         - Load merged dataset (stock_with_sentiment_features.csv)
         - Compute SMA (5, 10, 20), EMA (5, 20), RSI (14)
         - Compute MACD, Signal Line, MACD Histogram
         - Compute Bollinger Bands (Upper & Lower)
         - Compute Rolling Sentiment Averages (5, 10 days)
         - Compute Rolling Sentiment Volatility (std 5 days)
         - Compute Lagged Price & Sentiment Features
         - Output: stock_with_all_features.csv (FINAL MASTER FILE)


=========================================================
""")

    # Check if FinBERT output already exists to avoid re-running
    finbert_output = 'news_with_finbert_sentiment_and_logits.csv'
    finbert_already_done = os.path.exists(finbert_output)

    if finbert_already_done:
        print(f"FinBERT sentiment file already exists: '{finbert_output}'")
        print("Skipping Step 2 (FinBERT Sentiment Analysis) to save time.")
        print("To force re-run FinBERT, delete the file and re-run this pipeline.\n")

    pipeline_scripts = [
        "data_cleaning.py",
        "finbert_sentiment.py" if not finbert_already_done else None,
        "aggregate_sentiment_merge_stock.py",
        "feature_engineering_technical_sentiment.py"
    ]

    # Remove None entries (skipped steps)
    pipeline_scripts = [s for s in pipeline_scripts if s is not None]

    for script in pipeline_scripts:
        if os.path.exists(script):
            run_script(script)
        else:
            print(f"[ERROR] Required script '{script}' not found in current directory!")
            sys.exit(1)

    print("\n" + "="*60)
    print("PIPELINE 0 COMPLETED SUCCESSFULLY")
    print("The final master dataset is ready: stock_with_all_features.csv")
    print("\nYou can now run the training pipeline:")
    print(" -> python pipeline_1_train_and_evaluate.py")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
