import os
import sys
# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def get_feature_cols(all_cols):
    drop_cols = ['Date', 'Stock', 'Open', 'High', 'Low', 'Volume', 'Change Pct', 'Daily_Return', 'forward_return_3d', 'Target']
    noisy_lags = [f'return_lag_{i}' for i in [1, 2, 3, 5]] + \
                 [f'close_lag_{i}' for i in [1, 5, 10]] + \
                 [f'sentiment_compound_lag_{i}' for i in [1, 2, 3]] + \
                 [f'sentiment_positive_lag_{i}' for i in [1, 2, 3]] + \
                 [f'sentiment_negative_lag_{i}' for i in [1, 2, 3]]
    drop_cols.extend(noisy_lags)
    return [col for col in all_cols if col not in drop_cols]

def main():
    print("="*60)
    print("      PIPELINE 2: LIVE PREDICTION & BATCH INFERENCE")
    print("="*60)

    # 1. Load models
    try:
        xgb_model = xgb.Booster()
        xgb_model.load_model('xgb_model.json')
        lstm_model = load_model('lstm_model.keras')
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except Exception as e:
        print("[ERROR] Required model files not found. Run pipeline_1_train_and_evaluate.py first!")
        sys.exit(1)

    # 2. Get Data
    print("\n[Options]")
    print("1. Single Stock Forecast (Dashboard Mode)")
    print("2. Multi-row Batch Testing (Evaluate Accuracy on new CSV Data)")
    choice = input("Select option (1/2): ").strip()
    
    if choice == '1':
        if os.path.exists('stock_with_all_features.csv'):
            feature_file = 'stock_with_all_features.csv'
            print("[Info] Using master dataset: stock_with_all_features.csv")
            ticker_prompt = "Enter Stock Symbol (e.g., RELIANCE, TCS, GPIL): "
        elif os.path.exists('inference_sample.csv'):
            feature_file = 'inference_sample.csv'
            print("[Info] Using included sample dataset: inference_sample.csv")
            print("       Available sample stocks: GPIL, GPPL, GPTINFRA, GRANULES")
            print("       (Place full 'stock_with_all_features.csv' in folder for all NSE/BSE stocks).")
            ticker_prompt = "Enter Stock Symbol (e.g., GPIL, GPPL, GPTINFRA, GRANULES): "
        else:
            print("Error: Neither 'stock_with_all_features.csv' nor 'inference_sample.csv' found.")
            sys.exit(1)
            
        ticker = input(ticker_prompt).strip().replace('.NS', '')
        print(f"Loading data for {ticker} from {feature_file}...")
        
        # Slicing CSV to find the ticker
        chunks = []
        for chunk in pd.read_csv(feature_file, chunksize=100000, parse_dates=['Date']):
            stock_chunk = chunk[chunk['Stock'] == ticker]
            if not stock_chunk.empty:
                chunks.append(stock_chunk)
                
        if not chunks:
            print(f"Error: No data found for {ticker}.")
            sys.exit(1)
            
        df = pd.concat(chunks)
        df = df.sort_values('Date').reset_index(drop=True)
        
        if len(df) < 60:
            print(f"Error: Need at least 60 historical days for LSTM sequence. Got {len(df)}.")
            sys.exit(1)

        print(f"\nAnalyzing latest 60 trading days up to {df['Date'].iloc[-1].date()} for {ticker}...")

        # Select latest 60 rows
        df_recent = df.tail(60).copy()
        feature_cols = get_feature_cols(df_recent.columns)
        
        # Ensure columns match scaler exactly
        missing_features = [col for col in feature_cols if col not in df.columns]
        if missing_features:
            print(f"Error: Provided data is missing features required by model: {missing_features}")
            sys.exit(1)
            
        # 3. Scale Features
        X_raw = df_recent[feature_cols].values
        X_scaled = scaler.transform(X_raw)
        
        # 4. Predict
        # LSTM sequence of shape (1, 60, num_features)
        X_lstm = np.array([X_scaled])
        lstm_prob = lstm_model.predict(X_lstm, verbose=0)[0][0]
        
        # XGBoost flat features of last day (1, num_features)
        X_xgb = X_scaled[-1].reshape(1, -1)
        dtest = xgb.DMatrix(X_xgb)
        xgb_prob = xgb_model.predict(dtest)[0]
        
        # Hybrid Strategy
        confidence_gap = 0.15
        if abs(lstm_prob - xgb_prob) < confidence_gap:
            final_prob = lstm_prob
            model_used = 'LSTM Trusted'
        else:
            final_prob = xgb_prob
            model_used = 'XGBoost Fallback'
            
        prediction = "UP" if final_prob > 0.5 else "DOWN"
        confidence = final_prob if final_prob > 0.5 else (1 - final_prob)

        print("\n" + "="*45)
        print(f"   FINAL PREDICTION FOR NEXT 3 DAYS   ")
        print("="*45)
        print(f"Stock:        {ticker}")
        print(f"Target Move:  {prediction}")
        print(f"Confidence:   {confidence*100:.2f}% Sure")
        print(f"Logic Used:   {model_used}")
        print(f"(LSTM Score: {lstm_prob*100:.1f}% | XGB Score: {xgb_prob*100:.1f}%)")
        print("="*45)

        # 5. Generate Detailed Visual Plot
        try:
            plt.style.use('dark_background')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [2, 1]})
            
            # Top Plot: Price History
            last_60 = df.tail(60)
            ax1.plot(last_60['Date'], last_60['Close'], marker='o', color='lightskyblue', label='Close Price')
            ax1.set_title(f"{ticker} Recent Price Trend vs Hybrid Prediction", fontsize=14, fontweight='bold')
            ax1.set_ylabel("Price")
            ax1.grid(True, alpha=0.3)
            
            # Superimpose Prediction Gauge
            color = 'lightgreen' if prediction == 'UP' else 'salmon'
            ax1.text(df_recent['Date'].iloc[-1], df_recent['Close'].iloc[-1], 
                     f" PREDICTION: {prediction}\n CONF: {confidence*100:.1f}%\n",
                     fontsize=12, fontweight='bold', color=color,
                     bbox={'facecolor': 'black', 'alpha': 0.7, 'edgecolor': color})
            
            if 'sma_20' in df.columns:
                ax1.plot(last_60['Date'], last_60['sma_20'], color='orange', linestyle='--', label='SMA 20')
                
            ax1.legend(loc='upper left')
            
            # Bottom Plot: Sentiment or MACD
            if 'sentiment_compound' in df.columns:
                colors = ['g' if x>0 else 'r' for x in last_60['sentiment_compound']]
                ax2.bar(last_60['Date'], last_60['sentiment_compound'], color=colors, alpha=0.6, label='FinBERT Sentiment')
                ax2.set_title("FinBERT Daily News Sentiment", fontsize=10)
            elif 'macd' in df.columns:
                colors = ['g' if x>0 else 'r' for x in last_60['macd_histogram']]
                ax2.bar(last_60['Date'], last_60['macd_histogram'], color=colors, alpha=0.6, label='MACD Hist')
                ax2.plot(last_60['Date'], last_60['macd'], color='blue', label='MACD')
                ax2.plot(last_60['Date'], last_60['signal_line'], color='orange', label='Signal')
                ax2.set_title("MACD Indicator", fontsize=10)
                
            ax2.legend()
            plt.tight_layout()
            
            plot_path = f"plots/prediction_dashboard_{ticker}.png"
            plt.savefig(plot_path)
            print(f"\n Dashboard successfully saved to -> {plot_path}")
            plt.show(block=False)
        except Exception as e:
            print(f"\n[Warning] Could not generate plots: {e}")

    elif choice == '2':
        path = input("Enter path to CSV for Batch Evaluation [Press Enter for 'inference_sample.csv']: ").strip()
        if not path:
            path = 'inference_sample.csv'
        if not os.path.exists(path):
            print(f"File '{path}' not found!")
            sys.exit(1)
            
        print(f"Loading {path}...")
        df = pd.read_csv(path, parse_dates=['Date'])
        
        has_target = 'Target' in df.columns
        if not has_target:
            print("Notice: 'Target' column not found. Predictions will be output without accuracy comparison.")
        
        feature_cols = get_feature_cols(df.columns)
        
        missing_features = [col for col in feature_cols if col not in df.columns]
        if missing_features:
            print(f"Error: Missing features required by model: {missing_features}")
            sys.exit(1)

        if 'Stock' not in df.columns:
            df['Stock'] = 'Unknown'

        print("\nCollecting all sequences...")

        all_lstm_seqs  = []
        all_xgb_feats  = []
        all_actuals    = []
        all_meta       = []
        stocks_processed = 0

        for stock, group in df.groupby('Stock'):
            group = group.sort_values('Date').reset_index(drop=True)
            if len(group) <= 60:
                print(f"  -> Skipping {stock} (need > 60 rows, got {len(group)})")
                continue

            stocks_processed += 1
            X_raw    = group[feature_cols].values
            X_scaled = scaler.transform(X_raw)

            for i in range(60, len(group)):
                all_lstm_seqs.append(X_scaled[i-60:i])
                all_xgb_feats.append(X_scaled[i])
                all_meta.append((stock, group['Date'].iloc[i]))
                if has_target:
                    all_actuals.append(int(group['Target'].iloc[i]))

        if len(all_lstm_seqs) == 0:
            print("\nError: No valid sequences generated. Each stock needs more than 60 rows.")
            sys.exit(1)

        print(f"Collected {len(all_lstm_seqs):,} sequences across {stocks_processed} stocks.")

        # LSTM: single batch predict (was the bottleneck - now runs once for all rows)
        print("Running LSTM batch prediction...")
        X_lstm_batch = np.array(all_lstm_seqs)
        lstm_proba   = lstm_model.predict(X_lstm_batch, verbose=1, batch_size=2048).flatten()

        # XGBoost: row-by-row (fast, kept as original)
        print("Running XGBoost predictions (row-by-row)...")
        xgb_proba = np.zeros(len(all_xgb_feats))
        for idx, feat in enumerate(all_xgb_feats):
            dtest = xgb.DMatrix(feat.reshape(1, -1))
            xgb_proba[idx] = xgb_model.predict(dtest)[0]

        # Hybrid ensemble
        confidence_gap = 0.15
        use_lstm    = np.abs(lstm_proba - xgb_proba) < confidence_gap
        final_proba = np.where(use_lstm, lstm_proba, xgb_proba)
        preds_binary = (final_proba > 0.5).astype(int)

        print(f"\nSuccessfully generated {len(preds_binary):,} predictions.")

        results_df = pd.DataFrame({
            'Stock':            [m[0] for m in all_meta],
            'Date':             [m[1] for m in all_meta],
            'XGB_Proba':        np.round(xgb_proba, 4),
            'LSTM_Proba':       np.round(lstm_proba, 4),
            'Hybrid_Proba':     np.round(final_proba, 4),
            'Predicted_Target': preds_binary
        })
        if has_target:
            results_df['Actual_Target'] = all_actuals

        save_path = "batch_prediction_results.csv"
        results_df.to_csv(save_path, index=False)
        print(f"All predictions saved to -> {save_path}")

        if has_target:
            from sklearn.metrics import accuracy_score, confusion_matrix
            
            xgb_preds    = (xgb_proba > 0.5).astype(int)
            lstm_preds   = (lstm_proba > 0.5).astype(int)
            
            acc_xgb    = accuracy_score(all_actuals, xgb_preds)
            acc_lstm   = accuracy_score(all_actuals, lstm_preds)
            acc_hybrid = accuracy_score(all_actuals, preds_binary)
            cm         = confusion_matrix(all_actuals, preds_binary)

            print("\n" + "="*45)
            print("   BATCH EVALUATION ACCURACY REPORT")
            print("="*45)
            print(f"Total Rows Evaluated : {len(preds_binary):,}")
            print(f"Stocks Processed     : {stocks_processed}")
            print(f"LSTM Trusted Rows    : {int(use_lstm.sum()):,} ({use_lstm.mean()*100:.1f}%)")
            print(f"XGB Fallback Rows    : {int((~use_lstm).sum()):,} ({(~use_lstm).mean()*100:.1f}%)")
            print(f"\nModel Accuracy Breakdown:")
            print(f"  XGBoost (standalone) : {acc_xgb*100:.2f}%")
            print(f"  LSTM    (standalone) : {acc_lstm*100:.2f}%")
            print(f"  Hybrid  (XGB + LSTM) : {acc_hybrid*100:.2f}%")
            print(f"\nHybrid Confusion Matrix (rows=Actual, cols=Predicted):")
            print(f"             Pred DOWN  Pred UP")
            print(f"Actual DOWN  {cm[0,0]:>9,}  {cm[0,1]:>7,}")
            print(f"Actual UP    {cm[1,0]:>9,}  {cm[1,1]:>7,}")
            print("="*45)
            
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
