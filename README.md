# Stock-price-and-Trend-Prediction-Based-on-Financial-Sentiment-Analysis

An end-to-end Machine Learning and Deep Learning system for predicting Indian stock price movement and directional trends by fusing quantitative technical market indicators with NLP-driven financial sentiment analysis from news headlines.

---

## 📌 Project Overview

Stock market movements are driven by both quantitative technical signals (momentum, moving averages, volatility) and qualitative market sentiment (breaking financial news and market disclosures). This project implements a **hybrid prediction architecture**:

1. **FinBERT NLP Sentiment Extraction**: Processes financial news headlines to extract positive, negative, and neutral sentiment probabilities and compound polarity scores.
2. **Technical Feature Engineering**: Computes moving averages (SMA, EMA), RSI, MACD, Bollinger Bands, rolling sentiment statistics, and lagged market returns.
3. **Machine Learning & Deep Learning Ensemble**:
   - **XGBoost Classifier**: Optimized gradient boosting trees for high-dimensional feature importance and non-linear classification.
   - **LSTM Sequence Network**: Recurrent neural network capturing temporal dependencies across multi-day lookback windows.
   - **Hybrid Voting & Evaluation**: Evaluates individual and ensemble performance to select the highest directional accuracy model.

---

## 🏗️ Architecture & Pipelines

The project is organized into modular, automated pipelines:

```
├── pipeline_0_data_preprocessing.py    # Pipeline 0: Raw data cleaning, FinBERT sentiment, technical features
├── pipeline_1_train_and_evaluate.py     # Pipeline 1: Train/test preparation, XGBoost & LSTM training, model evaluation
├── pipeline_2_batch_inference.py        # Pipeline 2: Batch inference on unseen data with pre-trained models
│
├── data_cleaning.py                     # Cleans raw stock and financial news CSVs
├── finbert_sentiment.py                 # HuggingFace FinBERT sentiment scoring
├── aggregate_sentiment_merge_stock.py   # Aggregates daily sentiment and merges with stock records
├── feature_engineering_technical_sentiment.py # Computes SMA, EMA, RSI, MACD, Bollinger Bands
├── prepare_train_test_data.py           # Prepares scaled sequence windows for LSTM and flat feature sets for XGBoost
├── train_evaluate_xgboost.py            # Trains and evaluates XGBoost
├── train_evaluate_lstm.py               # Trains and evaluates Bidirectional LSTM
├── hybrid_prediction_final.py           # Evaluates hybrid ensemble and generates comparative metrics
├── plot_result_visualisation.py         # Generates presentation-grade performance charts
│
├── lstm_model.keras                     # Pre-trained LSTM sequence model
├── xgb_model.json                       # Pre-trained XGBoost model
├── scaler.pkl                           # Fitted MinMaxScaler for technical & sentiment features
│
├── indianfinancenews.csv                # Real Indian financial news headlines dataset
├── news_with_finbert_sentiment_and_logits.csv # Pre-computed FinBERT sentiment scores & logits
├── inference_sample.csv                 # Ready-to-use sample dataset for batch prediction & single stock test
├── batch_prediction_results.csv         # Sample inference predictions output
│
├── plots/                               # Model performance visualizations & dashboards
│   ├── plot_01_metrics_comparison.png
│   ├── plot_02_confusion_matrices.png
│   ├── plot_03_directional_accuracy.png
│   ├── plot_04_predicted_vs_actual.png
│   ├── plot_05_roc_curves.png
│   ├── plot_06_feature_importance.png
│   ├── plot_07_feature_correlation_heatmap.png
│   └── prediction_dashboard_TCS.png
│
├── requirements.txt                     # Python dependencies
└── .gitignore                           # Excludes large raw datasets (>100MB) from git
```

---

## 🚀 Quick Start & Usage (Try Out Right Away!)

Anyone cloning this repository can immediately test the models without any setup or training required.

### 1. Installation

```bash
git clone https://github.com/screwjuice777/Stock-price-and-Trend-Prediction-Based-on-Financial-Sentiment-Analysis.git
cd Stock-price-and-Trend-Prediction-Based-on-Financial-Sentiment-Analysis
pip install -r requirements.txt
```

### 2. Run Predictions & Dashboards

Launch the interactive prediction pipeline:

```bash
python pipeline_2_batch_inference.py
```

You have two ready-to-test modes:

- **Option 1: Single Stock Forecast (Dashboard Mode)**
  - Enter an included stock symbol (e.g. `GPIL`, `GPPL`, `GPTINFRA`, `GRANULES`).
  - Generates the 3-day directional prediction (UP / DOWN), confidence percentage, hybrid ensemble voting logic, and saves a dark-mode price & sentiment visual chart in `plots/`!
- **Option 2: Multi-Row Batch Evaluation**
  - Simply press **Enter** to evaluate against `inference_sample.csv`.
  - Computes predictions across thousands of records and saves results to `batch_prediction_results.csv`.

---

## 🔄 End-to-End Execution Flow

### Pipeline 0: Preprocessing & Feature Engineering
```bash
python pipeline_0_data_preprocessing.py
```
- Cleans raw stock prices and financial news headlines.
- Runs FinBERT deep learning inference to score sentiment logits.
- Aggregates daily sentiment scores and merges with price data by Date.
- Engineers technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands) and sentiment volatility features.

### Pipeline 1: Training & Model Evaluation
```bash
python pipeline_1_train_and_evaluate.py
```
- Splits data chronologically into train and test sets.
- Normalizes features using `MinMaxScaler` (saved as `scaler.pkl`).
- Trains XGBoost classifier and exports `xgb_model.json`.
- Trains Bidirectional LSTM network with Early Stopping and exports `lstm_model.keras`.
- Evaluates confusion matrices, ROC-AUC, Precision, Recall, F1, and directional accuracy.
- Saves evaluation plots to `plots/`.

---

## 📊 Evaluation Visualizations

All evaluation plots and dashboards are generated in the `plots/` directory:

- **Metrics Comparison**: Accuracy, Precision, Recall, and F1 across baseline and tuned models.
- **Confusion Matrices**: Detailed true positive / false positive breakdown.
- **Directional Accuracy**: Up/Down movement classification accuracy.
- **Feature Importance**: Most influential technical and sentiment indicators.
- **Prediction Dashboard**: Time-series visualization of actual vs predicted trends.

---

## 📦 Large Datasets & Storage Note

The complete raw and engineered datasets (`stock.csv`, `stock_with_all_features.csv` ~2.8GB, `lstm_X_train.npy` ~1.8GB) exceed GitHub's 100MB single-file limit and are excluded via `.gitignore`. 

They can be regenerated at any time by placing the raw data files (`stock.csv` and `indianfinancenews.csv`) in the root directory and running `python pipeline_0_data_preprocessing.py`.
