import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import xgboost as xgb
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


print("HYBRID MODEL EVALUATION AND COMPARISON")


# Load All Test Data 
X_test_xgb = np.load('xgb_X_test.npy')
y_test_xgb = np.load('xgb_y_test.npy')
X_test_lstm = np.load('lstm_X_test.npy')
y_test_lstm = np.load('lstm_y_test.npy')

# Load Models
xgb_model = xgb.Booster()
xgb_model.load_model('xgb_model.json')
lstm_model = load_model('lstm_model.keras')

# Get predictions
dtest_xgb = xgb.DMatrix(X_test_xgb)
y_pred_xgb_proba = xgb_model.predict(dtest_xgb)
y_pred_xgb = (y_pred_xgb_proba > 0.5).astype(int)

y_pred_lstm_proba = lstm_model.predict(X_test_lstm, verbose=0).flatten()
y_pred_lstm = (y_pred_lstm_proba > 0.5).astype(int)

# Standalone XGBoost report 
print("\nXGBoost True samples:", len(y_test_xgb))
acc = accuracy_score(y_test_xgb, y_pred_xgb)
prec = precision_score(y_test_xgb, y_pred_xgb, zero_division=0)
rec = recall_score(y_test_xgb, y_pred_xgb, zero_division=0)
f1 = f1_score(y_test_xgb, y_pred_xgb, zero_division=0)
cm = confusion_matrix(y_test_xgb, y_pred_xgb)
print("\nXGBoost (full test set):")
print(f"  Accuracy:  {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"Confusion matrix:\n{cm}")

# Standalone LSTM report (on its valid test set)
print("\nLSTM True samples:", len(y_test_lstm))
acc = accuracy_score(y_test_lstm, y_pred_lstm)
prec = precision_score(y_test_lstm, y_pred_lstm, zero_division=0)
rec = recall_score(y_test_lstm, y_pred_lstm, zero_division=0)
f1 = f1_score(y_test_lstm, y_pred_lstm, zero_division=0)
cm = confusion_matrix(y_test_lstm, y_pred_lstm)
print("\nLSTM (valid test set):")
print(f"  Accuracy:  {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"Confusion matrix:\n{cm}")

# 6. Hybrid Fallback Notice
print("\n" + "="*60)
print("NOTE: Because XGBoost was trained on an elite 1.5M rows,")
print("and LSTM was trained on 250k rows to protect your RAM,")
print("their baseline test sets are now entirely different sizes.")
print("\nTo test the Hybrid Fallback model on aligned new data layers,")
print("please use Pipeline 2 Option 2!")
print("="*60)
