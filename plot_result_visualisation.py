import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, auc
)

print("\n" + "="*60)
print("GENERATING FINAL PRESENTATION VISUALS")
print("="*60)

# Load isolated test datasets (XGBoost and LSTM have different sizes - by design)
X_test_xgb  = np.load('xgb_X_test.npy')
y_test_xgb  = np.load('xgb_y_test.npy')
X_test_lstm = np.load('lstm_X_test.npy')
y_test_lstm = np.load('lstm_y_test.npy')

# Load trained models
xgb_model = xgb.Booster()
xgb_model.load_model('xgb_model.json')
lstm_model = load_model('lstm_model.keras')

# Generate predictions
dtest_xgb         = xgb.DMatrix(X_test_xgb)
y_pred_xgb_proba  = xgb_model.predict(dtest_xgb)
y_pred_xgb        = (y_pred_xgb_proba > 0.5).astype(int)

y_pred_lstm_proba = lstm_model.predict(X_test_lstm, verbose=0).flatten()
y_pred_lstm       = (y_pred_lstm_proba > 0.5).astype(int)

print("Predictions loaded for both models.")

# ============================================================
# PLOT 1: Metrics Comparison Bar Chart
# ============================================================
print("\n1. Metrics Comparison Bar Chart...")
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
xgb_scores = [
    accuracy_score(y_test_xgb, y_pred_xgb) * 100,
    precision_score(y_test_xgb, y_pred_xgb, zero_division=0) * 100,
    recall_score(y_test_xgb, y_pred_xgb, zero_division=0) * 100,
    f1_score(y_test_xgb, y_pred_xgb, zero_division=0) * 100
]
lstm_scores = [
    accuracy_score(y_test_lstm, y_pred_lstm) * 100,
    precision_score(y_test_lstm, y_pred_lstm, zero_division=0) * 100,
    recall_score(y_test_lstm, y_pred_lstm, zero_division=0) * 100,
    f1_score(y_test_lstm, y_pred_lstm, zero_division=0) * 100
]

x = np.arange(len(metrics))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, xgb_scores, width, label='XGBoost (1.5M Rows)', color='royalblue')
rects2 = ax.bar(x + width/2, lstm_scores, width, label='LSTM (250K Rows)', color='mediumseagreen')
ax.set_ylabel('Score (%)', fontweight='bold')
ax.set_title('Model Performance Metrics Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontweight='bold')
ax.legend()
ax.set_ylim([0, 100])
ax.grid(axis='y', linestyle='--', alpha=0.6)
for rect in rects1 + rects2:
    h = rect.get_height()
    ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontweight='bold', fontsize=9)
plt.tight_layout()
plt.savefig('plots/plot_01_metrics_comparison.png', dpi=300)
plt.close()
print("   Saved: plot_01_metrics_comparison.png")

# ============================================================
# PLOT 2: Confusion Matrices Side-by-Side
# ============================================================
print("2. Confusion Matrices...")
cm_xgb  = confusion_matrix(y_test_xgb,  y_pred_xgb)
cm_lstm = confusion_matrix(y_test_lstm, y_pred_lstm)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['DOWN', 'UP'], yticklabels=['DOWN', 'UP'],
            annot_kws={'size': 13, 'weight': 'bold'})
axes[0].set_title('XGBoost Confusion Matrix', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Predicted Direction')
axes[0].set_ylabel('Actual Direction')

sns.heatmap(cm_lstm, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['DOWN', 'UP'], yticklabels=['DOWN', 'UP'],
            annot_kws={'size': 13, 'weight': 'bold'})
axes[1].set_title('LSTM Confusion Matrix', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Predicted Direction')
axes[1].set_ylabel('Actual Direction')

plt.tight_layout()
plt.savefig('plots/plot_02_confusion_matrices.png', dpi=300)
plt.close()
print("   Saved: plot_02_confusion_matrices.png")

# ============================================================
# PLOT 3: Directional Accuracy Bar Chart
# ============================================================
print("3. Directional Accuracy Bar Chart...")
acc_xgb  = accuracy_score(y_test_xgb,  y_pred_xgb)  * 100
acc_lstm = accuracy_score(y_test_lstm, y_pred_lstm) * 100

models_list = ['XGBoost\n(1.5M rows)', 'LSTM\n(250K rows)']
accs = [acc_xgb, acc_lstm]
colors_list = ['royalblue', 'mediumseagreen']

plt.figure(figsize=(8, 6))
bars = plt.bar(models_list, accs, color=colors_list, edgecolor='black', linewidth=1.5, alpha=0.85)
plt.axhline(y=50, color='gray', linestyle='--', alpha=0.6, label='Random Baseline (50%)')
plt.ylabel('Directional Accuracy (%)', fontweight='bold')
plt.title('UP/DOWN Directional Prediction Accuracy', fontsize=14, fontweight='bold')
plt.ylim([0, 100])
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
for bar, val in zip(bars, accs):
    plt.text(bar.get_x() + bar.get_width()/2, val + 1.5, f'{val:.2f}%',
             ha='center', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('plots/plot_03_directional_accuracy.png', dpi=300)
plt.close()
print("   Saved: plot_03_directional_accuracy.png")

# ============================================================
# PLOT 4: Predicted vs Actual (XGBoost probability - 300 samples)
# ============================================================
print("4. Predicted vs Actual Plot...")
sample_n = min(300, len(y_test_xgb))
indices  = np.arange(sample_n)

plt.figure(figsize=(14, 5))
plt.plot(indices, y_test_xgb[:sample_n],       label='Actual Label',       color='black',      linewidth=2, alpha=0.9)
plt.plot(indices, y_pred_xgb_proba[:sample_n],  label='XGBoost Predicted',  color='royalblue',  linewidth=1.5, alpha=0.75)
plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Decision Threshold (0.5)')
plt.fill_between(indices, y_test_xgb[:sample_n], y_pred_xgb_proba[:sample_n], alpha=0.15, color='royalblue')
plt.xlabel('Sample Index', fontweight='bold')
plt.ylabel('Predicted Probability / Actual Label', fontweight='bold')
plt.title('Predicted vs Actual - XGBoost (300 Sample Window)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/plot_04_predicted_vs_actual.png', dpi=300)
plt.close()
print("   Saved: plot_04_predicted_vs_actual.png")

# ============================================================
# PLOT 5: ROC Curves (both models)
# ============================================================
print("5. ROC Curves...")
fpr_xgb,  tpr_xgb,  _ = roc_curve(y_test_xgb,  y_pred_xgb_proba)
fpr_lstm, tpr_lstm, _ = roc_curve(y_test_lstm, y_pred_lstm_proba)
auc_xgb  = auc(fpr_xgb,  tpr_xgb)
auc_lstm = auc(fpr_lstm, tpr_lstm)

plt.figure(figsize=(8, 6))
plt.plot(fpr_xgb,  tpr_xgb,  label=f'XGBoost  (AUC = {auc_xgb:.4f})',  color='royalblue', linewidth=2.5)
plt.plot(fpr_lstm, tpr_lstm, label=f'LSTM     (AUC = {auc_lstm:.4f})', color='mediumseagreen', linewidth=2.5)
plt.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random Classifier')
plt.fill_between(fpr_xgb,  tpr_xgb,  alpha=0.08, color='royalblue')
plt.fill_between(fpr_lstm, tpr_lstm, alpha=0.08, color='mediumseagreen')
plt.xlabel('False Positive Rate', fontweight='bold')
plt.ylabel('True Positive Rate', fontweight='bold')
plt.title('ROC Curves - XGBoost vs LSTM', fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/plot_05_roc_curves.png', dpi=300)
plt.close()
print("   Saved: plot_05_roc_curves.png")

# ============================================================
# PLOT 6: XGBoost Feature Importance (Top 15)
# ============================================================
print("6. Feature Importance Chart...")
importance = xgb_model.get_score(importance_type='weight')
importance_df = (pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])
                 .sort_values('Importance', ascending=False).head(15))

plt.figure(figsize=(10, 7))
plt.barh(importance_df['Feature'], importance_df['Importance'],
         color='coral', edgecolor='black', alpha=0.85)
plt.xlabel('F-Score (Weight)', fontweight='bold')
plt.title('XGBoost Top 15 Feature Importance', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('plots/plot_06_feature_importance.png', dpi=300)
plt.close()
print("   Saved: plot_06_feature_importance.png")

# ============================================================
# PLOT 7: Feature Correlation Heatmap (all 20 inputs)
# ============================================================
print("7. Feature Correlation Heatmap...")
feature_names = [
    'Close', 'sentiment_compound', 'sentiment_positive', 'sentiment_negative',
    'sma_5', 'sma_10', 'sma_20', 'ema_5', 'ema_20',
    'volatility_5', 'volatility_20', 'rsi_14', 'macd', 'signal_line',
    'macd_histogram', 'bb_upper', 'bb_lower',
    'sentiment_rolling_5', 'sentiment_rolling_10', 'sentiment_std_5'
]
corr_df     = pd.DataFrame(X_test_xgb[:5000], columns=feature_names)
corr_matrix = corr_df.corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, mask=mask, cmap='RdYlGn', center=0,
            annot=False, linewidths=0.5,
            cbar_kws={'label': 'Pearson Correlation Coefficient'})
plt.title('Feature Correlation Heatmap (All 20 Input Features)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/plot_07_feature_correlation_heatmap.png', dpi=300)
plt.close()
print("   Saved: plot_07_feature_correlation_heatmap.png")

print("\n" + "="*60)
print("Visualisation plots saved:")
print(" -> plot_01_metrics_comparison.png")
print(" -> plot_02_confusion_matrices.png")
print(" -> plot_03_directional_accuracy.png")
print(" -> plot_04_predicted_vs_actual.png")
print(" -> plot_05_roc_curves.png")
print(" -> plot_06_feature_importance.png")
print(" -> plot_07_feature_correlation_heatmap.png")
print("="*60 + "\n")
