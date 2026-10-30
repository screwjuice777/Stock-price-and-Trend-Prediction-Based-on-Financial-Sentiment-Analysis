import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


print("XGBOOST CLASSIFICATION TRAINING")


# Load data
X_train = np.load('xgb_X_train.npy')
X_test = np.load('xgb_X_test.npy')
y_train = np.load('xgb_y_train.npy')
y_test = np.load('xgb_y_test.npy')

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

#  max 1500 trees with early stopping 
scale_weight = float(np.sum(y_train == 0)) / np.sum(y_train == 1)

best_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.05,         # Increased learning rate for the larger dataset
    'max_depth': 7,      # Deeper trees to capture complex patterns across 200 stocks
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': scale_weight,
    'lambda': 2,         # Regularization 
    'alpha': 0.5,
    'seed': 42
}

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

print("\nTraining XGBoost with 1500 trees...")
evals = [(dtrain, 'train'), (dtest, 'test')]
xgb_model = xgb.train(
    best_params, dtrain,
    num_boost_round=1500,   
    evals=evals,
    early_stopping_rounds=75,
    verbose_eval=50
)

# Predict using the native booster
y_pred_proba = xgb_model.predict(dtest)
y_pred = (y_pred_proba > 0.5).astype(int)

# Evaluate - F2-score and full metrics
def fbeta_score(y_true, y_pred, beta=2):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    if (p + r) == 0:
        return 0.0
    return (1 + beta ** 2) * p * r / ((beta ** 2) * p + r)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
f2 = fbeta_score(y_test, y_pred, beta=2)
cm = confusion_matrix(y_test, y_pred)


print("XGBOOST RESULTS")

print(f"Accuracy:      {acc:.4f}")
print(f"Precision:     {prec:.4f}")
print(f"Recall:        {rec:.4f}")
print(f"F1-Score:      {f1:.4f}")
print(f"F2-Score:      {f2:.4f}")
print("\nconfusion matrix:")
print(cm)

# Save model
xgb_model.save_model('xgb_model.json')
print("\n XGBoost model saved as xgb_model.json")
