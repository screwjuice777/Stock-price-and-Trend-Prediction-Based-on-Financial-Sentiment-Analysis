import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import class_weight

def fbeta_score(y_true, y_pred, beta=2):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    if (p + r) == 0:
        return 0.0
    return (1 + beta ** 2) * p * r / ((beta ** 2) * p + r)


print("LSTM CLASSIFICATION TRAINING")


# load data
X_train = np.load('lstm_X_train.npy')
X_test = np.load('lstm_X_test.npy')
y_train = np.load('lstm_y_train.npy')
y_test = np.load('lstm_y_test.npy')

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# bregularized LSTM model
print("\nbuilding LSTM model")
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2), input_shape=X_train.shape[1:]),
    BatchNormalization(),
    Bidirectional(LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dropout(0.3),   
    Dense(1, activation='sigmoid')  # sigmoid for binary classification
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# callbacks
es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-5)

# No severe class weight adjustment needed as classes are practically 50/50
# Force the model to learn raw patterns.

# train with validation
print("\nTraining LSTM with regularization (No class weights)")
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.18,
    callbacks=[es, lr_reduce],
    verbose=1
)

# predict
y_pred_proba = model.predict(X_test, verbose=0).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

# evaluate
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
f2 = fbeta_score(y_test, y_pred, beta=2)
cm = confusion_matrix(y_test, y_pred)


print("LSTM RESULT")

print(f"Accuracy:   {acc:.4f}")
print(f"Precision:  {prec:.4f}")
print(f"Recall:     {rec:.4f}")
print(f"F1-Score:   {f1:.4f}")
print(f"F2-Score:   {f2:.4f}")
print("\nconfusion Matrix:")
print(cm)

# save model
model.save('lstm_model.keras')
print("\nLSTM model saved  lstm_model.keras")
