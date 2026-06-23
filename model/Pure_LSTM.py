# ==========================================
# PURE LSTM FIXED — WITH VALIDATION THRESHOLD
# ==========================================

import pandas as pd
import numpy as np
import warnings

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(
    "data/04_Master_Dataset.csv",
    parse_dates=["Date"],
    index_col="Date"
)

technical_features = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_14", "EMA_14",
    "Ichimoku_Tenkan", "Ichimoku_Kijun",
    "Ichimoku_Senkou_A", "Ichimoku_Senkou_B",
    "ADX_14",
    "RSI_14",
    "MACD", "MACD_Signal", "MACD_Hist",
    "Williams_R",
    "K", "D", "J",
    "SQZ_Momentum",
    "BB_Middle", "BB_Upper", "BB_Lower", "BB_Width",
    "ATR_14"
]

# ==========================================
# TARGET
# ==========================================

df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
df["Target"] = (df["Return"].shift(-1) > 0).astype(int)

df.dropna(inplace=True)

X = df[technical_features]
y = df["Target"]

# ==========================================
# TRAIN / VAL / TEST SPLIT BEFORE SCALING
# ==========================================

n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train_raw = X.iloc[:train_end]
X_val_raw = X.iloc[train_end:val_end]
X_test_raw = X.iloc[val_end:]

y_train_raw = y.iloc[:train_end]
y_val_raw = y.iloc[train_end:val_end]
y_test_raw = y.iloc[val_end:]

# ==========================================
# SCALE — FIT ONLY ON TRAIN
# ==========================================

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train_raw)
X_val_scaled = scaler.transform(X_val_raw)
X_test_scaled = scaler.transform(X_test_raw)

# ==========================================
# CREATE SEQUENCES
# ==========================================

LOOKBACK = 14

def create_sequences(X_scaled, y_series, lookback=14):
    X_seq, y_seq = [], []

    for i in range(lookback, len(X_scaled)):
        X_seq.append(X_scaled[i - lookback:i])
        y_seq.append(y_series.iloc[i])

    return np.array(X_seq), np.array(y_seq)


X_train, y_train = create_sequences(X_train_scaled, y_train_raw, LOOKBACK)
X_val, y_val = create_sequences(X_val_scaled, y_val_raw, LOOKBACK)
X_test, y_test = create_sequences(X_test_scaled, y_test_raw, LOOKBACK)

print("Train:", X_train.shape, y_train.shape)
print("Val  :", X_val.shape, y_val.shape)
print("Test :", X_test.shape, y_test.shape)

print("\nTrain class distribution:")
print(pd.Series(y_train).value_counts())

# ==========================================
# CLASS WEIGHT
# ==========================================

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    0: weights[0],
    1: weights[1]
}

print("Class weights:", class_weights)

# ==========================================
# BUILD LSTM MODEL
# ==========================================

model = Sequential()

model.add(
    LSTM(
        units=128,
        activation="tanh",
        return_sequences=True,
        input_shape=(X_train.shape[1], X_train.shape[2])
    )
)
model.add(Dropout(0.3))

model.add(
    LSTM(
        units=64,
        activation="tanh",
        return_sequences=True
    )
)
model.add(Dropout(0.3))

model.add(
    LSTM(
        units=32,
        activation="tanh"
    )
)
model.add(Dropout(0.3))

model.add(Dense(1, activation="sigmoid"))

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================================
# TRAIN
# ==========================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=25,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=200,
    batch_size=32,
    validation_data=(X_val, y_val),
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

# ==========================================
# VALIDATION THRESHOLD OPTIMIZATION
# ==========================================

val_prob = model.predict(X_val).flatten()

print("\nValidation probability summary:")
print(pd.Series(val_prob).describe())

best_threshold = 0.5
best_f1 = 0

for threshold in np.arange(0.30, 0.71, 0.01):
    val_pred = (val_prob >= threshold).astype(int)
    f1 = f1_score(y_val, val_pred, zero_division=0)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print("\nBest threshold:", best_threshold)
print("Best validation F1:", best_f1)

# ==========================================
# TEST PREDICTION
# ==========================================

test_prob = model.predict(X_test).flatten()
test_pred = (test_prob >= best_threshold).astype(int)

print("\nTest probability summary:")
print(pd.Series(test_prob).describe())

print("\nPrediction distribution:")
print(np.unique(test_pred, return_counts=True))

# ==========================================
# EVALUATION
# ==========================================

print("\n=================================")
print("PURE LSTM RESULTS")
print("=================================")

print("Accuracy :", accuracy_score(y_test, test_pred))
print("Precision:", precision_score(y_test, test_pred, zero_division=0))
print("Recall   :", recall_score(y_test, test_pred, zero_division=0))
print("F1 Score :", f1_score(y_test, test_pred, zero_division=0))

print("\nClassification Report")
print(classification_report(y_test, test_pred, zero_division=0))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, test_pred))

# ==========================================
# SAVE RESULT
# ==========================================

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": test_pred,
    "Probability": test_prob
})

results.to_csv("pure_lstm_predictions.csv", index=False)

print("\nSaved: pure_lstm_predictions.csv")