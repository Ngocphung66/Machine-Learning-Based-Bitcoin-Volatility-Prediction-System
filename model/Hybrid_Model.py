# ==========================================
# HYBRID LSTM - RANDOM FOREST FINAL VERSION
# Threshold Optimization: Balanced Accuracy
# Binary Classification: Bitcoin Up / Down
# ==========================================

import os
import pandas as pd
import numpy as np
import warnings

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    matthews_corrcoef
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

# ==========================================
# CONFIG
# ==========================================

DATA_PATH = "data/04_Master_Dataset.csv"

if not os.path.exists(DATA_PATH):
    DATA_PATH = "../data/04_Master_Dataset.csv"

OUTPUT_DIR = "model_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOOKBACK = 14
RETURN_THRESHOLD = 0.0
RANDOM_STATE = 42

# ==========================================
# LOAD DATA
# ==========================================

print("Loading dataset...")

df = pd.read_csv(
    DATA_PATH,
    parse_dates=["Date"],
    index_col="Date"
)

df = df.sort_index()

print("Dataset shape:", df.shape)
print("Date range:", df.index.min(), "to", df.index.max())

# ==========================================
# FEATURE SETS
# ==========================================

technical_features = [
    "Open", "High", "Low", "Close", "Volume",

    "SMA_14", "EMA_14",

    "Ichimoku_Tenkan",
    "Ichimoku_Kijun",
    "Ichimoku_Senkou_A",
    "Ichimoku_Senkou_B",

    "ADX_14",

    "RSI_14",

    "MACD",
    "MACD_Signal",
    "MACD_Hist",

    "Williams_R",

    "K", "D", "J",

    "SQZ_Momentum",

    "BB_Middle",
    "BB_Upper",
    "BB_Lower",
    "BB_Width",

    "ATR_14"
]

# ==========================================
# TARGET CONSTRUCTION
# ==========================================

df["BTC_Return"] = np.log(df["Close"] / df["Close"].shift(1))
df["Future_Return"] = df["BTC_Return"].shift(-1)

# ==========================================
# FUNDAMENTAL FEATURE ENGINEERING
# ==========================================

df["SP500_Return"] = np.log(df["SP500"] / df["SP500"].shift(1))
df["DXY_Return"] = np.log(df["DXY"] / df["DXY"].shift(1))
df["CPI_Change"] = df["CPI"].pct_change()

fundamental_features = [
    "FearGreed",
    "SP500_Return",
    "DXY_Return",
    "FED",
    "CPI_Change"
]

# ==========================================
# CLEAN DATA
# ==========================================

required_cols = technical_features + fundamental_features + ["Future_Return"]

df = df.dropna(subset=required_cols)

df["Target"] = (
    df["Future_Return"] > RETURN_THRESHOLD
).astype(int)

print("\nFinal dataset shape:", df.shape)

print("\nTarget distribution:")
print(df["Target"].value_counts())
print(df["Target"].value_counts(normalize=True))

# ==========================================
# TIME-SERIES TRAIN / VALIDATION / TEST SPLIT
# ==========================================

n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

print("\nSplit information:")
print("Train:", train_df.shape)
print("Val  :", val_df.shape)
print("Test :", test_df.shape)

print("\nTrain period:", train_df.index.min(), "to", train_df.index.max())
print("Val period  :", val_df.index.min(), "to", val_df.index.max())
print("Test period :", test_df.index.min(), "to", test_df.index.max())

# ==========================================
# STAGE 1: LSTM
# Technical Variables -> LSTM_Probability
# ==========================================

print("\n=================================")
print("STAGE 1: LSTM")
print("=================================")

# ==========================================
# SCALE TECHNICAL FEATURES
# Fit scaler on train only
# ==========================================

scaler = MinMaxScaler()

X_train_tech_scaled = scaler.fit_transform(train_df[technical_features])
X_val_tech_scaled = scaler.transform(val_df[technical_features])
X_test_tech_scaled = scaler.transform(test_df[technical_features])

# ==========================================
# CREATE LSTM SEQUENCES
# ==========================================

def create_sequences(X_scaled, y_series, index_series, lookback):
    X_seq = []
    y_seq = []
    dates = []

    for i in range(lookback, len(X_scaled)):
        X_seq.append(X_scaled[i - lookback:i])
        y_seq.append(y_series.iloc[i])
        dates.append(index_series[i])

    return np.array(X_seq), np.array(y_seq), pd.Index(dates)


X_train_seq, y_train_seq, train_dates = create_sequences(
    X_train_tech_scaled,
    train_df["Target"],
    train_df.index,
    LOOKBACK
)

X_val_seq, y_val_seq, val_dates = create_sequences(
    X_val_tech_scaled,
    val_df["Target"],
    val_df.index,
    LOOKBACK
)

X_test_seq, y_test_seq, test_dates = create_sequences(
    X_test_tech_scaled,
    test_df["Target"],
    test_df.index,
    LOOKBACK
)

print("\nLSTM sequence shapes:")
print("Train:", X_train_seq.shape, y_train_seq.shape)
print("Val  :", X_val_seq.shape, y_val_seq.shape)
print("Test :", X_test_seq.shape, y_test_seq.shape)

# ==========================================
# CLASS WEIGHT FOR LSTM
# ==========================================

classes = np.unique(y_train_seq)

if len(classes) == 2:
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train_seq
    )

    lstm_class_weights = {
        0: weights[0],
        1: weights[1]
    }
else:
    lstm_class_weights = None

print("\nLSTM class weights:", lstm_class_weights)

# ==========================================
# BUILD 3-LAYER LSTM MODEL
# ==========================================

lstm = Sequential()

lstm.add(
    LSTM(
        units=128,
        activation="tanh",
        return_sequences=True,
        input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])
    )
)
lstm.add(Dropout(0.2))

lstm.add(
    LSTM(
        units=64,
        activation="tanh",
        return_sequences=True
    )
)
lstm.add(Dropout(0.2))

lstm.add(
    LSTM(
        units=32,
        activation="tanh"
    )
)
lstm.add(Dropout(0.2))

lstm.add(Dense(1, activation="sigmoid"))

lstm.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

lstm.summary()

# ==========================================
# EARLY STOPPING
# ==========================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)

# ==========================================
# TRAIN LSTM
# ==========================================

history = lstm.fit(
    X_train_seq,
    y_train_seq,
    epochs=200,
    batch_size=32,
    validation_data=(X_val_seq, y_val_seq),
    class_weight=lstm_class_weights,
    callbacks=[early_stop],
    verbose=1
)

# ==========================================
# GENERATE LSTM PROBABILITY
# ==========================================

print("\nGenerating LSTM probabilities...")

lstm_prob_train = lstm.predict(X_train_seq).flatten()
lstm_prob_val = lstm.predict(X_val_seq).flatten()
lstm_prob_test = lstm.predict(X_test_seq).flatten()

print("\nLSTM Probability Summary:")
print("Train:")
print(pd.Series(lstm_prob_train).describe())

print("\nValidation:")
print(pd.Series(lstm_prob_val).describe())

print("\nTest:")
print(pd.Series(lstm_prob_test).describe())

# ==========================================
# BUILD HYBRID DATASET FOR STAGE 2
# ==========================================

rf_train = train_df.loc[train_dates].copy()
rf_val = val_df.loc[val_dates].copy()
rf_test = test_df.loc[test_dates].copy()

rf_train["LSTM_Probability"] = lstm_prob_train
rf_val["LSTM_Probability"] = lstm_prob_val
rf_test["LSTM_Probability"] = lstm_prob_test

rf_features = [
    "LSTM_Probability"
] + fundamental_features

X_rf_train = rf_train[rf_features]
X_rf_val = rf_val[rf_features]
X_rf_test = rf_test[rf_features]

y_rf_train = rf_train["Target"]
y_rf_val = rf_val["Target"]
y_rf_test = rf_test["Target"]

print("\nHybrid RF dataset shapes:")
print("RF Train:", X_rf_train.shape, y_rf_train.shape)
print("RF Val  :", X_rf_val.shape, y_rf_val.shape)
print("RF Test :", X_rf_test.shape, y_rf_test.shape)

# ==========================================
# STAGE 2: RANDOM FOREST
# LSTM_Probability + Fundamental Variables -> Final Prediction
# ==========================================

print("\n=================================")
print("STAGE 2: RANDOM FOREST")
print("=================================")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

print("\nTraining Hybrid Random Forest...")

rf.fit(X_rf_train, y_rf_train)

# ==========================================
# VALIDATION THRESHOLD OPTIMIZATION
# Balanced Accuracy Based
# ==========================================

val_prob = rf.predict_proba(X_rf_val)[:, 1]

print("\nHybrid validation probability summary:")
print(pd.Series(val_prob).describe())

best_threshold = 0.5
best_f1 = -1

threshold_results = []

threshold_grid = np.linspace(
    val_prob.min(),
    val_prob.max(),
    100
)

for threshold in threshold_grid:

    val_pred = (val_prob >= threshold).astype(int)

    # Bỏ qua threshold nếu chỉ predict ra 1 class
    if len(np.unique(val_pred)) < 2:
        continue

    acc = accuracy_score(y_rf_val, val_pred)
    bal_acc = balanced_accuracy_score(y_rf_val, val_pred)
    prec = precision_score(y_rf_val, val_pred, zero_division=0)
    rec = recall_score(y_rf_val, val_pred, zero_division=0)
    f1 = f1_score(y_rf_val, val_pred, zero_division=0)
    mcc = matthews_corrcoef(y_rf_val, val_pred)

    threshold_results.append({
        "Threshold": threshold,
        "Accuracy": acc,
        "Balanced_Accuracy": bal_acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "MCC": mcc
    })

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold


threshold_df = pd.DataFrame(threshold_results)

if threshold_df.empty:
    print("\nWarning: No valid threshold found. Using default threshold = 0.5")
    best_threshold = 0.5
else:
    print("\nBest threshold based on validation F1 Score:")
    print("Threshold:", round(best_threshold, 4))
    print("Validation F1 Score:", best_f1)

# ==========================================
# TEST PREDICTION
# ==========================================

test_prob = rf.predict_proba(X_rf_test)[:, 1]

test_pred = (
    test_prob >= best_threshold
).astype(int)

print("\nHybrid test probability summary:")
print(pd.Series(test_prob).describe())

print("\nPrediction distribution:")
print(pd.Series(test_pred).value_counts())

# ==========================================
# EVALUATION
# ==========================================

accuracy = accuracy_score(y_rf_test, test_pred)
balanced_acc = balanced_accuracy_score(y_rf_test, test_pred)
precision = precision_score(y_rf_test, test_pred, zero_division=0)
recall = recall_score(y_rf_test, test_pred, zero_division=0)
f1 = f1_score(y_rf_test, test_pred, zero_division=0)
mcc = matthews_corrcoef(y_rf_test, test_pred)

try:
    roc_auc = roc_auc_score(y_rf_test, test_prob)
except ValueError:
    roc_auc = np.nan

print("\n=================================")
print("HYBRID LSTM-RF RESULTS")
print("=================================")

print("Best Threshold :", round(best_threshold, 4))
print("Accuracy       :", accuracy)
print("Balanced Acc   :", balanced_acc)
print("Precision      :", precision)
print("Recall         :", recall)
print("F1 Score       :", f1)
print("ROC-AUC        :", roc_auc)
print("MCC            :", mcc)

print("\nClassification Report")
print(classification_report(y_rf_test, test_pred, zero_division=0))

print("\nConfusion Matrix")
print(confusion_matrix(y_rf_test, test_pred))

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    "Feature": rf_features,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n=================================")
print("HYBRID FEATURE IMPORTANCE")
print("=================================")

print(importance)

# ==========================================
# SAVE RESULTS
# ==========================================

predictions = pd.DataFrame({
    "Date": X_rf_test.index,
    "Actual": y_rf_test.values,
    "Predicted": test_pred,
    "LSTM_Probability": rf_test["LSTM_Probability"].values,
    "Hybrid_Probability": test_prob
})

metrics = pd.DataFrame({
    "Model": ["Hybrid LSTM-RF"],
    "Best_Threshold": [best_threshold],
    "Accuracy": [accuracy],
    "Balanced_Accuracy": [balanced_acc],
    "Precision": [precision],
    "Recall": [recall],
    "F1_Score": [f1],
    "ROC_AUC": [roc_auc],
    "MCC": [mcc]
})

lstm_probability_table = pd.concat([
    pd.DataFrame({
        "Date": train_dates,
        "Set": "Train",
        "LSTM_Probability": lstm_prob_train,
        "Actual": y_train_seq
    }),
    pd.DataFrame({
        "Date": val_dates,
        "Set": "Validation",
        "LSTM_Probability": lstm_prob_val,
        "Actual": y_val_seq
    }),
    pd.DataFrame({
        "Date": test_dates,
        "Set": "Test",
        "LSTM_Probability": lstm_prob_test,
        "Actual": y_test_seq
    })
])

predictions.to_csv(
    os.path.join(OUTPUT_DIR, "hybrid_predictions.csv"),
    index=False
)

importance.to_csv(
    os.path.join(OUTPUT_DIR, "hybrid_feature_importance.csv"),
    index=False
)

metrics.to_csv(
    os.path.join(OUTPUT_DIR, "hybrid_metrics.csv"),
    index=False
)

threshold_df.to_csv(
    os.path.join(OUTPUT_DIR, "hybrid_threshold_tuning.csv"),
    index=False
)

lstm_probability_table.to_csv(
    os.path.join(OUTPUT_DIR, "hybrid_lstm_probability.csv"),
    index=False
)

print("\nSaved files:")
print("- model_outputs/hybrid_predictions.csv")
print("- model_outputs/hybrid_feature_importance.csv")
print("- model_outputs/hybrid_metrics.csv")
print("- model_outputs/hybrid_threshold_tuning.csv")
print("- model_outputs/hybrid_lstm_probability.csv")