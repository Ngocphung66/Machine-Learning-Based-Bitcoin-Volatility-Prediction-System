# ==========================================
# RUN 3 MODELS: PURE RF / PURE LSTM / HYBRID LSTM-RF
# Assumption: High Volatility Detection
# Target: 1 if next-day absolute log return exceeds high-volatility threshold
# Threshold Selection: Controlled F2-score on Validation Set
# Main Metrics: Accuracy, Balanced Accuracy, Precision, Recall, F1, F2, ROC-AUC, MCC
# controlled f2 
# ==========================================

import os
import random
import warnings

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)

import tensorflow as tf
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

OUTPUT_DIR = "model_outputs_volatility_controlled_f2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOOKBACK = 14
VOLATILITY_QUANTILE = 0.75

# Controlled F2 threshold settings
# Mục tiêu: vẫn ưu tiên phát hiện ngày biến động cao bằng F2,
# nhưng không để mô hình cảnh báo class 1 gần như toàn bộ quan sát.
THRESHOLD_GRID_SIZE = 200
CONTROL_MIN_BALANCED_ACCURACY = 0.50
CONTROL_MIN_MCC = 0.00
CONTROL_MIN_PRED_POSITIVE_RATE = 0.05
CONTROL_MAX_PRED_POSITIVE_RATE = 0.50

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def safe_auc(y_true, y_prob):
    try:
        return roc_auc_score(y_true, y_prob)
    except ValueError:
        return np.nan


def find_best_threshold_by_controlled_f2(
    y_true,
    y_prob,
    min_balanced_accuracy=CONTROL_MIN_BALANCED_ACCURACY,
    min_mcc=CONTROL_MIN_MCC,
    min_pred_positive_rate=CONTROL_MIN_PRED_POSITIVE_RATE,
    max_pred_positive_rate=CONTROL_MAX_PRED_POSITIVE_RATE
):
    """
    Chọn ngưỡng phân loại theta trên validation set bằng Controlled F2.

    Logic:
    - Vẫn tối đa hóa F2-score để ưu tiên Recall.
    - Nhưng chỉ chọn trong các threshold còn duy trì kiểm soát tối thiểu:
        + Balanced Accuracy >= min_balanced_accuracy
        + MCC >= min_mcc
        + Predicted Positive Rate nằm trong [min_pred_positive_rate, max_pred_positive_rate]
    - Điều kiện Predicted Positive Rate giúp tránh tình trạng mô hình cảnh báo
      biến động cao gần như mọi ngày.

    Lưu ý:
    - Đây là ngưỡng theta để chuyển xác suất mô hình thành nhãn 0/1.
    - Không phải ngưỡng tau_vol dùng để tạo Target.
    """

    threshold_grid = np.linspace(
        np.min(y_prob),
        np.max(y_prob),
        THRESHOLD_GRID_SIZE
    )

    threshold_rows = []

    actual_positive_rate = np.mean(y_true)

    for threshold in threshold_grid:
        y_pred = (y_prob >= threshold).astype(int)

        # Tránh trường hợp model predict toàn 0 hoặc toàn 1 khi chọn ngưỡng
        if len(np.unique(y_pred)) < 2:
            continue

        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        auc = safe_auc(y_true, y_prob)
        mcc = matthews_corrcoef(y_true, y_pred)

        pred_positive_rate = np.mean(y_pred)
        pred_positive_count = int(np.sum(y_pred))

        threshold_rows.append({
            "Threshold": threshold,
            "Accuracy": acc,
            "Balanced_Accuracy": bal_acc,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1,
            "F2_Score": f2,
            "ROC_AUC": auc,
            "MCC": mcc,
            "Actual_Positive_Rate": actual_positive_rate,
            "Predicted_Positive_Rate": pred_positive_rate,
            "Predicted_Positive_Count": pred_positive_count
        })

    threshold_df = pd.DataFrame(threshold_rows)

    if threshold_df.empty:
        return 0.5, np.nan, threshold_df

    controlled_df = threshold_df[
        (threshold_df["Balanced_Accuracy"] >= min_balanced_accuracy) &
        (threshold_df["MCC"] >= min_mcc) &
        (threshold_df["Predicted_Positive_Rate"] >= min_pred_positive_rate) &
        (threshold_df["Predicted_Positive_Rate"] <= max_pred_positive_rate)
    ].copy()

    if not controlled_df.empty:
        selection_method = "CONTROLLED_F2"
        candidate_df = controlled_df
        sort_cols = ["F2_Score", "Balanced_Accuracy", "MCC", "Precision"]
    else:
        relaxed_df = threshold_df[
            (threshold_df["Predicted_Positive_Rate"] >= min_pred_positive_rate) &
            (threshold_df["Predicted_Positive_Rate"] <= max_pred_positive_rate)
        ].copy()

        if not relaxed_df.empty:
            selection_method = "RELAXED_F2_WITH_PRED_RATE_CONTROL"
            candidate_df = relaxed_df
            sort_cols = ["F2_Score", "Balanced_Accuracy", "MCC", "Precision"]
        else:
            selection_method = "FALLBACK_MAX_BALANCED_ACCURACY"
            candidate_df = threshold_df.copy()
            sort_cols = ["Balanced_Accuracy", "MCC", "F2_Score", "Precision"]

    selected_idx = (
        candidate_df
        .sort_values(by=sort_cols, ascending=False)
        .index[0]
    )

    best_threshold = threshold_df.loc[selected_idx, "Threshold"]
    best_f2 = threshold_df.loc[selected_idx, "F2_Score"]

    threshold_df["Is_Controlled_Candidate"] = False
    if not controlled_df.empty:
        threshold_df.loc[controlled_df.index, "Is_Controlled_Candidate"] = True

    threshold_df["Selected"] = False
    threshold_df.loc[selected_idx, "Selected"] = True
    threshold_df["Selection_Method"] = ""
    threshold_df.loc[selected_idx, "Selection_Method"] = selection_method

    print("\nControlled F2 threshold selection")
    print("Selection method:", selection_method)
    print("Selected threshold:", best_threshold)
    print("Selected F2:", best_f2)
    print("Selected Balanced Accuracy:", threshold_df.loc[selected_idx, "Balanced_Accuracy"])
    print("Selected MCC:", threshold_df.loc[selected_idx, "MCC"])
    print("Selected Predicted Positive Rate:", threshold_df.loc[selected_idx, "Predicted_Positive_Rate"])
    print("Actual Positive Rate:", actual_positive_rate)

    return best_threshold, best_f2, threshold_df


def evaluate_predictions(model_name, set_name, y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    pred_positive_rate = np.mean(y_pred)
    actual_positive_rate = np.mean(y_true)

    result = {
        "Model": model_name,
        "Set": set_name,
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1_Score": f1_score(y_true, y_pred, zero_division=0),
        "F2_Score": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "ROC_AUC": safe_auc(y_true, y_prob),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Actual_Positive_Rate": actual_positive_rate,
        "Predicted_Positive_Rate": pred_positive_rate
    }

    print("\n=================================")
    print(f"{model_name} - {set_name}")
    print("=================================")
    print("Threshold              :", result["Threshold"])
    print("Accuracy               :", result["Accuracy"])
    print("Balanced Accuracy      :", result["Balanced_Accuracy"])
    print("Precision              :", result["Precision"])
    print("Recall                 :", result["Recall"])
    print("F1 Score               :", result["F1_Score"])
    print("F2 Score               :", result["F2_Score"])
    print("ROC-AUC                :", result["ROC_AUC"])
    print("MCC                    :", result["MCC"])
    print("Actual Positive Rate   :", result["Actual_Positive_Rate"])
    print("Predicted Positive Rate:", result["Predicted_Positive_Rate"])

    print("\nClassification Report")
    print(classification_report(y_true, y_pred, zero_division=0))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_true, y_pred))

    return result, y_pred


def create_sequences(X_scaled, y_series, index_series, lookback):
    X_seq = []
    y_seq = []
    dates = []

    for i in range(lookback, len(X_scaled)):
        X_seq.append(X_scaled[i - lookback:i])
        y_seq.append(y_series.iloc[i])
        dates.append(index_series[i])

    return np.array(X_seq), np.array(y_seq), pd.Index(dates)


def build_lstm_model(input_shape):
    model = Sequential()

    # model.add(
    #     LSTM(
    #         units=128,
    #         activation="tanh",
    #         return_sequences=True,
    #         input_shape=input_shape
    #     )
    # )
    # model.add(Dropout(0.2))

    model.add(
        LSTM(
            units=64,
            activation="tanh",
            return_sequences=True
        )
    )
    model.add(Dropout(0.2))

    model.add(
        LSTM(
            units=32,
            activation="tanh"
        )
    )
    model.add(Dropout(0.2))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


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

# Fundamental feature engineering
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
# TARGET CONSTRUCTION: HIGH VOLATILITY / NORMAL VOLATILITY
# ==========================================

# Log return of Bitcoin
df["BTC_Return"] = np.log(df["Close"] / df["Close"].shift(1))

# Next-day return
df["Future_Return"] = df["BTC_Return"].shift(-1)

# Next-day volatility magnitude: absolute return
# Target will be based on the magnitude of movement, not direction.
df["Future_Volatility"] = df["Future_Return"].abs()

required_cols = technical_features + fundamental_features + ["Future_Volatility"]

# Drop rows that do not have enough information for features or future volatility
df = df.dropna(subset=required_cols)

print("\nFinal dataset shape before target assignment:", df.shape)

# ==========================================
# TRAIN / VALIDATION / TEST SPLIT
# ==========================================

n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

# High-volatility threshold must be computed ONLY on train set
# to avoid data leakage from validation/test.
VOLATILITY_THRESHOLD = train_df["Future_Volatility"].quantile(VOLATILITY_QUANTILE)

print("\nHigh-volatility threshold:")
print(f"Quantile used: {VOLATILITY_QUANTILE}")
print(f"VOLATILITY_THRESHOLD: {VOLATILITY_THRESHOLD}")

# Target = 1 means next-day Bitcoin volatility is high.
# Target = 0 means next-day Bitcoin volatility is normal/low.
for split_df in [train_df, val_df, test_df]:
    split_df["Target"] = (
        split_df["Future_Volatility"] > VOLATILITY_THRESHOLD
    ).astype(int)

# Recombine only for reporting target distribution
df = pd.concat([train_df, val_df, test_df], axis=0).sort_index()

print("\nTarget meaning:")
print("Target = 1: next-day Bitcoin volatility is HIGH")
print("Target = 0: next-day Bitcoin volatility is NORMAL/LOW")

print("\nTarget distribution - Full dataset:")
print(df["Target"].value_counts())
print(df["Target"].value_counts(normalize=True))

print("\nTarget distribution - Train:")
print(train_df["Target"].value_counts())
print(train_df["Target"].value_counts(normalize=True))

print("\nTarget distribution - Validation:")
print(val_df["Target"].value_counts())
print(val_df["Target"].value_counts(normalize=True))

print("\nTarget distribution - Test:")
print(test_df["Target"].value_counts())
print(test_df["Target"].value_counts(normalize=True))

print("\nSplit information:")
print("Train:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)

print("\nTrain period:", train_df.index.min(), "to", train_df.index.max())
print("Validation period:", val_df.index.min(), "to", val_df.index.max())
print("Test period:", test_df.index.min(), "to", test_df.index.max())

pd.DataFrame({
    "Parameter": ["VOLATILITY_QUANTILE", "VOLATILITY_THRESHOLD"],
    "Value": [VOLATILITY_QUANTILE, VOLATILITY_THRESHOLD]
}).to_csv(
    os.path.join(OUTPUT_DIR, "target_volatility_threshold.csv"),
    index=False
)

all_metrics = []
all_predictions = []

# ==========================================
# MODEL 1: PURE RANDOM FOREST
# ==========================================

print("\n\n=================================")
print("MODEL 1: PURE RANDOM FOREST")
print("=================================")

X_rf_train = train_df[fundamental_features]
X_rf_val = val_df[fundamental_features]
X_rf_test = test_df[fundamental_features]

y_rf_train = train_df["Target"]
y_rf_val = val_df["Target"]
y_rf_test = test_df["Target"]

pure_rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=4,
    min_samples_split=30,
    min_samples_leaf=20,
    max_features="sqrt",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

pure_rf.fit(X_rf_train, y_rf_train)

pure_rf_train_prob = pure_rf.predict_proba(X_rf_train)[:, 1]
pure_rf_val_prob = pure_rf.predict_proba(X_rf_val)[:, 1]
pure_rf_test_prob = pure_rf.predict_proba(X_rf_test)[:, 1]

pure_rf_threshold, pure_rf_val_f2, pure_rf_threshold_df = (
    find_best_threshold_by_controlled_f2(
        y_rf_val,
        pure_rf_val_prob
    )
)

print("\nPure RF controlled threshold:", pure_rf_threshold)
print("Pure RF validation controlled F2-score:", pure_rf_val_f2)

for set_name, y_true, y_prob, index in [
    ("Train", y_rf_train, pure_rf_train_prob, X_rf_train.index),
    ("Validation", y_rf_val, pure_rf_val_prob, X_rf_val.index),
    ("Test", y_rf_test, pure_rf_test_prob, X_rf_test.index)
]:
    metrics, pred = evaluate_predictions(
        "Pure Random Forest",
        set_name,
        y_true,
        y_prob,
        pure_rf_threshold
    )

    all_metrics.append(metrics)

    all_predictions.append(pd.DataFrame({
        "Date": index,
        "Model": "Pure Random Forest",
        "Set": set_name,
        "Actual": y_true.values,
        "Predicted": pred,
        "Probability": y_prob
    }))

pure_rf_importance = pd.DataFrame({
    "Feature": fundamental_features,
    "Importance": pure_rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n=================================")
print("PURE RF FEATURE IMPORTANCE")
print("=================================")
print(pure_rf_importance)

pure_rf_threshold_df.to_csv(
    os.path.join(OUTPUT_DIR, "pure_rf_threshold_tuning_controlled_f2.csv"),
    index=False
)

pure_rf_importance.to_csv(
    os.path.join(OUTPUT_DIR, "pure_rf_feature_importance.csv"),
    index=False
)

# ==========================================
# MODEL 2: PURE LSTM
# ==========================================

print("\n\n=================================")
print("MODEL 2: PURE LSTM")
print("=================================")

scaler = MinMaxScaler()

X_train_tech_scaled = scaler.fit_transform(train_df[technical_features])
X_val_tech_scaled = scaler.transform(val_df[technical_features])
X_test_tech_scaled = scaler.transform(test_df[technical_features])

X_lstm_train, y_lstm_train, lstm_train_dates = create_sequences(
    X_train_tech_scaled,
    train_df["Target"],
    train_df.index,
    LOOKBACK
)

X_lstm_val, y_lstm_val, lstm_val_dates = create_sequences(
    X_val_tech_scaled,
    val_df["Target"],
    val_df.index,
    LOOKBACK
)

X_lstm_test, y_lstm_test, lstm_test_dates = create_sequences(
    X_test_tech_scaled,
    test_df["Target"],
    test_df.index,
    LOOKBACK
)

print("LSTM Train:", X_lstm_train.shape)
print("LSTM Validation:", X_lstm_val.shape)
print("LSTM Test:", X_lstm_test.shape)

classes = np.unique(y_lstm_train)

if len(classes) == 2:
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_lstm_train
    )

    lstm_class_weights = {
        0: weights[0],
        1: weights[1]
    }
else:
    lstm_class_weights = None

print("LSTM class weights:", lstm_class_weights)

pure_lstm = build_lstm_model(
    input_shape=(X_lstm_train.shape[1], X_lstm_train.shape[2])
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)

history = pure_lstm.fit(
    X_lstm_train,
    y_lstm_train,
    epochs=200,
    batch_size=32,
    validation_data=(X_lstm_val, y_lstm_val),
    class_weight=lstm_class_weights,
    callbacks=[early_stop],
    verbose=1
)

pure_lstm_train_prob = pure_lstm.predict(X_lstm_train).flatten()
pure_lstm_val_prob = pure_lstm.predict(X_lstm_val).flatten()
pure_lstm_test_prob = pure_lstm.predict(X_lstm_test).flatten()

pure_lstm_threshold, pure_lstm_val_f2, pure_lstm_threshold_df = (
    find_best_threshold_by_controlled_f2(
        y_lstm_val,
        pure_lstm_val_prob
    )
)

print("\nPure LSTM controlled threshold:", pure_lstm_threshold)
print("Pure LSTM validation controlled F2-score:", pure_lstm_val_f2)

for set_name, y_true, y_prob, index in [
    ("Train", y_lstm_train, pure_lstm_train_prob, lstm_train_dates),
    ("Validation", y_lstm_val, pure_lstm_val_prob, lstm_val_dates),
    ("Test", y_lstm_test, pure_lstm_test_prob, lstm_test_dates)
]:
    metrics, pred = evaluate_predictions(
        "Pure LSTM",
        set_name,
        y_true,
        y_prob,
        pure_lstm_threshold
    )

    all_metrics.append(metrics)

    all_predictions.append(pd.DataFrame({
        "Date": index,
        "Model": "Pure LSTM",
        "Set": set_name,
        "Actual": y_true,
        "Predicted": pred,
        "Probability": y_prob
    }))

print("\n=================================")
print("PURE LSTM NOTE")
print("=================================")
print("LSTM không có feature_importances_ native như Random Forest.")
print("Nếu cần feature importance cho LSTM, nên dùng permutation importance hoặc SHAP ở bước phân tích mở rộng.")

pure_lstm_threshold_df.to_csv(
    os.path.join(OUTPUT_DIR, "pure_lstm_threshold_tuning_controlled_f2.csv"),
    index=False
)

# ==========================================
# MODEL 3: HYBRID LSTM-RF
# ==========================================

print("\n\n=================================")
print("MODEL 3: HYBRID LSTM-RF")
print("=================================")

hybrid_train_df = train_df.loc[lstm_train_dates].copy()
hybrid_val_df = val_df.loc[lstm_val_dates].copy()
hybrid_test_df = test_df.loc[lstm_test_dates].copy()

hybrid_train_df["LSTM_Probability"] = pure_lstm_train_prob
hybrid_val_df["LSTM_Probability"] = pure_lstm_val_prob
hybrid_test_df["LSTM_Probability"] = pure_lstm_test_prob

hybrid_features = [
    "LSTM_Probability"
] + fundamental_features

X_hybrid_train = hybrid_train_df[hybrid_features]
X_hybrid_val = hybrid_val_df[hybrid_features]
X_hybrid_test = hybrid_test_df[hybrid_features]

y_hybrid_train = hybrid_train_df["Target"]
y_hybrid_val = hybrid_val_df["Target"]
y_hybrid_test = hybrid_test_df["Target"]

hybrid_rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=3,
    min_samples_split=40,
    min_samples_leaf=25,
    max_features="sqrt",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

hybrid_rf.fit(X_hybrid_train, y_hybrid_train)

hybrid_train_prob = hybrid_rf.predict_proba(X_hybrid_train)[:, 1]
hybrid_val_prob = hybrid_rf.predict_proba(X_hybrid_val)[:, 1]
hybrid_test_prob = hybrid_rf.predict_proba(X_hybrid_test)[:, 1]

hybrid_threshold, hybrid_val_f2, hybrid_threshold_df = (
    find_best_threshold_by_controlled_f2(
        y_hybrid_val,
        hybrid_val_prob
    )
)

print("\nHybrid controlled threshold:", hybrid_threshold)
print("Hybrid validation controlled F2-score:", hybrid_val_f2)

for set_name, y_true, y_prob, index in [
    ("Train", y_hybrid_train, hybrid_train_prob, X_hybrid_train.index),
    ("Validation", y_hybrid_val, hybrid_val_prob, X_hybrid_val.index),
    ("Test", y_hybrid_test, hybrid_test_prob, X_hybrid_test.index)
]:
    metrics, pred = evaluate_predictions(
        "Hybrid LSTM-RF",
        set_name,
        y_true,
        y_prob,
        hybrid_threshold
    )

    all_metrics.append(metrics)

    all_predictions.append(pd.DataFrame({
        "Date": index,
        "Model": "Hybrid LSTM-RF",
        "Set": set_name,
        "Actual": y_true.values,
        "Predicted": pred,
        "Probability": y_prob
    }))

hybrid_importance = pd.DataFrame({
    "Feature": hybrid_features,
    "Importance": hybrid_rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n=================================")
print("HYBRID FEATURE IMPORTANCE")
print("=================================")
print(hybrid_importance)

hybrid_threshold_df.to_csv(
    os.path.join(OUTPUT_DIR, "hybrid_threshold_tuning_controlled_f2.csv"),
    index=False
)

hybrid_importance.to_csv(
    os.path.join(OUTPUT_DIR, "hybrid_feature_importance.csv"),
    index=False
)

hybrid_lstm_probability = pd.concat([
    pd.DataFrame({
        "Date": lstm_train_dates,
        "Set": "Train",
        "LSTM_Probability": pure_lstm_train_prob,
        "Actual": y_lstm_train
    }),
    pd.DataFrame({
        "Date": lstm_val_dates,
        "Set": "Validation",
        "LSTM_Probability": pure_lstm_val_prob,
        "Actual": y_lstm_val
    }),
    pd.DataFrame({
        "Date": lstm_test_dates,
        "Set": "Test",
        "LSTM_Probability": pure_lstm_test_prob,
        "Actual": y_lstm_test
    })
])

hybrid_lstm_probability.to_csv(
    os.path.join(OUTPUT_DIR, "hybrid_lstm_probability.csv"),
    index=False
)

# ==========================================
# SAVE FINAL COMPARISON TABLES
# ==========================================

metrics_df = pd.DataFrame(all_metrics)

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)

metrics_df.to_csv(
    os.path.join(OUTPUT_DIR, "model_comparison_metrics_train_val_test.csv"),
    index=False
)

predictions_df.to_csv(
    os.path.join(OUTPUT_DIR, "all_model_predictions_train_val_test.csv"),
    index=False
)

# Test-only comparison
test_comparison = metrics_df[
    metrics_df["Set"] == "Test"
].copy()

test_comparison = test_comparison.sort_values(
    by="F2_Score",
    ascending=False
)

test_comparison.to_csv(
    os.path.join(OUTPUT_DIR, "test_model_comparison_controlled_f2.csv"),
    index=False
)

# Pivot tables
accuracy_table = metrics_df.pivot(
    index="Model",
    columns="Set",
    values="Accuracy"
)

balanced_accuracy_table = metrics_df.pivot(
    index="Model",
    columns="Set",
    values="Balanced_Accuracy"
)

precision_table = metrics_df.pivot(
    index="Model",
    columns="Set",
    values="Precision"
)

recall_table = metrics_df.pivot(
    index="Model",
    columns="Set",
    values="Recall"
)

f1_table = metrics_df.pivot(
    index="Model",
    columns="Set",
    values="F1_Score"
)

f2_table = metrics_df.pivot(
    index="Model",
    columns="Set",
    values="F2_Score"
)

auc_table = metrics_df.pivot(
    index="Model",
    columns="Set",
    values="ROC_AUC"
)

mcc_table = metrics_df.pivot(
    index="Model",
    columns="Set",
    values="MCC"
)

pred_positive_rate_table = metrics_df.pivot(
    index="Model",
    columns="Set",
    values="Predicted_Positive_Rate"
)

actual_positive_rate_table = metrics_df.pivot(
    index="Model",
    columns="Set",
    values="Actual_Positive_Rate"
)

accuracy_table.to_csv(
    os.path.join(OUTPUT_DIR, "comparison_accuracy_table.csv")
)

balanced_accuracy_table.to_csv(
    os.path.join(OUTPUT_DIR, "comparison_balanced_accuracy_table.csv")
)

precision_table.to_csv(
    os.path.join(OUTPUT_DIR, "comparison_precision_table.csv")
)

recall_table.to_csv(
    os.path.join(OUTPUT_DIR, "comparison_recall_table.csv")
)

f1_table.to_csv(
    os.path.join(OUTPUT_DIR, "comparison_f1_table.csv")
)

f2_table.to_csv(
    os.path.join(OUTPUT_DIR, "comparison_f2_table.csv")
)

auc_table.to_csv(
    os.path.join(OUTPUT_DIR, "comparison_auc_table.csv")
)

mcc_table.to_csv(
    os.path.join(OUTPUT_DIR, "comparison_mcc_table.csv")
)

pred_positive_rate_table.to_csv(
    os.path.join(OUTPUT_DIR, "comparison_predicted_positive_rate_table.csv")
)

actual_positive_rate_table.to_csv(
    os.path.join(OUTPUT_DIR, "comparison_actual_positive_rate_table.csv")
)

# ==========================================
# PRINT FINAL TABLES
# ==========================================

print("\n\n=================================")
print("FINAL MODEL COMPARISON - TRAIN / VALIDATION / TEST")
print("=================================")
print(metrics_df)

print("\n\n=================================")
print("ACCURACY TABLE")
print("=================================")
print(accuracy_table)

print("\n\n=================================")
print("BALANCED ACCURACY TABLE")
print("=================================")
print(balanced_accuracy_table)

print("\n\n=================================")
print("PRECISION TABLE")
print("=================================")
print(precision_table)

print("\n\n=================================")
print("RECALL TABLE")
print("=================================")
print(recall_table)

print("\n\n=================================")
print("F1 TABLE")
print("=================================")
print(f1_table)

print("\n\n=================================")
print("F2 TABLE")
print("=================================")
print(f2_table)

print("\n\n=================================")
print("ROC-AUC TABLE")
print("=================================")
print(auc_table)

print("\n\n=================================")
print("MCC TABLE")
print("=================================")
print(mcc_table)

print("\n\n=================================")
print("PREDICTED POSITIVE RATE TABLE")
print("=================================")
print(pred_positive_rate_table)

print("\n\n=================================")
print("ACTUAL POSITIVE RATE TABLE")
print("=================================")
print(actual_positive_rate_table)

print("\n\n=================================")
print("PURE RF FEATURE IMPORTANCE")
print("=================================")
print(pure_rf_importance)

print("\n\n=================================")
print("HYBRID FEATURE IMPORTANCE")
print("=================================")
print(hybrid_importance)

print("\n\n=================================")
print("TEST SET COMPARISON - SORTED BY CONTROLLED F2 SCORE")
print("=================================")
print(test_comparison)

print("\nSaved files:")
print("- model_outputs_volatility/model_comparison_metrics_train_val_test.csv")
print("- model_outputs_volatility/all_model_predictions_train_val_test.csv")
print("- model_outputs_volatility/test_model_comparison_controlled_f2.csv")
print("- model_outputs_volatility/comparison_accuracy_table.csv")
print("- model_outputs_volatility/comparison_balanced_accuracy_table.csv")
print("- model_outputs_volatility/comparison_precision_table.csv")
print("- model_outputs_volatility/comparison_recall_table.csv")
print("- model_outputs_volatility/comparison_f1_table.csv")
print("- model_outputs_volatility/comparison_f2_table.csv")
print("- model_outputs_volatility/comparison_auc_table.csv")
print("- model_outputs_volatility/comparison_mcc_table.csv")
print("- model_outputs_volatility/pure_rf_feature_importance.csv")
print("- model_outputs_volatility/hybrid_feature_importance.csv")
print("- model_outputs_volatility/hybrid_lstm_probability.csv")