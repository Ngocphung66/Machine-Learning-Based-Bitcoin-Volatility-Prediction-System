# ==========================================
# PURE RANDOM FOREST - FINAL VERSION
# Binary Classification: Bitcoin Up / Down
# Added: Validation ROC-AUC + Test ROC-AUC
# ==========================================

import os
import pandas as pd
import numpy as np
import warnings

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

warnings.filterwarnings("ignore")

# ==========================================
# CONFIG
# ==========================================

DATA_PATH = "data/04_Master_Dataset.csv"

# Nếu chạy file trong thư mục model/
if not os.path.exists(DATA_PATH):
    DATA_PATH = "../data/04_Master_Dataset.csv"

OUTPUT_DIR = "model_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
# TARGET CONSTRUCTION
# ==========================================

# Daily log return from t-1 to t
df["BTC_Return"] = np.log(
    df["Close"] / df["Close"].shift(1)
)

# Future return from t to t+1
df["Future_Return"] = df["BTC_Return"].shift(-1)

# ==========================================
# FUNDAMENTAL FEATURE ENGINEERING
# ==========================================

# FearGreed giữ dạng level vì là chỉ số tâm lý 0-100
# SP500 và DXY chuyển sang return để giảm non-stationarity
# FED giữ dạng level vì lãi suất chính sách có ý nghĩa theo mức
# CPI chuyển sang pct_change để phản ánh thay đổi lạm phát

df["SP500_Return"] = np.log(
    df["SP500"] / df["SP500"].shift(1)
)

df["DXY_Return"] = np.log(
    df["DXY"] / df["DXY"].shift(1)
)

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

# Drop NaN trước khi tạo target để tránh dòng cuối bị gán nhầm class 0
df = df.dropna(
    subset=fundamental_features + ["Future_Return"]
)

df["Target"] = (
    df["Future_Return"] > RETURN_THRESHOLD
).astype(int)

X = df[fundamental_features]
y = df["Target"]

print("\nFinal dataset shape:", df.shape)

print("\nTarget distribution:")
print(y.value_counts())
print(y.value_counts(normalize=True))

# ==========================================
# TIME-SERIES TRAIN / VALIDATION / TEST SPLIT
# ==========================================

n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train = X.iloc[:train_end]
X_val = X.iloc[train_end:val_end]
X_test = X.iloc[val_end:]

y_train = y.iloc[:train_end]
y_val = y.iloc[train_end:val_end]
y_test = y.iloc[val_end:]

print("\nSplit information:")
print("Train:", X_train.shape, y_train.shape)
print("Val  :", X_val.shape, y_val.shape)
print("Test :", X_test.shape, y_test.shape)

print("\nTrain period:", X_train.index.min(), "to", X_train.index.max())
print("Val period  :", X_val.index.min(), "to", X_val.index.max())
print("Test period :", X_test.index.min(), "to", X_test.index.max())

# ==========================================
# BUILD RANDOM FOREST MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# ==========================================
# TRAIN MODEL
# ==========================================

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

# ==========================================
# VALIDATION PROBABILITY + ROC-AUC
# ==========================================

val_prob = model.predict_proba(X_val)[:, 1]

try:
    val_auc = roc_auc_score(y_val, val_prob)
except ValueError:
    val_auc = np.nan

print("\nValidation probability summary:")
print(pd.Series(val_prob).describe())

print("\nValidation ROC-AUC:", val_auc)

# ==========================================
# VALIDATION THRESHOLD OPTIMIZATION
# ==========================================

best_threshold = 0.5
best_f1 = -1

threshold_results = []

for threshold in np.arange(0.30, 0.71, 0.01):

    val_pred = (
        val_prob >= threshold
    ).astype(int)

    acc = accuracy_score(y_val, val_pred)
    prec = precision_score(y_val, val_pred, zero_division=0)
    rec = recall_score(y_val, val_pred, zero_division=0)
    f1 = f1_score(y_val, val_pred, zero_division=0)

    threshold_results.append({
        "Threshold": threshold,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "Validation_ROC_AUC": val_auc
    })

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

threshold_df = pd.DataFrame(threshold_results)

print("\nBest threshold based on validation F1:")
print("Threshold:", round(best_threshold, 2))
print("Validation F1:", best_f1)
print("Validation ROC-AUC:", val_auc)

# ==========================================
# TEST PREDICTION
# ==========================================

test_prob = model.predict_proba(X_test)[:, 1]

test_pred = (
    test_prob >= best_threshold
).astype(int)

print("\nTest probability summary:")
print(pd.Series(test_prob).describe())

print("\nPrediction distribution:")
print(pd.Series(test_pred).value_counts())

# ==========================================
# EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, test_pred)

precision = precision_score(
    y_test,
    test_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_pred,
    zero_division=0
)

try:
    test_auc = roc_auc_score(y_test, test_prob)
except ValueError:
    test_auc = np.nan

print("\n=================================")
print("PURE RANDOM FOREST RESULTS")
print("=================================")

print("Best Threshold    :", round(best_threshold, 2))
print("Validation ROC-AUC:", val_auc)
print("Test ROC-AUC      :", test_auc)
print("Accuracy          :", accuracy)
print("Precision         :", precision)
print("Recall            :", recall)
print("F1 Score          :", f1)

print("\nClassification Report")
print(classification_report(y_test, test_pred, zero_division=0))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, test_pred))

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    "Feature": fundamental_features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n=================================")
print("FEATURE IMPORTANCE")
print("=================================")
print(importance)

# ==========================================
# SAVE RESULTS
# ==========================================

predictions = pd.DataFrame({
    "Date": X_test.index,
    "Actual": y_test.values,
    "Predicted": test_pred,
    "Probability": test_prob
})

metrics = pd.DataFrame({
    "Model": ["Pure Random Forest"],
    "Best_Threshold": [best_threshold],
    "Validation_ROC_AUC": [val_auc],
    "Test_ROC_AUC": [test_auc],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1_Score": [f1]
})

predictions.to_csv(
    os.path.join(OUTPUT_DIR, "pure_rf_predictions.csv"),
    index=False
)

importance.to_csv(
    os.path.join(OUTPUT_DIR, "pure_rf_feature_importance.csv"),
    index=False
)

metrics.to_csv(
    os.path.join(OUTPUT_DIR, "pure_rf_metrics.csv"),
    index=False
)

threshold_df.to_csv(
    os.path.join(OUTPUT_DIR, "pure_rf_threshold_tuning.csv"),
    index=False
)

print("\nSaved files:")
print("- model_outputs/pure_rf_predictions.csv")
print("- model_outputs/pure_rf_feature_importance.csv")
print("- model_outputs/pure_rf_metrics.csv")
print("- model_outputs/pure_rf_threshold_tuning.csv")