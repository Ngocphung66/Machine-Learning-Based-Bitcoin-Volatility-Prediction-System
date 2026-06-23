import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import adfuller

import warnings
warnings.filterwarnings("ignore")

# ==========================================
# CONFIG
# ==========================================

plt.style.use("seaborn-v0_8")

pd.set_option("display.max_columns", None)

# ==========================================
# LOAD DATASET
# ==========================================

print("Loading dataset...")

df = pd.read_csv(
    "data/04_Master_Dataset.csv",
    parse_dates=["Date"],
    index_col="Date"
)

print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

# ==========================================
# BASIC INFORMATION
# ==========================================

print("\n================ DATA INFO ================")
print(df.info())

print("\n================ COLUMN NAMES ================")
print(df.columns.tolist())

print("\n================ DATA TYPES ================")
print(df.dtypes)

# ==========================================
# MISSING VALUE ANALYSIS
# ==========================================

print("\n================ MISSING VALUES ================")

missing = df.isnull().sum()

missing = missing[missing > 0]

print(missing)

plt.figure(figsize=(14,6))

sns.heatmap(
    df.isnull(),
    cbar=False
)

plt.title("Missing Value Heatmap")

plt.show()

# ==========================================
# DESCRIPTIVE STATISTICS
# ==========================================

print("\n================ DESCRIPTIVE STATISTICS ================")

desc = df.describe().T

desc["skewness"] = df.skew(numeric_only=True)
desc["kurtosis"] = df.kurtosis(numeric_only=True)

print(desc)

# Save descriptive statistics
desc.to_csv("results_descriptive_statistics.csv")

# ==========================================
# CREATE RETURN & TARGET
# ==========================================

print("\n================ TARGET CREATION ================")

df["Return"] = np.log(
    df["Close"] / df["Close"].shift(1)
)

df["Target"] = (
    df["Return"].shift(-1) > 0
).astype(int)

print(df[["Return", "Target"]].head())

# ==========================================
# TARGET DISTRIBUTION
# ==========================================

print("\n================ TARGET DISTRIBUTION ================")

print(df["Target"].value_counts())

plt.figure(figsize=(6,4))

sns.countplot(x=df["Target"])

plt.title("Target Distribution")

plt.xlabel("Target")
plt.ylabel("Count")

plt.show()

# ==========================================
# BTC PRICE ANALYSIS
# ==========================================

print("\n================ BTC PRICE ANALYSIS ================")

plt.figure(figsize=(16,6))

plt.plot(
    df.index,
    df["Close"]
)

plt.title("BTC Close Price")

plt.xlabel("Date")
plt.ylabel("Price")

plt.show()

# ==========================================
# BTC RETURN ANALYSIS
# ==========================================

plt.figure(figsize=(16,6))

plt.plot(
    df.index,
    df["Return"]
)

plt.title("BTC Daily Log Return")

plt.xlabel("Date")
plt.ylabel("Return")

plt.show()

# ==========================================
# ROLLING VOLATILITY
# ==========================================

df["Volatility_30"] = (
    df["Return"]
    .rolling(30)
    .std()
)

plt.figure(figsize=(16,6))

plt.plot(
    df.index,
    df["Volatility_30"]
)

plt.title("30-Day Rolling Volatility")

plt.xlabel("Date")
plt.ylabel("Volatility")

plt.show()

# ==========================================
# RSI ANALYSIS
# ==========================================

if "RSI_14" in df.columns:

    plt.figure(figsize=(16,6))

    plt.plot(
        df.index,
        df["RSI_14"]
    )

    plt.axhline(
        70,
        color="red",
        linestyle="--"
    )

    plt.axhline(
        30,
        color="green",
        linestyle="--"
    )

    plt.title("RSI Indicator")

    plt.show()

# ==========================================
# MACD ANALYSIS
# ==========================================

if "MACD" in df.columns and "MACD_Signal" in df.columns:

    plt.figure(figsize=(16,6))

    plt.plot(
        df.index,
        df["MACD"],
        label="MACD"
    )

    plt.plot(
        df.index,
        df["MACD_Signal"],
        label="Signal"
    )

    plt.legend()

    plt.title("MACD Indicator")

    plt.show()

# ==========================================
# BOLLINGER BANDS
# ==========================================

if (
    "BB_Upper" in df.columns and
    "BB_Lower" in df.columns
):

    plt.figure(figsize=(16,6))

    plt.plot(
        df.index,
        df["Close"],
        label="Close"
    )

    plt.plot(
        df.index,
        df["BB_Upper"],
        linestyle="--",
        label="Upper Band"
    )

    plt.plot(
        df.index,
        df["BB_Lower"],
        linestyle="--",
        label="Lower Band"
    )

    plt.legend()

    plt.title("Bollinger Bands")

    plt.show()

# ==========================================
# ADX ANALYSIS
# ==========================================

if "ADX_14" in df.columns:

    plt.figure(figsize=(16,6))

    plt.plot(
        df.index,
        df["ADX_14"]
    )

    plt.axhline(
        25,
        linestyle="--"
    )

    plt.title("ADX Trend Strength")

    plt.show()

# ==========================================
# FEAR & GREED
# ==========================================

if "FearGreed" in df.columns:

    plt.figure(figsize=(16,6))

    plt.plot(
        df.index,
        df["FearGreed"]
    )

    plt.title("Fear & Greed Index")

    plt.show()

# ==========================================
# BTC VS SP500
# ==========================================

if "SP500" in df.columns:

    fig, ax1 = plt.subplots(figsize=(16,6))

    ax1.plot(
        df.index,
        df["Close"],
        label="BTC"
    )

    ax1.set_ylabel("BTC")

    ax2 = ax1.twinx()

    ax2.plot(
        df.index,
        df["SP500"],
        color="orange",
        label="SP500"
    )

    ax2.set_ylabel("SP500")

    plt.title("BTC vs SP500")

    plt.show()

# ==========================================
# DXY
# ==========================================

if "DXY" in df.columns:

    plt.figure(figsize=(16,6))

    plt.plot(
        df.index,
        df["DXY"]
    )

    plt.title("US Dollar Index (DXY)")

    plt.show()

# ==========================================
# FED RATE
# ==========================================

if "FED" in df.columns:

    plt.figure(figsize=(16,6))

    plt.plot(
        df.index,
        df["FED"]
    )

    plt.title("FED Interest Rate")

    plt.show()

# ==========================================
# CPI
# ==========================================

if "CPI" in df.columns:

    plt.figure(figsize=(16,6))

    plt.plot(
        df.index,
        df["CPI"]
    )

    plt.title("Consumer Price Index")

    plt.show()

# ==========================================
# CORRELATION ANALYSIS
# ==========================================

print("\n================ CORRELATION MATRIX ================")

corr = df.corr(numeric_only=True)

plt.figure(figsize=(20,14))

sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Matrix")

plt.show()

# Save correlation matrix
corr.to_csv("../results_correlation_matrix.csv")

# ==========================================
# DISTRIBUTION ANALYSIS
# ==========================================

print("\n================ DISTRIBUTION ANALYSIS ================")

# BTC Return Distribution
plt.figure(figsize=(10,5))

sns.histplot(
    df["Return"].dropna(),
    kde=True
)

plt.title("BTC Return Distribution")

plt.show()

# RSI Distribution
if "RSI_14" in df.columns:

    plt.figure(figsize=(10,5))

    sns.histplot(
        df["RSI_14"].dropna(),
        kde=True
    )

    plt.title("RSI Distribution")

    plt.show()

# FearGreed Distribution
if "FearGreed" in df.columns:

    plt.figure(figsize=(10,5))

    sns.histplot(
        df["FearGreed"].dropna(),
        kde=True
    )

    plt.title("Fear & Greed Distribution")

    plt.show()

# ==========================================
# STATIONARITY TEST
# ==========================================

print("\n================ ADF TEST ================")

def adf_test(series, name="Variable"):

    result = adfuller(series.dropna())

    print(f"\nADF Test: {name}")
    print(f"ADF Statistic : {result[0]}")
    print(f"p-value       : {result[1]}")

    if result[1] < 0.05:
        print("=> Stationary")
    else:
        print("=> Non-stationary")

# Run tests
adf_test(df["Return"], "BTC Return")

if "FearGreed" in df.columns:
    adf_test(df["FearGreed"], "Fear & Greed")

if "DXY" in df.columns:
    adf_test(df["DXY"], "DXY")

# ==========================================
# SAVE FINAL DATASET
# ==========================================

df.to_csv(
    "../data/04_Master_Dataset_EDA.csv",
    index=True
)

print("\n==========================================")
print("EDA COMPLETED SUCCESSFULLY")
print("==========================================")