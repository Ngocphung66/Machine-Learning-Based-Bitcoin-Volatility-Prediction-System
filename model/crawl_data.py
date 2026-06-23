import pandas as pd
import yfinance as yf
import requests
from fredapi import Fred
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# CONFIG
# ==========================================

START_DATE = "2018-01-01"
END_DATE = "2024-12-31"
FRED_API_KEY = '3d7e128765b8ed081f4dcb4ecca6aae6'

print("Bắt đầu thu thập dữ liệu...")


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def fix_yfinance(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df


def safe_request_json(url, timeout=30):
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


# ==========================================
# 1. BTC OHLCV
# ==========================================

print("1. Đang tải BTC OHLCV...")

btc = yf.download(
    "BTC-USD",
    start=START_DATE,
    end=END_DATE,
    progress=False
)

btc = fix_yfinance(btc)
btc = btc[["Open", "High", "Low", "Close", "Volume"]]

print("BTC OHLCV:", btc.shape)


# ==========================================
# 2. TECHNICAL INDICATORS
# ==========================================

print("2. Đang tạo 11 technical indicators...")

tech_df = btc.copy()

# ---------- Nhóm 1: Xu hướng ----------
# 1. SMA
tech_df["SMA_14"] = tech_df["Close"].rolling(window=14).mean()

# 2. EMA
tech_df["EMA_14"] = tech_df["Close"].ewm(span=14, adjust=False).mean()

# 3. Ichimoku
high_9 = tech_df["High"].rolling(window=9).max()
low_9 = tech_df["Low"].rolling(window=9).min()
tech_df["Ichimoku_Tenkan"] = (high_9 + low_9) / 2

high_26 = tech_df["High"].rolling(window=26).max()
low_26 = tech_df["Low"].rolling(window=26).min()
tech_df["Ichimoku_Kijun"] = (high_26 + low_26) / 2

tech_df["Ichimoku_Senkou_A"] = (
    (tech_df["Ichimoku_Tenkan"] + tech_df["Ichimoku_Kijun"]) / 2
).shift(26)

high_52 = tech_df["High"].rolling(window=52).max()
low_52 = tech_df["Low"].rolling(window=52).min()
tech_df["Ichimoku_Senkou_B"] = ((high_52 + low_52) / 2).shift(26)

# 4. ADX
period = 14

tech_df["H-L"] = tech_df["High"] - tech_df["Low"]
tech_df["H-PC"] = abs(tech_df["High"] - tech_df["Close"].shift(1))
tech_df["L-PC"] = abs(tech_df["Low"] - tech_df["Close"].shift(1))
tech_df["TR"] = tech_df[["H-L", "H-PC", "L-PC"]].max(axis=1)

tech_df["+DM"] = np.where(
    (tech_df["High"] - tech_df["High"].shift(1)) >
    (tech_df["Low"].shift(1) - tech_df["Low"]),
    np.maximum(tech_df["High"] - tech_df["High"].shift(1), 0),
    0
)

tech_df["-DM"] = np.where(
    (tech_df["Low"].shift(1) - tech_df["Low"]) >
    (tech_df["High"] - tech_df["High"].shift(1)),
    np.maximum(tech_df["Low"].shift(1) - tech_df["Low"], 0),
    0
)

tech_df["TR_14"] = tech_df["TR"].rolling(window=period).sum()
tech_df["+DM_14"] = tech_df["+DM"].rolling(window=period).sum()
tech_df["-DM_14"] = tech_df["-DM"].rolling(window=period).sum()

tech_df["+DI_14"] = 100 * (tech_df["+DM_14"] / tech_df["TR_14"])
tech_df["-DI_14"] = 100 * (tech_df["-DM_14"] / tech_df["TR_14"])

tech_df["DX"] = (
    abs(tech_df["+DI_14"] - tech_df["-DI_14"]) /
    (tech_df["+DI_14"] + tech_df["-DI_14"])
) * 100

tech_df["ADX_14"] = tech_df["DX"].rolling(window=period).mean()


# ---------- Nhóm 2: Động lượng ----------
# 5. RSI
delta = tech_df["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()

rs = avg_gain / avg_loss
tech_df["RSI_14"] = 100 - (100 / (1 + rs))

# 6. MACD
ema_12 = tech_df["Close"].ewm(span=12, adjust=False).mean()
ema_26 = tech_df["Close"].ewm(span=26, adjust=False).mean()

tech_df["MACD"] = ema_12 - ema_26
tech_df["MACD_Signal"] = tech_df["MACD"].ewm(span=9, adjust=False).mean()
tech_df["MACD_Hist"] = tech_df["MACD"] - tech_df["MACD_Signal"]

# 7. Williams %R
highest_high_14 = tech_df["High"].rolling(window=14).max()
lowest_low_14 = tech_df["Low"].rolling(window=14).min()

tech_df["Williams_R"] = (
    (highest_high_14 - tech_df["Close"]) /
    (highest_high_14 - lowest_low_14)
) * -100

# 8. KDJ
low_min = tech_df["Low"].rolling(window=9).min()
high_max = tech_df["High"].rolling(window=9).max()

tech_df["RSV"] = (
    (tech_df["Close"] - low_min) /
    (high_max - low_min)
) * 100

tech_df["K"] = tech_df["RSV"].ewm(com=2).mean()
tech_df["D"] = tech_df["K"].ewm(com=2).mean()
tech_df["J"] = 3 * tech_df["K"] - 2 * tech_df["D"]

# 9. SQZ Momentum
bb_mid = tech_df["Close"].rolling(window=20).mean()
bb_std = tech_df["Close"].rolling(window=20).std()

tech_df["SQZ_Momentum"] = tech_df["Close"] - bb_mid


# ---------- Nhóm 3: Biến động ----------
# 10. Bollinger Bands
tech_df["BB_Middle"] = bb_mid
tech_df["BB_Upper"] = bb_mid + 2 * bb_std
tech_df["BB_Lower"] = bb_mid - 2 * bb_std
tech_df["BB_Width"] = (
    tech_df["BB_Upper"] - tech_df["BB_Lower"]
) / tech_df["BB_Middle"]

# 11. ATR
tech_df["ATR_14"] = tech_df["TR"].rolling(window=14).mean()


# Xóa cột trung gian không dùng trực tiếp
drop_cols = [
    "H-L", "H-PC", "L-PC", "TR",
    "+DM", "-DM", "TR_14", "+DM_14", "-DM_14",
    "+DI_14", "-DI_14", "DX", "RSV"
]

tech_df.drop(columns=drop_cols, inplace=True)

print("Technical dataset:", tech_df.shape)


# ==========================================
# 3. FUNDAMENTAL VARIABLES
# ==========================================

print("3. Đang tải Fundamental variables...")

fundamental_df = pd.DataFrame(index=btc.index)


# ---------- 3.1 Fear & Greed ----------
print("3.1 Fear & Greed Index...")

try:
    url_fng = "https://api.alternative.me/fng/?limit=0"
    r_fng = safe_request_json(url_fng)

    fng_df = pd.DataFrame(r_fng["data"])
    fng_df["Date"] = pd.to_datetime(
        fng_df["timestamp"].astype(int),
        unit="s"
    ).dt.normalize()

    fng_df.set_index("Date", inplace=True)
    fng_df = fng_df[["value"]].astype(float)
    fng_df.columns = ["FearGreed"]

    fundamental_df = fundamental_df.join(fng_df, how="left")

except Exception as e:
    print(f"Lỗi Fear & Greed: {e}")


# ---------- 3.2 SP500 ----------
print("3.2 S&P 500...")

try:
    sp500 = yf.download(
        "^GSPC",
        start=START_DATE,
        end=END_DATE,
        progress=False
    )

    sp500 = fix_yfinance(sp500)
    fundamental_df["SP500"] = sp500["Close"]

except Exception as e:
    print(f"Lỗi SP500: {e}")


# ---------- 3.3 DXY ----------
print("3.3 DXY...")

try:
    dxy = yf.download(
        "DX-Y.NYB",
        start=START_DATE,
        end=END_DATE,
        progress=False
    )

    dxy = fix_yfinance(dxy)
    fundamental_df["DXY"] = dxy["Close"]

except Exception as e:
    print(f"Lỗi DXY: {e}")


# ---------- 3.4 FED & CPI ----------
print("3.4 FED Rate & CPI...")

fundamental_df["FED"] = np.nan
fundamental_df["CPI"] = np.nan

try:
    fred = Fred(api_key=FRED_API_KEY)

    fed = fred.get_series(
        "FEDFUNDS",
        observation_start=START_DATE,
        observation_end=END_DATE
    )

    cpi = fred.get_series(
        "CPIAUCSL",
        observation_start=START_DATE,
        observation_end=END_DATE
    )

    fed.index = pd.to_datetime(fed.index).normalize()
    cpi.index = pd.to_datetime(cpi.index).normalize()

    fundamental_df["FED"] = fed.reindex(fundamental_df.index)
    fundamental_df["CPI"] = cpi.reindex(fundamental_df.index)

except Exception as e:
    print(f"Lỗi FRED API: {e}")

print("Fundamental dataset:", fundamental_df.shape)


# ==========================================
# 4. BUILD MASTER DATASET
# ==========================================

print("4. Gộp dữ liệu theo research stage...")

df = tech_df.join(fundamental_df, how="left")

print(f"Kích thước trước xử lý missing: {df.shape}")

# Chỉ forward-fill, không bfill để tránh leakage
df.ffill(inplace=True)

# Drop những dòng đầu bị thiếu do rolling indicators
df.dropna(inplace=True)

df = df[(df.index >= START_DATE) & (df.index <= END_DATE)]

print(f"Kích thước sau xử lý missing: {df.shape}")


# ==========================================
# 5. EXPORT FILES
# ==========================================

print("5. Xuất dữ liệu...")

btc.to_csv("01_BTC_OHLCV.csv", index_label="Date")
tech_df.to_csv("02_Technical_Variables.csv", index_label="Date")
fundamental_df.to_csv("03_Fundamental_Variables.csv", index_label="Date")
df.to_csv("04_Master_Dataset.csv", index_label="Date")

print("HOÀN TẤT.")
print("Đã xuất file:")
print("01_BTC_OHLCV.csv")
print("02_Technical_Variables.csv")
print("03_Fundamental_Variables.csv")
print("04_Master_Dataset.csv")

print(df.head())
print(df.columns.tolist())
