import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO

# -------------------------------------------------
# 21EMA Screener  –  Streamlit Web App (v0.4.2)
# -------------------------------------------------
# • KeyError("Close") の対策: yfinance の列名揺れを吸収し
#   - MultiIndex を解除
#   - 'Adj Close' を 'Close' に昇格
#   - 必須列欠損なら None を返す
# -------------------------------------------------

st.set_page_config(page_title="21EMA Screener", layout="wide")
st.title("📈 21EMA 成長株スクリーナー")
st.caption("条件：21EMA乖離 ±5%、ATR% 3〜5%、出来高50日平均 > 10万株、MA順序チェック")

# -------------------------------------------------
# ティッカー取得ユーティリティ
# -------------------------------------------------

def _clean(series: pd.Series) -> list[str]:
    return (
        series.astype(str)
        .str.upper()
        .str.replace(".", "-", regex=False)
        .str.strip()
        .dropna()
        .tolist()
    )

def fetch_wikipedia_tickers(url: str, cols: list[str]) -> list[str]:
    try:
        tables = pd.read_html(url, flavor="lxml")
        for tbl in tables:
            for col in cols:
                if col in tbl.columns:
                    return _clean(tbl[col])
        return []
    except Exception:
        return []

def fetch_russell2000() -> list[str]:
    try:
        txt = requests.get("https://www.nasdaqtrader.com/dynamic/SymDir/russell2000.txt", timeout=10, headers={"User-Agent": "Mozilla/5.0"}).text
        rows = [r.split("|")[0] for r in txt.splitlines() if "|" in r]
        return _clean(pd.Series(rows[1:]))
    except Exception:
        return fetch_wikipedia_tickers("https://en.wikipedia.org/wiki/Russell_2000_Index", ["Ticker", "Symbol"])

@st.cache_data(show_spinner=False)
def load_ticker_lists() -> list[str]:
    sp = fetch_wikipedia_tickers("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", ["Symbol", "Ticker"])
    ndq = fetch_wikipedia_tickers("https://en.wikipedia.org/wiki/NASDAQ-100", ["Ticker", "Symbol"])
    rus = fetch_russell2000()
    return sorted(set(sp + ndq + rus))

# -------------------------------------------------
# テクニカル指標計算
# -------------------------------------------------

def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["21EMA"] = df["Close"].ewm(span=21).mean()
    prev = df["Close"].shift()
    tr = np.maximum(df["High"] - df["Low"], np.maximum(abs(df["High"] - prev), abs(df["Low"] - prev)))
    df["ATR_21"] = tr.rolling(21, min_periods=1).mean()
    df["ATR_pct"] = df["ATR_21"] / df["Close"] * 100
    df["10SMA"] = df["Close"].rolling(10).mean()
    df["50SMA"] = df["Close"].rolling(50).mean()
    df["150SMA"] = df["Close"].rolling(150).mean()
    df["200SMA"] = df["Close"].rolling(200).mean()
    df["Vol50Avg"] = df["Volume"].rolling(50).mean()
    return df

@st.cache_data(show_spinner=False)
def get_data(tic: str):
    df = yf.download(tic, period="1y", auto_adjust=False, progress=False, threads=False)
    if df.empty or len(df) < 200:
        return None
    # MultiIndex -> single
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)
    # Elevate Adj Close if Close missing
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    required = {"Close", "High", "Low", "Volume"}
    if not required.issubset(df.columns):
        return None
    df = df.dropna(subset=list(required))
    return calc_indicators(df)

# -------------------------------------------------
# UI & メイン
# -------------------------------------------------
ema_min, ema_max = st.slider("21EMA乖離率 (%)", -10.0, 10.0, (-5.0, 5.0))
atr_min, atr_max = st.slider("21日ATR%", 0.0, 15.0, (3.0, 5.0))
vol_thr = st.number_input("出来高(50日平均 株数)", 100000, step=10000)
if st.button("🔍 スクリーニング実行"):
    tics = load_ticker_lists()
    if not tics:
        st.stop()
    st.info(f"{len(tics)}銘柄をチェック中…")
    bar = st.progress(0.0)
    res = []
    for i, tic in enumerate(tics):
        bar.progress((i+1)/len(tics), text=tic)
        d = get_data(tic)
        if d is None:
            continue
        r = d.iloc[-1]
        gap = (r["Close"] - r["21EMA"]) / r["21EMA"] * 100
        if not (ema_min <= gap <= ema_max):
            continue
        if not (atr_min <= r["ATR_pct"] <= atr_max):
            continue
        if not (r["50SMA"] > r["150SMA"] > r["200SMA"]):
            continue
        if not (r["10SMA"] > r["21EMA"] > r["50SMA"]):
            continue
        if r["Vol50Avg"] < vol_thr:
            continue
        res.append({
            "Ticker": tic,
            "Close": r["Close"],
            "EMA Gap%": round(gap,2),
            "ATR%": round(r["ATR_pct"],2),
            "Vol50Avg": int(r["Vol50Avg"])
        })
    bar.empty()
    st.success(f"{len(res)}件ヒット")
    if res:
        df_out = pd.DataFrame(res).sort_values("ATR%", ascending=False)
        st.dataframe(df_out, use_container_width=True)
        st.download_button("CSVダウンロード", df_out.to_csv(index=False).encode(), "screened.csv")
        st.code(",".join(df_out.Ticker), language="text")
    else:
        st.warning("該当銘柄なし")
