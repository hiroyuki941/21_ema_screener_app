import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO

# -------------------------------------------------
# 21EMA Screener  –  Streamlit Web App (v0.4)
# -------------------------------------------------
# • NASDAQ‑100 / Russell2000: Wikipedia & NasdaqTrader のみ使用 (404 URL 完全撤廃)
# • DataFrame 代入時の ValueError を防ぐため assign() に書き換え
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

def fetch_wikipedia_tickers(url: str, possible_cols: list[str]) -> list[str]:
    try:
        tables = pd.read_html(url, flavor="lxml")
        for tbl in tables:
            for col in possible_cols:
                if col in tbl.columns:
                    return _clean(tbl[col])
        st.warning(f"⚠️ Wikipediaページに列が見つかりません: {url}")
        return []
    except Exception as e:
        st.warning(f"⚠️ Wikipedia取得失敗: {url} → {e}")
        st.exception(e)
        return []

def fetch_russell2000() -> list[str]:
    txt_url = "https://www.nasdaqtrader.com/dynamic/SymDir/russell2000.txt"
    try:
        txt = requests.get(txt_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"}).text
        rows = [line.split("|")[0] for line in txt.splitlines() if "|" in line]
        return _clean(pd.Series(rows[1:]))
    except Exception:
        return fetch_wikipedia_tickers("https://en.wikipedia.org/wiki/Russell_2000_Index", ["Ticker", "Symbol"])

# @st.cache_data(show_spinner=False)
def load_ticker_lists() -> list[str]:
    sp500 = fetch_wikipedia_tickers("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", ["Symbol", "Ticker"])
    nasdaq100 = fetch_wikipedia_tickers("https://en.wikipedia.org/wiki/NASDAQ-100", ["Ticker", "Symbol"])
    russell2000 = fetch_russell2000()

    tickers = sorted(set(sp500 + nasdaq100 + russell2000))
    if not tickers:
        st.error("インデックスのティッカー取得に失敗しました。ネットワーク環境を確認してください。")
    return tickers

# -------------------------------------------------
# テクニカル指標計算
# -------------------------------------------------

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["21EMA"] = df["Close"].ewm(span=21).mean()

    prev_close = df["Close"].shift()
    tr = np.maximum(df["High"] - df["Low"], np.maximum(abs(df["High"] - prev_close), abs(df["Low"] - prev_close)))
    df["ATR_21"] = tr.rolling(window=21, min_periods=1).mean()
    df = df.assign(ATR_pct=(df["ATR_21"] / df["Close"]).mul(100))

    df["10SMA"] = df["Close"].rolling(window=10).mean()
    df["50SMA"] = df["Close"].rolling(window=50).mean()
    df["150SMA"] = df["Close"].rolling(window=150).mean()
    df["200SMA"] = df["Close"].rolling(window=200).mean()
    df["Vol50Avg"] = df["Volume"].rolling(window=50).mean()
    return df

@st.cache_data(show_spinner=False)
def get_data(ticker: str):
    df = yf.download(ticker, period="1y", progress=False, auto_adjust=True, threads=False)
    if df.empty or len(df) < 200:
        return None
    return calculate_technical_indicators(df)

# -------------------------------------------------
# UI
# -------------------------------------------------
ema_min, ema_max = st.slider("21EMA乖離率 (%)", -10.0, 10.0, (-5.0, 5.0), 0.1)
atr_min, atr_max = st.slider("21日ATR%", 0.0, 15.0, (3.0, 5.0), 0.1)
vol_threshold = st.number_input("出来高 (50日平均, 株数)", value=100000, step=10000, format="%d")
run_button = st.button("🔍 スクリーニング実行")

# -------------------------------------------------
# メイン処理
# -------------------------------------------------
if run_button:
    tickers = load_ticker_lists()
    if not tickers:
        st.stop()

    st.info(f"対象ティッカー数: {len(tickers)} 件 – データ取得には数分かかる場合があります。")
    prog = st.progress(0.0)

    results = []
    for i, tic in enumerate(tickers):
        prog.progress((i + 1) / len(tickers), text=f"{tic} 取得中…")
        df = get_data(tic)
        if df is None:
            continue
        row = df.iloc[-1]

        gap_pct = (row["Close"] - row["21EMA"]) / row["21EMA"] * 100
        conds = [
            ema_min <= gap_pct <= ema_max,
            atr_min <= row["ATR_pct"] <= atr_max,
            row["50SMA"] > row["150SMA"] > row["200SMA"],
            row["10SMA"] > row["21EMA"] > row["50SMA"],
            row["Vol50Avg"] >= vol_threshold,
        ]
        if not all(conds):
            continue

        results.append({
            "Ticker": tic,
            "Close": row["Close"],
            "EMA Gap%": round(gap_pct, 2),
            "ATR%": round(row["ATR_pct"], 2),
            "Vol50Avg": int(row["Vol50Avg"]),
        })

    prog.empty()
    st.success(f"スクリーニング完了：{len(results)} 件ヒット")

    if results:
        out = pd.DataFrame(results).sort_values("ATR%", ascending=False)
        st.dataframe(out, use_container_width=True)
        st.download_button("CSVをダウンロード", out.to_csv(index=False).encode("utf-8"), "screened.csv")
        st.code(",".join(out["Ticker"].tolist()), language="text")
    else:
        st.warning("条件に一致する銘柄がありませんでした。")
