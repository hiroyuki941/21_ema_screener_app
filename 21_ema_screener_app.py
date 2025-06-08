import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from io import StringIO

# -------------------------------------------------
# 21EMA Screener  –  Streamlit Web App (v0.2)
# -------------------------------------------------
# * 改修点 *
#   • S&P500 のティッカー取得先を constituents.csv に変更（URL修正）
#   • requests + User-Agent 付きでダウンロード → 403/404 回避
#   • 取得失敗時は警告を出してスキップ（アプリは落とさない）
# -------------------------------------------------

st.set_page_config(page_title="21EMA Screener", layout="wide")
st.title("📈 21EMA 成長株スクリーナー")
st.markdown("条件：21EMA乖離 ±5%、ATR% 3〜5%、出来高50日平均 > 10万株、MA順序チェック")

# -------------------------------------------------
# ユーティリティ
# -------------------------------------------------

def fetch_tickers_from_url(url: str, column: str | None = None) -> list[str]:
    """指定URLからCSV/TSVをダウンロードし、ティッカーのリストを返す。"""
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        content = StringIO(resp.text)
        if column:
            df = pd.read_csv(content)
            return (
                df[column]
                .astype(str)
                .str.upper()
                .str.replace(".", "-", regex=False)
                .dropna()
                .tolist()
            )
        else:
            # header=None の場合は 1 列目をティッカーとみなす
            return (
                pd.read_csv(content, header=None)[0]
                .astype(str)
                .str.upper()
                .str.replace(".", "-", regex=False)
                .dropna()
                .tolist()
            )
    except Exception as e:
        st.warning(f"⚠️ ティッカー取得失敗: {url} → {e}")
        return []

@st.cache_data(show_spinner=False)
def load_ticker_lists() -> list[str]:
    """S&P500 / NASDAQ100 / Russell2000 のティッカーを結合して返す"""
    sp500_url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
    nasdaq100_url = "https://raw.githubusercontent.com/raphaelmoritz/nasdaq100-list/main/nasdaq100.csv"
    russell_url = "https://raw.githubusercontent.com/rohan-paul/Misc-datasets/main/russell2000.csv"

    sp500 = fetch_tickers_from_url(sp500_url, column="Symbol")
    nasdaq100 = fetch_tickers_from_url(nasdaq100_url, column="Symbol")
    russell2000 = fetch_tickers_from_url(russell_url, column="Ticker")

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
    df["ATR"] = df["High"] - df["Low"]
    df["ATR_21"] = df["ATR"].rolling(window=21).mean()
    df["ATR_pct"] = df["ATR_21"] / df["Close"] * 100
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
# UI コンポーネント
# -------------------------------------------------
ema_min, ema_max = st.slider("21EMA乖離率 (%)", -10.0, 10.0, (-5.0, 5.0), 0.1)
atr_min, atr_max = st.slider("21日ATR%", 0.0, 10.0, (3.0, 5.0), 0.1)
vol_threshold = st.number_input("出来高 (50日平均、株数)", value=100000, step=10000)
run_button = st.button("🔍 スクリーニング実行")

# -------------------------------------------------
# スクリーニング処理
# -------------------------------------------------
if run_button:
    tickers = load_ticker_lists()
    if not tickers:
        st.stop()

    st.info(f"対象ティッカー数: {len(tickers)} 件 – データ取得には数分かかる場合があります。")
    progress = st.progress(0.0, text="スクリーニング中...")

    results = []
    for i, ticker in enumerate(tickers):
        progress.progress((i + 1) / len(tickers), text=f"{ticker} 処理中…")
        df = get_data(ticker)
        if df is None:
            continue
        row = df.iloc[-1]

        gap_pct = (row["Close"] - row["21EMA"]) / row["21EMA"] * 100
        if not (ema_min <= gap_pct <= ema_max):
            continue
        if not (atr_min <= row["ATR_pct"] <= atr_max):
            continue
        if not (row["50SMA"] > row["150SMA"] > row["200SMA"]):
            continue
        if not (row["10SMA"] > row["21EMA"] > row["50SMA"]):
            continue
        if row["Vol50Avg"] < vol_threshold:
            continue

        results.append({
            "Ticker": ticker,
            "Close": row["Close"],
            "EMA Gap%": round(gap_pct, 2),
            "ATR%": round(row["ATR_pct"], 2),
            "Vol50Avg": int(row["Vol50Avg"])
        })

    progress.empty()

    st.success(f"スクリーニング完了：{len(results)} 件ヒット")
    if results:
        df_out = pd.DataFrame(results).sort_values("ATR%", ascending=False)
        st.dataframe(df_out, use_container_width=True)
        st.download_button("CSVをダウンロード", df_out.to_csv(index=False).encode("utf-8"), "screened.csv")

        ticker_list = ",".join(df_out["Ticker"].tolist())
        st.code(ticker_list, language="text")
    else:
        st.warning("条件に一致する銘柄がありませんでした。")
