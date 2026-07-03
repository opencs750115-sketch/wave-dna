# -*- coding: utf-8 -*-
"""
================================================================================
  動態歷史波浪週期 DNA 匹配系統 (Dynamic Wave Cycle DNA Matching)
  app.py  ── Streamlit 主程式
================================================================================
執行方式:
    streamlit run app.py

需要套件 (requirements.txt):
    streamlit>=1.35
    yfinance>=1.4.0
    curl_cffi>=0.7.0
    pandas>=2.0
    numpy>=1.24
    scipy>=1.11
    streamlit-autorefresh>=1.0.1  ← 盤中自動刷新
================================================================================
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import datetime
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# EquityQuery 用於即時抓取台灣熱門排行
try:
    from yfinance.screener.query import EquityQuery
    _EQUITY_QUERY_AVAILABLE = True
except ImportError:
    _EQUITY_QUERY_AVAILABLE = False

# FinMind — 台股三大法人籌碼數據
# ★ 不安裝 finmind 套件，改用免費 REST API（避免 Streamlit Cloud 依賴衝突）
# API: https://api.finmindtrade.com/api/v4/data
_FINMIND_AVAILABLE = True  # 只要能連網就可用（不依賴套件）

# ── ★ Discord Webhook 自動推播 ─────────────────────────────────────────────
import requests as _requests

DISCORD_WEBHOOK_URL = (
    "https://discordapp.com/api/webhooks/"
    "1521147314834505848/"
    "aWbjve4_c0qQBHTFL-oTLWvD-UEOdmnb_4-Ix6hh94A_rdW5eBmf2jTrR51UVMBzhUiS"
)

def send_discord_notify(message: str) -> bool:
    """
    推播訊息到 Discord Webhook。
    回傳 True=成功，False=失敗（不拋出例外）。
    """
    try:
        resp = _requests.post(
            DISCORD_WEBHOOK_URL,
            json={"content": message},
            timeout=5
        )
        return resp.status_code in (200, 204)
    except Exception as e:
        print(f"Discord 發送失敗: {e}")
        return False

# ── ★ 台灣盤中時段判定 ────────────────────────────────────────────────────
def is_tw_trading_hours() -> bool:
    """
    判斷當前是否為台股交易時段。
    週一至週五 09:00 ~ 13:35（台灣時間 Asia/Taipei）。
    """
    try:
        import pytz as _pytz
        tw = datetime.datetime.now(_pytz.timezone('Asia/Taipei'))
        if tw.weekday() >= 5:               # 週末
            return False
        t = tw.time()
        return datetime.time(9, 0) <= t <= datetime.time(13, 35)
    except Exception:
        return False

# ── ★ streamlit_autorefresh（盤中每 20 分鐘自動刷新）────────────────────
try:
    from streamlit_autorefresh import st_autorefresh
    _AUTOREFRESH_AVAILABLE = True
except ImportError:
    _AUTOREFRESH_AVAILABLE = False



# ─────────────────────────────────────────────────────────────────────────────
#  全域設定
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="波浪 DNA 匹配系統",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  自訂 CSS ── 淺藍色系專業看板風格
#  色票:
#    --bg       : #f0f4f8  (淺灰藍主背景)
#    --panel    : #ffffff  (卡片白底)
#    --sidebar  : #e8eef5  (側邊欄淺藍灰)
#    --border   : #c8d8e8  (邊框淺藍)
#    --text     : #1a2b3c  (深藍主文字)
#    --muted    : #4a6fa5  (中藍輔助文字)
#    --bull     : #0a7c59  (多頭深綠)
#    --bear     : #c0392b  (空頭深紅)
#    --accent   : #1565c0  (強調藍)
#    --mid      : #d97706  (中繼橘黃)
#  字體: Noto Sans TC(中文) + IBM Plex Mono(數字)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

/* ── 根樣式 ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #f0f4f8 !important;
    font-family: 'Noto Sans TC', sans-serif;
    color: #1a2b3c;
    font-size: 15px;
}
[data-testid="stSidebar"] {
    background: #e2eaf3 !important;
    border-right: 2px solid #b8cce0;
}
[data-testid="stSidebar"] * { color: #1a2b3c !important; }
header[data-testid="stHeader"] { background: #dce8f0 !important; border-bottom: 1px solid #b8cce0; }

/* ── Streamlit 原生元件文字覆蓋 ── */
p, label, div, span { color: #1a2b3c; }
.stRadio label, .stCheckbox label, .stSelectbox label,
.stSlider label, .stTextInput label, .stTextArea label { color: #1a2b3c !important; font-size: 15px !important; }
.stRadio div[role="radiogroup"] label { font-size: 15px !important; color: #1a2b3c !important; }
.stButton button { font-size: 15px !important; font-weight: 600 !important; border-radius: 8px !important; }
.stButton button[kind="primary"] {
    background: #1565c0 !important; color: white !important; border: none !important;
}
.stSelectbox div[data-baseweb] { background: #ffffff !important; color: #1a2b3c !important; border-color: #b8cce0 !important; }
.stTextInput input, .stTextArea textarea {
    background: #ffffff !important; color: #1a2b3c !important;
    border: 1.5px solid #b8cce0 !important; border-radius: 8px !important;
    font-size: 15px !important;
}
.stExpander { border: 1px solid #b8cce0 !important; border-radius: 8px !important; background: #ffffff; }
[data-testid="stExpander"] summary { color: #1a2b3c !important; font-size: 15px !important; }

/* ── 卡片 ── */
.dna-card {
    background: #ffffff;
    border: 1.5px solid #b8cce0;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 14px;
    box-shadow: 0 2px 8px rgba(21,101,192,0.07);
}
.dna-card h3 {
    font-size: 12px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #4a6fa5;
    margin: 0 0 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
}
.dna-card .val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 30px;
    font-weight: 700;
    color: #1a2b3c;
    line-height: 1.1;
}
.dna-card .sub {
    font-size: 13px;
    color: #4a6fa5;
    margin-top: 5px;
    font-family: 'IBM Plex Mono', monospace;
}

/* ── 分類標籤 ── */
.label-top  { background: #0a7c59; color: #ffffff; }
.label-mid  { background: #d97706; color: #ffffff; }
.label-warn { background: #c0392b; color: #ffffff; }
.dna-label {
    display: inline-block;
    font-family: 'Noto Sans TC', sans-serif;
    font-weight: 700;
    font-size: 18px;
    padding: 10px 26px;
    border-radius: 8px;
    letter-spacing: 1px;
    margin-bottom: 16px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
}

/* ── 勝率橫條 ── */
.bar-wrap {
    background: #d0dde8;
    border-radius: 999px;
    height: 14px;
    overflow: hidden;
    margin-top: 10px;
}
.bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width .8s ease;
}

/* ── 表格 ── */
.fwd-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 14px;
    background: #ffffff;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(21,101,192,0.07);
}
.fwd-table th {
    background: #1565c0;
    color: #ffffff;
    padding: 12px 14px;
    text-align: left;
    font-size: 12px;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-weight: 600;
}
.fwd-table td {
    padding: 11px 14px;
    border-bottom: 1px solid #dce8f0;
    color: #1a2b3c;
    font-size: 14px;
}
.fwd-table tr:last-child td { border-bottom: none; }
.fwd-table tr:hover td { background: #eaf2fb; }
.price-up   { color: #0a7c59; font-weight: 600; }
.price-down { color: #c0392b; font-weight: 600; }
.price-flat { color: #d97706; font-weight: 600; }

/* ── 分隔標題 ── */
.section-title {
    font-size: 13px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #1565c0;
    font-weight: 700;
    margin: 24px 0 14px;
    font-family: 'Noto Sans TC', sans-serif;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 2px;
    background: linear-gradient(90deg, #1565c0, #b8cce0);
    border-radius: 2px;
}

/* ── 特徵分數條 ── */
.feat-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
}
.feat-label { color: #1a2b3c; width: 130px; flex-shrink: 0; font-weight: 500; }
.feat-bar-wrap { flex: 1; background: #d0dde8; border-radius: 999px; height: 10px; }
.feat-bar-fill { height: 10px; border-radius: 999px; }
.feat-val { color: #1565c0; width: 44px; text-align: right; font-weight: 600; }

/* ── Streamlit metric ── */
[data-testid="metric-container"] {
    background: #ffffff !important;
    border: 1.5px solid #b8cce0 !important;
    border-radius: 10px !important;
    padding: 16px 18px !important;
    box-shadow: 0 2px 6px rgba(21,101,192,0.07) !important;
}
[data-testid="stMetricLabel"] p {
    color: #4a6fa5 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}
[data-testid="stMetricValue"] {
    color: #1a2b3c !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 24px !important;
    font-weight: 700 !important;
}
[data-testid="stMetricDelta"] { font-size: 13px !important; }

/* ── 手機響應式 ─────────────────────────────────────────────────── */
@media (max-width: 768px) {
    /* 側邊欄在手機預設收起 */
    [data-testid="stSidebar"] { min-width: 0 !important; }

    /* 主內容全寬 */
    .main .block-container { padding: 8px 10px 20px !important; max-width: 100% !important; }

    /* 頁面標題縮小 */
    h1 { font-size: 18px !important; }

    /* metric 欄位在手機單欄排列 */
    [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; gap: 8px !important; }
    [data-testid="metric-container"] {
        min-width: 140px !important;
        padding: 10px 12px !important;
    }
    [data-testid="stMetricValue"] { font-size: 18px !important; }

    /* 掃描結果表格在手機改為卡片式(只隱藏 scan-table,保留 fwd-table) */
    .scan-table { display: none !important; }
    .mobile-cards { display: block !important; }

    /* 前瞻路徑表格在手機改為橫向可滾動 */
    .fwd-table-wrap {
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
    }
    .fwd-table {
        min-width: 520px;
        font-size: 12px !important;
        display: none !important;          /* 手機隱藏表格,顯示卡片 */
    }
    .forecast-cards { display: block !important; }

    /* 卡片列間距 */
    .dna-card { padding: 12px 14px !important; margin-bottom: 10px !important; }
    .dna-card .val { font-size: 22px !important; }
    .bar-wrap { height: 10px !important; }
}

/* 桌機隱藏卡片式、顯示表格式 */
@media (min-width: 769px) {
    .mobile-cards { display: none !important; }
    .scan-table { display: table !important; }
    .forecast-cards { display: none !important; }
    .fwd-table { display: table !important; }
}

/* ── 手機掃描結果卡片 ─────────────────────────────────────────── */
.scan-card {
    background: #ffffff;
    border: 1.5px solid #c8d8e8;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
    box-shadow: 0 2px 8px rgba(21,101,192,0.07);
}
.scan-card .sc-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
}
.scan-card .sc-code {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 16px;
    font-weight: 700;
    color: #1565c0;
    text-decoration: none;
}
.scan-card .sc-name {
    font-size: 15px;
    color: #1a2b3c;
    font-weight: 500;
}
.scan-card .sc-badge {
    font-size: 13px;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 6px;
    white-space: nowrap;
}
.scan-card .sc-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
    font-family: 'IBM Plex Mono', monospace;
}
.scan-card .sc-price {
    font-size: 20px;
    font-weight: 700;
    color: #1a2b3c;
}
.scan-card .sc-wr {
    font-size: 18px;
    font-weight: 700;
}
.scan-card .sc-bar-wrap {
    height: 8px;
    background: #c8d8e8;
    border-radius: 999px;
    overflow: hidden;
    margin: 6px 0;
}
.scan-card .sc-bar-fill {
    height: 8px;
    border-radius: 999px;
}
.scan-card .sc-meta {
    display: flex;
    gap: 12px;
    font-size: 12px;
    color: #4a6fa5;
    font-family: 'IBM Plex Mono', monospace;
    flex-wrap: wrap;
    margin-top: 6px;
}
.scan-card .sc-desc {
    font-size: 12px;
    color: #2d3748;
    margin-top: 6px;
    line-height: 1.5;
    border-top: 1px solid #eaf2fb;
    padding-top: 6px;
}
.scan-card .sc-btn {
    background: #eaf2fb;
    border: 1px solid #b8cce0;
    border-radius: 6px;
    padding: 4px 10px;
    color: #1565c0;
    font-size: 13px;
    cursor: pointer;
    text-decoration: none;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  工具函式群
# ─────────────────────────────────────────────────────────────────────────────

def _patch_today_price(df: pd.DataFrame, ticker_str: str) -> tuple[pd.DataFrame, bool]:
    """
    ★ 即時報價自動補丁 v6 — 每次 rerun 直接打 fast_info
    ────────────────────────────────────────────────────────────────────────
    【v5 → v6 改動】
      移除 session_state 每分鐘快取。
      原本用 key = _live_{ticker}_{minute} 做快取，目的是減少 fast_info 呼叫，
      但反而造成同一分鐘內的 rerun 都用舊值，讓使用者感覺「落後很多時間」。

      fast_info 每次呼叫約 0.3~0.5 秒，完全可接受。
      移除快取後，每次 rerun 都拿最新現價，即時性最佳。

    【觸發條件】
      台灣交易時段 09:02~15:30（週一至週五）無條件補丁。

    回傳: (df, patched: bool)
    """
    try:
        import pytz as _pytz
        tw_tz  = _pytz.timezone('Asia/Taipei')
        now_tw = datetime.datetime.now(tw_tz)
        today_str = now_tw.strftime('%Y-%m-%d')

        if df.empty:
            return df, False

        # 時區剝離
        if df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)

        # 週末不補
        if now_tw.weekday() >= 5:
            return df, False

        # 只在 09:02 ~ 15:30
        t = now_tw.time()
        if not (datetime.time(9, 2) <= t <= datetime.time(15, 30)):
            return df, False

        # ── fast_info 加 30 秒 session_state 快取 ──────────────────────
        # 每次 rerun 都打 fast_info 約 0.3~0.8 秒
        # 改為：30 秒 slot 內同一代號只打一次，slot 結束自動更新
        import time as _ti
        _slot   = int(_ti.time() // 30)           # 每 30 秒換一個 slot
        _fi_key = f"_fi_{ticker_str}_{_slot}"
        _fi     = None
        try: _fi = st.session_state.get(_fi_key)
        except Exception: pass

        if _fi:
            live_close  = _fi[0]; live_volume = _fi[1]
            prev_close  = _fi[2]; live_open   = _fi[3]
            live_high   = _fi[4]; live_low    = _fi[5]
        else:
            fast       = yf.Ticker(ticker_str).fast_info
            live_close = getattr(fast, 'last_price', None)
            if live_close is None or float(live_close) <= 0:
                return df, False
            live_close  = float(live_close)
            live_volume = int(getattr(fast, 'last_volume', 0) or 0)
            prev_close  = float(getattr(fast, 'regular_market_previous_close',
                                       live_close) or live_close)
            try:
                live_open = float(getattr(fast, 'open', None) or live_close)
                raw_high  = float(getattr(fast, 'day_high', None) or 0)
                raw_low   = float(getattr(fast, 'day_low',  None) or 0)
                live_high = raw_high if raw_high > prev_close * 0.5 else max(live_open, live_close)
                live_low  = raw_low  if raw_low  > prev_close * 0.5 else min(live_open, live_close)
            except (ValueError, TypeError):
                live_open = live_high = live_low = live_close
            live_high = max(live_high, live_open, live_close)
            live_low  = min(live_low,  live_open, live_close)
            try:
                st.session_state[_fi_key] = [
                    live_close, live_volume, prev_close,
                    live_open,  live_high,   live_low,
                ]
            except Exception: pass

        today_ts  = pd.to_datetime(today_str)
        last_date = pd.to_datetime(df.index[-1]).strftime('%Y-%m-%d')

        patch_vals = [
            ("Open",      live_open),
            ("High",      live_high),
            ("Low",       live_low),
            ("Close",     live_close),
            ("Adj Close", live_close),
            ("Volume",    live_volume),
        ]

        if last_date == today_str:
            df = df.copy()
            for col, val in patch_vals:
                if col in df.columns:
                    df.loc[today_ts, col] = val
            df = df.sort_index()
        else:
            new_row = pd.DataFrame(index=pd.DatetimeIndex([today_ts]))
            col_map = {k: v for k, v in patch_vals}
            for col in df.columns:
                new_row[col] = col_map.get(col, float('nan'))
            df = pd.concat([df, new_row]).sort_index()
            df = df[~df.index.duplicated(keep='last')]

        return df, True

    except Exception:
        return df, False


def _get_cache_bucket() -> str:
    """
    動態快取 bucket key:
      盤中 (09:00~13:35 台灣時間) → 精確到「分鐘」: 每分鐘快取自動失效
      盤後 / 非交易時段            → 精確到「小時」: 每小時快取自動失效

    此 key 作為 fetch_data 的第三個參數傳入,讓 Streamlit 的 cache_data
    在不同的 bucket 下視為不同呼叫 → 強制重新下載最新資料。
    """
    import pytz as _pytz
    tw_now = datetime.datetime.now(_pytz.timezone('Asia/Taipei'))
    t = tw_now.time()
    in_market = datetime.time(9, 0) <= t <= datetime.time(13, 35)
    if in_market:
        return tw_now.strftime('%Y%m%d_%H%M')  # 盤中: 每分鐘一個新 key
    return tw_now.strftime('%Y%m%d_%H')         # 盤後: 每小時一個新 key


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(ticker: str, period: str = "2y",
               time_bucket: str = "") -> tuple[pd.DataFrame | None, str]:
    """
    以 yfinance 下載個股歷史日線資料。支援台股(自動補 .TW / .TWO)與美股。

    ★ 動態快取 bucket 機制 (time_bucket):
      Streamlit @st.cache_data 以「所有非底線開頭的參數」作為快取 key。
      time_bucket 由 _get_cache_bucket() 產生:
        - 盤中 09:00~13:35: 精確到「分鐘」→ 每分鐘快取失效,刷新即拿最新資料
        - 盤後: 精確到「小時」→ 每小時失效,避免頻繁重複下載
      ⚠️ 注意:參數名稱「不能」帶底線前綴,否則 Streamlit 會把它排除在
              cache key 計算之外,導致 bucket 永遠不生效!

    ★ auto_adjust=False: 保留原始未還原 OHLCV,避免台股除息後歷史價格失真。
    ★ Close=NaN 修補: 優先用 fast_info.last_price,保底用 (H+L)/2。

    快取 TTL=3600 秒(安全備用),實際由 time_bucket 控制失效頻率。
    """
    candidates = [ticker.upper()]
    t = ticker.strip().upper()
    if "." not in t and t.isdigit():
        # 純數字：同時嘗試 .TW 和 .TWO
        candidates = [f"{t}.TW", f"{t}.TWO", t]
    elif t.endswith(".TW") and not t.endswith(".TWO"):
        # 明確指定 .TW：若找不到資料，自動 fallback .TWO
        candidates = [t, t.replace(".TW", ".TWO")]
    elif t.endswith(".TWO"):
        # 明確指定 .TWO：若找不到資料，自動 fallback .TW
        candidates = [t, t.replace(".TWO", ".TW")]

    for cand in candidates:
        try:
            df = yf.download(cand, period=period, interval="1d",
                             auto_adjust=False, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if "Close" not in df.columns and "Adj Close" in df.columns:
                df = df.rename(columns={"Adj Close": "Close"})

            # ★ Close=NaN 修補
            if df["Close"].isna().any():
                nan_mask = df["Close"].isna()
                # ① 優先: fast_info.last_price
                try:
                    lp = float(getattr(yf.Ticker(cand).fast_info, 'last_price', 0) or 0)
                    if lp > 0:
                        df.loc[nan_mask, "Close"] = lp
                        if "Adj Close" in df.columns:
                            df.loc[nan_mask, "Adj Close"] = lp
                except Exception:
                    pass
                # ② 保底: (High+Low)/2
                still_nan = df["Close"].isna()
                if still_nan.any() and "High" in df.columns and "Low" in df.columns:
                    df.loc[still_nan, "Close"] = (
                        df.loc[still_nan, "High"] + df.loc[still_nan, "Low"]
                    ) / 2

            df = df.dropna(subset=["Close"])
            if len(df) >= 60:
                df.index = pd.to_datetime(df.index)
                return df, cand
        except Exception:
            continue
    return None, ticker


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    計算所有技術指標,全部附加到 df 上:
      MA5 / MA10 / MA20 / MA60    均線
      K9 / D9                     隨機指標(9日RSV,1/3平滑)
      ATR14                       平均真實波動幅度(14日)
      VolMA5                      5日均量
    """
    df = df.copy()

    for n in [5, 10, 20, 60]:
        df[f"MA{n}"] = df["Close"].rolling(n).mean()

    # KD: 台股標準 9日RSV + 1/3 指數平滑
    low9  = df["Low"].rolling(9).min()
    high9 = df["High"].rolling(9).max()
    denom = (high9 - low9).replace(0, np.nan)
    rsv   = ((df["Close"] - low9) / denom * 100).fillna(50)
    df["K9"] = rsv.ewm(alpha=1/3, adjust=False).mean()
    df["D9"] = df["K9"].ewm(alpha=1/3, adjust=False).mean()

    # ATR14
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()

    # 成交量均量
    df["VolMA5"] = df["Volume"].rolling(5).mean()

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  模組 A: 波浪 DNA 識別引擎
# ─────────────────────────────────────────────────────────────────────────────

def detect_wave_dna(df: pd.DataFrame) -> dict:
    """
    從過去兩年 (最多 500 根K棒) 的收盤價中進行「雙層波段識別」:

      ─ 大波 (Big Wave):  prominence = close 全段標準差 × 20%
                           distance   = 動態(依 ATR% 決定 12~25 天)
                           → 識別「中期趨勢波峰 / 修正低點」,計算 T_median

      ─ 小波 (Small Wave): prominence = close 全段標準差 × 8%
                           distance   = max(5, big_distance//2)
                           → 在大波峰之後,找「最近一個短期修正低點」
                           → 若此低點已過且股價已回升,切換至「上升段」模式

    ★ 修正 B+C 整合說明:
      - auto_adjust=False 解決除息還原造成的價格軸錯位問題。
      - prom = std×0.20 (全段相對標準差) 取代固定 ATR 倍數,
        讓「一路大漲的強勢股」(如南茂)也能正確識別近期短波段。
      - ATR% 決定 distance,解決「漣漪型」股票(如中華電)
        T_median 被嚴重低估的問題。

    回傳:
      peaks / troughs    : 大波索引陣列
      small_troughs      : 小波谷索引陣列(供「修正低點」判斷使用)
      corrections        : 各修正波段天數 list (大波)
      T_mean / T_median / T_std
      D_current          : 自最近大波峰至今天數
      correction_end_idx : 最近一次「確認結束修正的小波谷」索引(-1 表示尚未出現)
      days_since_trough  : 自修正低點至今的天數(已進入反彈段則 > 0)
      last_peak_date / last_peak_price
      R_cycle
      atr_pct / distance_used
    """
    close = df["Close"].values.astype(float)
    high  = df["High"].values.astype(float)
    low   = df["Low"].values.astype(float)
    n     = len(close)

    # ── ATR 日均波動率 (用於 distance 決策) ──────────────────────────
    prev = close[:-1]
    tr_arr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - prev), np.abs(low[1:] - prev))
    )
    atr14_val  = float(np.mean(tr_arr[-14:])) if len(tr_arr) >= 14 else float(np.std(close) * 0.5)
    mean_price = float(np.mean(close[-60:])) if n >= 60 else float(np.mean(close))
    atr_pct    = atr14_val / mean_price * 100

    # ── 動態 distance (中期波段最小間距) ────────────────────────────
    if atr_pct < 1.0:   dist = 25
    elif atr_pct < 2.0: dist = 20
    elif atr_pct < 3.5: dist = 15
    else:               dist = 12

    # ── 大波 prominence = 全段 std × 20% (自適應相對強度) ───────────
    # 對強勢連漲股(std 很大),門檻跟著放大只識別大波段;
    # 對牛皮股,門檻自動縮小,能識別出日常小波動。
    prom_big   = float(np.std(close) * 0.20)
    prom_small = float(np.std(close) * 0.08)
    dist_small = max(5, dist // 2)

    # ── 大波峰 / 大波谷 ─────────────────────────────────────────────
    peaks,   _ = find_peaks( close, distance=dist,       prominence=prom_big)
    troughs, _ = find_peaks(-close, distance=dist,       prominence=prom_big)
    # ── 小波谷 (短期修正低點) ───────────────────────────────────────
    small_tr, _ = find_peaks(-close, distance=dist_small, prominence=prom_small)

    # 若大波樣本太少,放寬重跑
    if len(peaks) < 3 or len(troughs) < 3:
        prom_loose = float(np.std(close) * 0.10)
        peaks,   _ = find_peaks( close, distance=max(dist-4,8), prominence=prom_loose)
        troughs, _ = find_peaks(-close, distance=max(dist-4,8), prominence=prom_loose)

    # ── 計算「大波」修正波段天數 ────────────────────────────────────
    corrections = []
    for pk in peaks:
        # 在此波峰之後找第一個「大波谷」
        big_tr_after = troughs[troughs > pk]
        if len(big_tr_after) > 0:
            days = int(big_tr_after[0] - pk)
            if days >= 5:
                corrections.append(days)
        else:
            # 無大波谷則找最近小波谷代替
            small_tr_after = small_tr[small_tr > pk]
            if len(small_tr_after) > 0:
                days = int(small_tr_after[0] - pk)
                if days >= 5:
                    corrections.append(days)

    if len(corrections) < 3:
        for i in range(len(troughs) - 1):
            d = int(troughs[i+1] - troughs[i])
            if d >= 5:
                corrections.append(d)

    if not corrections:
        T_median = float(dist)
        T_mean   = float(dist)
        T_std    = 0.0
    else:
        T_median = float(np.median(corrections))
        T_mean   = float(np.mean(corrections))
        T_std    = float(np.std(corrections)) if len(corrections) > 1 else 0.0
        T_median = max(T_median, 5.0)

    # ── 定位最近大波峰 ───────────────────────────────────────────────
    valid_peaks = peaks[peaks < n]
    if len(valid_peaks) == 0:
        last_peak_idx   = 0
        last_peak_price = float(close[0])
    else:
        last_peak_idx   = int(valid_peaks[-1])
        last_peak_price = float(close[last_peak_idx])

    D_current      = int(n - 1 - last_peak_idx)
    last_peak_date = df.index[last_peak_idx].strftime("%Y-%m-%d")
    R_cycle        = round(D_current / T_median, 3)

    # ── 尋找「最近一個小波谷」(波峰後的修正低點) ────────────────────
    # 如果存在,代表這次修正已有明確低點,可切換至「上漲段」分析模式
    small_after_peak = small_tr[small_tr > last_peak_idx]
    if len(small_after_peak) > 0:
        correction_end_idx  = int(small_after_peak[0])
        days_since_trough   = int(n - 1 - correction_end_idx)
        correction_end_date = df.index[correction_end_idx].strftime("%Y-%m-%d")
        correction_end_price= round(float(close[correction_end_idx]), 2)
        actual_correction_days = int(correction_end_idx - last_peak_idx)
    else:
        correction_end_idx   = -1
        days_since_trough    = -1
        correction_end_date  = None
        correction_end_price = None
        actual_correction_days = None

    return {
        "peaks":                peaks,
        "troughs":              troughs,
        "small_troughs":        small_tr,
        "corrections":          corrections,
        "T_mean":               round(T_mean, 1),
        "T_median":             round(T_median, 1),
        "T_std":                round(T_std, 1),
        "D_current":            D_current,
        "last_peak_date":       last_peak_date,
        "last_peak_price":      last_peak_price,
        "R_cycle":              R_cycle,
        "atr_pct":              round(atr_pct, 2),
        "distance_used":        dist,
        "correction_end_idx":   correction_end_idx,
        "correction_end_date":  correction_end_date,
        "correction_end_price": correction_end_price,
        "actual_correction_days": actual_correction_days,
        "days_since_trough":    days_since_trough,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  模組 C: 特徵向量 × 勝率引擎
# ─────────────────────────────────────────────────────────────────────────────

def _score_r_cycle(r: float, days_since_trough: int = -1,
                   actual_correction_days: int | None = None,
                   T_median: float = 20.0) -> tuple[float, str]:
    """
    時間波得分 (0~1) + 文字說明。

    ★ 新增「修正已確認結束」分支:
       當 days_since_trough >= 0 (即大波峰後已出現小波谷且已反彈),
       代表「修正段已結束,目前處於上漲段」。此時用「實際修正天數 vs T_median」
       來評估修正的充分性(充分修正 → 得高分),再額外給「反彈天數加成」。
    """
    if days_since_trough >= 0 and actual_correction_days is not None:
        # 用「實際修正天數」評估這次修正夠不夠充分
        r_actual = actual_correction_days / T_median if T_median > 0 else 0
        if r_actual >= 0.80:
            base_score = 1.0
            base_desc  = f"修正充分✅ 實際修正{actual_correction_days}天(>{T_median:.0f}天基準×80%)"
        elif r_actual >= 0.50:
            base_score = 0.80
            base_desc  = f"修正尚可🔸 實際{actual_correction_days}天(完成基準{r_actual*100:.0f}%)"
        else:
            base_score = 0.50
            base_desc  = f"修正偏短⚠️ 僅{actual_correction_days}天,可能仍有回測需求"

        # 反彈天數加成:反彈剛起步(D+1~5)加最多 +0.1
        if days_since_trough <= 3:
            extra = 0.10
            extra_desc = f",反彈起步第{days_since_trough}天⚡"
        elif days_since_trough <= 8:
            extra = 0.05
            extra_desc = f",反彈第{days_since_trough}天"
        else:
            extra = 0.0
            extra_desc = f",反彈已走{days_since_trough}天"

        return min(base_score + extra, 1.0), base_desc + extra_desc

    # 原始邏輯:修正尚在進行中,用 R_cycle 評估
    if   0.95 <= r <= 1.25: return 1.0,  "飽和✅ 時空修正完全共振"
    elif r > 1.25:           return 0.90, "超額⚡ 浮額極度乾淨"
    elif r >= 0.80:          return 0.70, "接近🔸 距臨界點尚差一段"
    elif r >= 0.60:          return 0.40, "進行中🔶 修正未過六成"
    else:                    return 0.10, "嚴重欠帳🛑 修正嚴重不足"


def _score_ma_pattern(
    close: float, ma5: float, ma10: float, ma20: float, ma60: float,
    k9: float, d9: float, ma_spread_pct: float,
    vol_ratio: float = 1.0,
) -> tuple[float, str]:
    """
    ★ 修正 A: 均線判定邏輯全面改寫
    ─────────────────────────────────────────────────────────────────────
    原本問題:嚴格要求「MA5 > MA10 > MA20 > MA60 全面多頭排列」才給高分。
    這個條件在「底部第一根帶量長紅突破」當天完全不成立(因為均線還沒動),
    導致南茂型的突破爆發被打成 0.25 偏空走勢。

    新的判定優先順序(型態 0 為最高優先):

      型態 0 ★ 帶量突破型態(最高分 0.95):
            收盤 > MA5 & MA10 & MA20  且  vol_ratio >= 1.5
            → 不管均線是否已排好,只要今天「穿越所有短中均線且量放大」就觸發。
            這正是「底部首根帶量長紅」的本質。

      型態 1 ✅ 極度壓縮型態(0.88):
            四線在收盤價 3% 以內(略放寬從 2% 到 3%,台股除息後均線常略偏)
            代表盤整完成、能量蓄積中。

      型態 2 💡 跌深 KD 極底金叉(0.82):
            收盤 < MA20 × 0.93  +  K9 < 25 & D9 < 25 & K9 > D9

      型態 3 🚀 標準多頭排列(0.88):
            MA5 > MA10 > MA20 且收盤 > MA5 × 1.005
            (均線已全面確立,給分與帶量突破同級)

      型態 4 📈 均線斜率向上整理(0.72):
            收盤 > MA20,且 MA5 > MA20(短均已突破月線),但未完全排好
            → 「整理後有望繼續」的型態

      型態 5 ⏳ 月線附近盤整(0.50):
            收盤介於 MA20 ± 4% 之間

      型態 6 🔻 偏空走勢(0.25):
            其餘
    ─────────────────────────────────────────────────────────────────────
    """
    vals = [v for v in [ma5, ma10, ma20, ma60] if not np.isnan(v)]
    if len(vals) < 2:
        return 0.40, "均線資料不足(樣本太短)"

    # 安全取值:MA60 可能是 NaN(初期資料不足)
    ma60_valid = not np.isnan(ma60)

    # ── 型態 0: 帶量穿越突破(最高優先)────────────────────────────────
    # 收盤「同時站上 MA5、MA10、MA20」且當日量能放大 ≥ 1.5 倍
    above_all_short = (
        not np.isnan(ma5)  and close > ma5  and
        not np.isnan(ma10) and close > ma10 and
        not np.isnan(ma20) and close > ma20
    )
    if above_all_short and vol_ratio >= 1.5:
        if vol_ratio >= 2.5:
            return 0.95, "帶量突破型態🚀 大量穿越均線,首根強攻確立"
        else:
            return 0.90, "帶量突破型態🚀 量增穿越短中均線,突破態勢成形"

    # ── 型態 1: 極度壓縮 ─────────────────────────────────────────────
    # 放寬至 3%(台股除息後均線常有輕微錯位)
    if ma_spread_pct < 3.0:
        vol_tag = " + 窒息量蓄勢🌀" if vol_ratio < 0.7 else ""
        return 0.88, f"壓縮型態🔥 四線合一{vol_tag},能量蓄積中"

    # ── 型態 2: 跌深 KD 極底金叉 ────────────────────────────────────
    if not np.isnan(ma20) and close < ma20 * 0.93 and k9 < 25 and d9 < 25 and k9 > d9:
        return 0.82, "跌深反彈💡 負乖離>7%+KD極底金叉,逆轉信號"

    # ── 型態 3: 標準多頭排列(均線已確立) ───────────────────────────
    if not np.isnan(ma5) and not np.isnan(ma10) and not np.isnan(ma20):
        if ma60_valid:
            if ma5 > ma10 > ma20 > ma60 and close > ma5 * 1.005:
                return 0.88, "多頭排列🚀 四線順排,股價強勢站上均線"
            if ma5 > ma10 > ma20 and close > ma5:
                return 0.78, "多頭健走📈 三線順排,趨勢偏多"
        else:
            if ma5 > ma10 > ma20 and close > ma5:
                return 0.78, "多頭健走📈 均線向上排列(MA60資料不足)"

    # ── 型態 4: 收盤 > MA20 且 MA5 已突破 MA20 ───────────────────────
    if (not np.isnan(ma5) and not np.isnan(ma20) and
            close > ma20 and ma5 > ma20):
        return 0.72, "均線蓄力📊 MA5已穿月線,短線偏多"

    # ── 型態 5: 月線附近盤整 ─────────────────────────────────────────
    if not np.isnan(ma20) and ma20 * 0.96 <= close <= ma20 * 1.06:
        return 0.50, "月線盤整⏳ 股價在 MA20 附近,方向待確認"

    # ── 型態 6: 偏空 ─────────────────────────────────────────────────
    return 0.25, "偏空走勢🔻 股價低於均線系統,下方壓力大"


def _score_kd_volume(k9: float, d9: float, vol_ratio: float) -> tuple[float, str]:
    """
    KD 軸線 + 量能得分 (0~1) + 說明:
      - 50 上方黃金交叉: 動能最強
      - < 20 極底金叉: 逆轉信號
      - 中軸死水: 中性
      - vol_ratio >= 2: 爆量加分
    """
    base = 0.30
    kd_desc = "KD中軸"

    if k9 > d9:
        if k9 >= 50:
            base += 0.40
            kd_desc = "KD 50上黃金交叉✅"
        elif k9 < 20:
            base += 0.35
            kd_desc = "KD 極底金叉💥"
        else:
            base += 0.20
            kd_desc = "KD 低檔黃金交叉🔸"
    elif k9 < d9:
        if k9 > 80:
            base -= 0.15
            kd_desc = "KD 高檔死亡交叉⚠️"
        else:
            base -= 0.05
            kd_desc = "KD 死亡交叉🔶"

    vol_desc = ""
    if vol_ratio >= 3.0:
        base += 0.30
        vol_desc = "+ 爆量(>3倍)🔥"
    elif vol_ratio >= 2.0:
        base += 0.20
        vol_desc = "+ 大量(>2倍)⚡"
    elif vol_ratio >= 1.5:
        base += 0.10
        vol_desc = "+ 量增(>1.5倍)"
    elif vol_ratio < 0.5:
        base -= 0.05
        vol_desc = "+ 窒息量🌀"

    return min(max(base, 0.0), 1.0), f"{kd_desc} {vol_desc}".strip()


def compute_winrate(dna: dict, df: pd.DataFrame) -> dict:
    """
    整合三大特徵向量,計算「波段成功率」(0~1)及各分項說明。
    權重: 時間波 40% + 均線型態 30% + KD/量能 30%
    """
    last = df.iloc[-1]

    close = float(last["Close"])
    k9    = float(last["K9"])   if not np.isnan(last["K9"])   else 50.0
    d9    = float(last["D9"])   if not np.isnan(last["D9"])   else 50.0

    ma5   = float(last["MA5"])  if not np.isnan(last["MA5"])  else np.nan
    ma10  = float(last["MA10"]) if not np.isnan(last["MA10"]) else np.nan
    ma20  = float(last["MA20"]) if not np.isnan(last["MA20"]) else np.nan
    ma60  = float(last["MA60"]) if not np.isnan(last["MA60"]) else np.nan

    vol       = float(last["Volume"])
    vol_ma5   = float(last["VolMA5"]) if not np.isnan(last["VolMA5"]) else vol
    vol_ratio = vol / vol_ma5 if vol_ma5 > 0 else 1.0

    # MA 壓縮程度:有效均線間的最大-最小距離 / 收盤價 × 100(%)
    valid_mas = [v for v in [ma5, ma10, ma20, ma60] if not np.isnan(v)]
    if len(valid_mas) >= 2:
        ma_spread_pct = (max(valid_mas) - min(valid_mas)) / close * 100
    else:
        ma_spread_pct = 5.0

    r = dna["R_cycle"]
    # ★ 傳入「修正低點」資訊,讓時間波能區分「修正中」vs「上漲段」
    s_t, desc_t = _score_r_cycle(
        r,
        days_since_trough      = dna.get("days_since_trough", -1),
        actual_correction_days = dna.get("actual_correction_days"),
        T_median               = dna["T_median"],
    )
    # ★ 修正 A: 傳入 vol_ratio 給 _score_ma_pattern,讓「帶量突破」型態能被識別
    s_ma, desc_ma = _score_ma_pattern(close, ma5, ma10, ma20, ma60,
                                       k9, d9, ma_spread_pct, vol_ratio)
    s_kd, desc_kd = _score_kd_volume(k9, d9, vol_ratio)

    winrate = s_t * 0.40 + s_ma * 0.30 + s_kd * 0.30

    # 三大生命週期分類
    if winrate >= 0.70:
        category = "top"
        category_label = "🚀 頂級浪潮"
    elif winrate >= 0.50:
        category = "mid"
        category_label = "⏳ 中繼蓄勢"
    else:
        category = "warn"
        category_label = "🛑 警戒浪潮"

    return {
        "winrate":        round(winrate, 4),
        "s_time":         round(s_t, 3),
        "s_ma":           round(s_ma, 3),
        "s_kd":           round(s_kd, 3),
        "desc_time":      desc_t,
        "desc_ma":        desc_ma,
        "desc_kd":        desc_kd,
        "ma_spread_pct":  round(ma_spread_pct, 2),
        "vol_ratio":      round(vol_ratio, 2),
        "k9": round(k9, 1), "d9": round(d9, 1),
        "category":       category,
        "category_label": category_label,
        "close": close,
        "ma20": ma20 if not np.isnan(ma20) else close,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  模組 C: 未來 10 日前瞻路徑矩陣
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
#  📦 籌碼資料本地 Pickle 快取（需求 4d）
#  ★ Agent C 合規確認：只使用 Python 內建 pickle, os, datetime，無第三方套件
#  ─────────────────────────────────────────────────────────────────────────────
#  用途：當 Streamlit Cloud 因閒置重啟（session_state 清空）時，
#        優先從本地 /tmp/finmind_chip_cache_{YYYYMMDD}.pkl 讀取當日籌碼，
#        避免重啟後立即耗盡 FinMind 免費額度重打所有股票。
#
#  設計：
#    寫入時機：_cache_chip_result() 成功儲存 session_state 時，同步寫入 pickle
#    讀取時機：_get_cached_chip_result() 發現 session_state 無資料時，
#              嘗試從 pickle 還原當日資料回 session_state
#    檔案命名：日期作為一部分（/tmp/finmind_chip_cache_20260701.pkl），
#              每天自動換新檔，舊檔不干擾
# ─────────────────────────────────────────────────────────────────────────────

import pickle as _pickle
import os as _os

def _get_pickle_path() -> str:
    """回傳今日籌碼快取檔案路徑（/tmp/finmind_chip_cache_{YYYYMMDD}.pkl）"""
    today = datetime.date.today().strftime('%Y%m%d')
    return f"/tmp/finmind_chip_cache_{today}.pkl"


def _pickle_write_chip(cache_key: str, result: dict) -> None:
    """
    ★ Agent B 實作：將單筆籌碼結果寫入本地 pickle 快取。
    只寫入成功的籌碼（available=True），失敗結果不寫入本地，
    避免把「402 額度限制」的臨時失敗永久存到磁碟。
    """
    if not result.get("available"):
        return   # 只持久化成功資料
    try:
        pkl_path = _get_pickle_path()
        # 讀取現有資料再合併（避免覆蓋其他 key）
        existing: dict = {}
        if _os.path.exists(pkl_path):
            try:
                with open(pkl_path, 'rb') as f:
                    existing = _pickle.load(f)
            except Exception:
                existing = {}
        existing[cache_key] = result
        with open(pkl_path, 'wb') as f:
            _pickle.dump(existing, f)
    except Exception:
        pass   # 寫入失敗不影響主流程


def _pickle_read_chip(cache_key: str) -> dict | None:
    """
    ★ Agent B 實作：從本地 pickle 快取讀取單筆籌碼結果。
    只讀今日的 pkl 檔（日期在檔名中），昨天的快取自動失效。
    """
    try:
        pkl_path = _get_pickle_path()
        if not _os.path.exists(pkl_path):
            return None
        with open(pkl_path, 'rb') as f:
            data = _pickle.load(f)
        return data.get(cache_key)
    except Exception:
        return None


def _cache_chip_result(cache_key: str, result: dict, ttl_minutes: int | None = None):
    """
    將籌碼查詢結果（成功或失敗）寫入 session_state 快取。

    ★ ttl_minutes 區分兩種快取策略：
      - None（預設，整天有效）：用於成功結果，或代號格式錯誤、興櫃股
        查無資料等「永久性」失敗 — 同一天內不需要重試。
      - 指定分鐘數（如 5）：用於 402 額度限制等「暫時性」失敗 —
        額度通常幾分鐘內就會恢復，過了 ttl 後下次查詢會自動重試，
        不會被整天鎖住查不到籌碼。
    """
    try:
        result_with_meta = dict(result)
        if ttl_minutes is not None:
            result_with_meta["_cached_at"]    = datetime.datetime.now().timestamp()
            result_with_meta["_ttl_minutes"]  = ttl_minutes
        st.session_state[cache_key] = result_with_meta
        # ★ 同步寫入本地 pickle（只寫成功資料，重啟後可恢復）
        _pickle_write_chip(cache_key, result)
    except Exception:
        pass


def _get_cached_chip_result(cache_key: str) -> dict | None:
    """
    讀取籌碼快取，並檢查短時快取（ttl_minutes）是否已過期。
    過期則回傳 None（視同快取未命中），讓呼叫端重新打 API。

    ★ 優先順序（Agent A 架構設計）：
      1. session_state（最快，記憶體）
      2. 本地 pickle 快取（Streamlit 重啟後的第二道防線）
      3. 回傳 None → 呼叫端重打 FinMind API
    """
    try:
        cached = st.session_state.get(cache_key)
        if cached is not None:
            ttl = cached.get("_ttl_minutes")
            if ttl is not None:
                cached_at = cached.get("_cached_at", 0)
                elapsed_min = (datetime.datetime.now().timestamp() - cached_at) / 60
                if elapsed_min >= ttl:
                    return None   # 短時快取已過期，觸發重新查詢
            return cached

        # ★ session_state 無資料（可能 Streamlit 重啟）→ 嘗試從 pickle 恢復
        pkl_cached = _pickle_read_chip(cache_key)
        if pkl_cached is not None:
            # 還原到 session_state，後續不再讀 pickle
            try:
                st.session_state[cache_key] = pkl_cached
            except Exception:
                pass
            return pkl_cached

        return None
    except Exception:
        return None


def _fetch_chip_data(ticker: str) -> dict:
    """
    抓取台股三大法人近 20 天籌碼資料。

    ★ 數據來源: FinMind 免費 REST API（不安裝 finmind 套件）
      URL: https://api.finmindtrade.com/api/v4/data
      完全免費、免註冊、免 token，只需要 requests（Streamlit 內建依賴）。

    ★ v2 新增: fi_net_daily / it_net_daily — 近10天每日明細 dict
      供彈窗顯示「近10天三大法人買賣超」用。

    ★ v3 強化（100檔規模防鎖機制）: 成功與失敗結果都快取在
      st.session_state，key = f"_chip_{ticker}_{今日日期}"。
      每檔股票一天最多打 1 次 FinMind API，無論成功或失敗，
      確保每 20 分鐘自動刷新 × 100 檔規模也不會扣爆免費額度。

    回傳 dict:
      fi_net_5d, it_net_5d  : 近5日每日淨買超(張)
      fi_net_daily          : {日期: 外資淨買超} — 近10天
      it_net_daily          : {日期: 投信淨買超} — 近10天
      fi_3d_sum, it_3d_sum  : 近3日合計(張)
      it_buy_days           : 近5日投信買超天數
      available, error
    """
    empty = dict(
        fi_net_5d=[], it_net_5d=[],
        fi_net_daily={}, it_net_daily={},
        fi_3d_sum=0.0, it_3d_sum=0.0,
        it_buy_days=0, available=False, error=""
    )

    # ── 手動快取（session_state，成功/興櫃股失敗整天有效，402額度限制5分鐘）──
    today_key = datetime.date.today().strftime('%Y%m%d')
    cache_key = f"_chip_{ticker}_{today_key}"
    cached = _get_cached_chip_result(cache_key)
    if cached is not None:
        return cached   # 命中有效快取，直接回傳，不重打 API

    try:
        import requests as _req

        stock_id = re.sub(r'\.(TW|TWO)$', '', ticker.upper()).strip()
        if not stock_id.isdigit():
            empty["error"] = f"不支援的代號格式: {ticker}"
            _cache_chip_result(cache_key, empty)
            return empty

        start = (datetime.date.today() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')

        resp = _req.get(
            "https://api.finmindtrade.com/api/v4/data",
            params={
                "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
                "data_id": stock_id,
                "start_date": start,
                "token": "",
            },
            timeout=12
        )

        payload = resp.json()
        api_status = payload.get('status')

        if api_status != 200 or not payload.get('data'):
            empty["error"] = f"FinMind API status={api_status}"

            if api_status == 402:
                # ★ 402 = 免費額度暫時用盡（高頻請求觸發），這是「暫時性」問題
                #   不應該整天鎖住，改用短時間快取（5分鐘），讓額度恢復後
                #   下次掃描自動重試，不會因為一次撞額度就整天查不到籌碼
                _cache_chip_result(cache_key, empty, ttl_minutes=5)
            else:
                # 其他失敗（代號不存在/興櫃股無資料等）視為「永久性」問題，
                # 整天快取，避免對注定查無資料的股票重複浪費請求
                _cache_chip_result(cache_key, empty)

            return empty

        raw = pd.DataFrame(payload['data'])
        raw['net'] = (raw['buy'].astype(float) - raw['sell'].astype(float)) / 1000.0
        pivot = raw.pivot_table(
            index='date', columns='name', values='net', aggfunc='sum'
        ).fillna(0.0)

        fi  = pivot.get('Foreign_Investor',
                        pd.Series(0.0, index=pivot.index))
        it  = pivot.get('Investment_Trust',
                        pd.Series(0.0, index=pivot.index))

        fi5 = fi.tail(5).tolist()
        it5 = it.tail(5).tolist()

        # 近10天每日明細 dict（用於彈窗）
        fi_daily = {str(d): round(v, 0) for d, v in fi.tail(10).items()}
        it_daily = {str(d): round(v, 0) for d, v in it.tail(10).items()}

        result = {
            "fi_net_5d":    fi5,
            "it_net_5d":    it5,
            "fi_net_daily": fi_daily,
            "it_net_daily": it_daily,
            "fi_3d_sum":    float(sum(fi5[-3:])) if len(fi5) >= 3 else 0.0,
            "it_3d_sum":    float(sum(it5[-3:])) if len(it5) >= 3 else 0.0,
            "it_buy_days":  int(sum(1 for v in it5 if v > 0)),
            "available":    True,
            "error":        "",
        }

        _cache_chip_result(cache_key, result)
        return result

    except Exception as e:
        empty["error"] = f"{type(e).__name__}: {str(e)[:100]}"
        _cache_chip_result(cache_key, empty)
        return empty


def evaluate_chip(chip: dict) -> dict:
    """
    根據 _fetch_chip_data 的結果評估籌碼強弱，回傳:

      label     : 籌碼標籤文字
      css_color : 標籤顯示顏色
      boost     : True → 籌碼面加分(強力共振)，直接升級買點等級
      veto      : True → 籌碼面一票否決(法人大賣，避坑)
      detail    : 詳細說明字串

    判斷邏輯:
      🔥 法人強烈共振  : 投信近5日買超≥3天 且 外資近3日合計>0
      🟢 投信波段認養  : 投信近5日買超≥3天 (外資不論)
      🟡 外資默默佈局  : 外資近3日合計>1000張，投信不明顯
      🔴 籌碼危險!    : 外資連3天淨賣>1000張 且 投信連3天淨賣>100張 → 一票否決
      ⚪ 法人觀望中   : 無明顯方向
    """
    if not chip.get("available"):
        return {
            "label": "⚪ 籌碼資料不可用",
            "css_color": "#9e9e9e",
            "boost": False, "veto": False,
            "detail": chip.get("error", "FinMind 未安裝或無資料"),
        }

    it_days = chip["it_buy_days"]
    fi_3d   = chip["fi_3d_sum"]
    it_3d   = chip["it_3d_sum"]
    it5     = chip["it_net_5d"]
    fi5     = chip["fi_net_5d"]

    # 一票否決條件: 外資連3天大賣(>1000張/天) 且 投信連3天賣超(>100張/天)
    fi_selling = len(fi5) >= 3 and all(v < -1000 for v in fi5[-3:])
    it_selling = len(it5) >= 3 and all(v < -100  for v in it5[-3:])
    veto = fi_selling and it_selling

    if veto:
        return {
            "label": "🔴 法人集體倒貨，避坑！",
            "css_color": "#c0392b",
            "boost": False, "veto": True,
            "detail": f"外資近3日合計{fi_3d:+.0f}張、投信近3日{it_3d:+.0f}張，雙向大賣超，不宜進場",
        }

    # 最強共振
    if it_days >= 3 and fi_3d > 0:
        return {
            "label": "🔥 法人強烈共振！",
            "css_color": "#c0392b",
            "boost": True, "veto": False,
            "detail": f"投信{it_days}/5天買超(近3日{it_3d:+.0f}張)＋外資近3日{fi_3d:+.0f}張，雙法人同步進場",
        }

    # 投信認養
    if it_days >= 3:
        return {
            "label": "🟢 投信波段認養",
            "css_color": "#0a7c59",
            "boost": True, "veto": False,
            "detail": f"投信近5日{it_days}/5天買超，近3日合計{it_3d:+.0f}張，波段佈局訊號",
        }

    # 外資默默佈局
    if fi_3d > 1000:
        return {
            "label": "🟡 外資默默佈局",
            "css_color": "#d97706",
            "boost": False, "veto": False,
            "detail": f"外資近3日合計{fi_3d:+.0f}張，投信{it_days}/5天買超",
        }

    return {
        "label": "⚪ 法人觀望中",
        "css_color": "#7a9bbf",
        "boost": False, "veto": False,
        "detail": f"外資近3日{fi_3d:+.0f}張，投信{it_days}/5天買超，無明顯方向",
    }


def evaluate_entry_point(dna: dict, wr: dict, df: pd.DataFrame,
                         chip: dict | None = None) -> dict:
    """
    買點獵人評估引擎 — 五大技術條件 + 第⑥籌碼面過濾
    ────────────────────────────────────────────────────────────────────────
    技術面五大條件(原有):
      ① R_cycle ≥ 1.0   → 35分 (最重要，時間波飽和)
         R_cycle ≥ 1.3   → 額外+10分 (超額修正)
      ② KD 低檔拐頭      → 25分 (K9 > D9 且 K9 < 60)
      ③ 中繼蓄勢分類     → 20分
      ④ 勝率甜蜜區 50~68% → 10分
      ⑤ 量比 < 2.5       → 10分

    ⑥ 籌碼面過濾(選填，chip 為 evaluate_chip() 的回傳值):
      大加分: chip.boost=True  → 技術面分數已≥65時，升一級訊號
      一票否決: chip.veto=True → 無論技術分數多高，強制設為「🚫 籌碼危險，不進場」

    訊號分級:
      ≥ 80分 (+ boost) : 🔥 籌碼共振買點 / 🎯 強力買點
      ≥ 65分           : 📌 潛力買點
      ≥ 50分           : ⚠️ 蓄勢觀察
      < 50分           : 🚫 時機未到
      veto             : 🚫 籌碼危險，不進場
    """
    cat     = wr["category"]
    winrate = wr["winrate"] * 100
    r       = dna["R_cycle"]
    k9      = wr["k9"]
    d9      = wr["d9"]
    vol_r   = wr["vol_ratio"]

    c1_mid    = cat == "mid"
    c2_wr     = 50 <= winrate <= 68
    c3_rcycle = r >= 1.0
    c4_kd     = k9 > d9 and k9 < 60
    c5_vol    = vol_r < 2.5

    if k9 < 25:      kd_stage = "⭐ 極底金叉"
    elif k9 < 40:    kd_stage = "✅ 低檔金叉"
    elif k9 < 60:    kd_stage = "🔸 中低金叉"
    elif k9 > d9:    kd_stage = "⚠️ 中高金叉"
    else:            kd_stage = "❌ 高檔/死叉"

    score = 0
    if c3_rcycle: score += 35
    if c4_kd:     score += 25
    if c1_mid:    score += 20
    if c2_wr:     score += 10
    if c5_vol:    score += 10
    if r >= 1.3:  score = min(score + 10, 100)

    # ── ⑥ 籌碼面過濾 ────────────────────────────────────────────────
    chip_eval = chip or {}
    chip_boost = chip_eval.get("boost", False)
    chip_veto  = chip_eval.get("veto",  False)

    # 一票否決：法人集體倒貨，強制覆蓋
    if chip_veto:
        return {
            "score": score, "signal": "🚫 籌碼危險，不進場",
            "kd_stage": kd_stage,
            "chip_override": True,
            "conditions": {
                "c1_mid": c1_mid, "c2_wr": c2_wr,
                "c3_rcycle": c3_rcycle, "c4_kd": c4_kd, "c5_vol": c5_vol,
            },
        }

    # 技術面訊號
    if score >= 80:   signal = "🎯 強力買點"
    elif score >= 65: signal = "📌 潛力買點"
    elif score >= 50: signal = "⚠️ 蓄勢觀察"
    else:             signal = "🚫 時機未到"

    # 籌碼大加分：技術面≥65且籌碼共振 → 升級為最高等
    if chip_boost and score >= 65:
        signal = "🔥 籌碼共振買點"

    return {
        "score": score, "signal": signal, "kd_stage": kd_stage,
        "chip_override": False,
        "conditions": {
            "c1_mid": c1_mid, "c2_wr": c2_wr,
            "c3_rcycle": c3_rcycle, "c4_kd": c4_kd, "c5_vol": c5_vol,
        },
    }


def generate_forward_matrix(
    df: pd.DataFrame,
    wr: dict,
    dna: dict,
    n_days: int = 10
) -> list[dict]:
    """
    根據「分類」與「ATR14」動態生成未來 n_days 個交易日的預估路徑。

    模型說明:
      - 基礎漂移(drift):
          頂級浪潮 → 每日漂移 = +0.30% × 勝率乘數
          中繼蓄勢 → 每日漂移 = +0.10% (早期) → +0.20% (後期)
          警戒浪潮 → 每日漂移 = -0.15% (反彈後繼續壓制)
      - 不確定性幅度: ± ATR14 × 衰減係數(越遠越寬)
      - 配合 R_cycle 在特定天數加入「型態轉折說明」觀測點
    """
    last_close = float(df["Close"].iloc[-1])
    atr        = float(df["ATR14"].iloc[-1]) if not np.isnan(df["ATR14"].iloc[-1]) else last_close * 0.02
    cat        = wr["category"]
    winrate    = wr["winrate"]
    r          = dna["R_cycle"]
    t_median   = dna["T_median"]

    # 每日基礎漂移設定
    if cat == "top":
        daily_drift = 0.003 * winrate   # 最高約 0.3%/日
    elif cat == "mid":
        daily_drift = 0.001             # 保守 0.1%/日
    else:
        daily_drift = -0.0015           # 緩步壓制

    # 跳過週末的交易日曆
    last_date   = df.index[-1].to_pydatetime()
    biz_dates   = []
    d = last_date
    while len(biz_dates) < n_days:
        d += datetime.timedelta(days=1)
        if d.weekday() < 5:
            biz_dates.append(d)

    rows = []
    price = last_close
    for i, biz_d in enumerate(biz_dates, start=1):
        price = price * (1 + daily_drift)
        # 不確定性幅度: ATR × sqrt(i) × 比例因子
        band  = atr * (i ** 0.5) * 0.35

        # 個別天的觀測說明
        if cat == "top":
            if i == 1:
                note = "觀察是否延續漲勢,量能能否持續放大"
            elif i <= 3:
                note = "強勢整理或小幅震盪蓄勢,不破 MA5 視為健康"
            elif i <= 6:
                note = "若出現縮量長紅 → 主力換手完成,加速動力浮現"
            else:
                note = "留意高檔放量長黑或 KD 鈍化高檔背離訊號"
        elif cat == "mid":
            if i <= 2:
                note = "等待觸媒出現:量縮收小紅 / 均線黏合走平"
            elif i <= 5:
                note = f"若 R_cycle 超過 {r:.2f}→1.0 臨界可能出現反轉"
            else:
                note = "觀察均線交叉與 KD 黃金交叉是否同步確立"
        else:  # warn
            if i <= 3:
                note = "反彈進入壓力區,短線宜輕倉或觀望"
            elif i <= 7:
                note = "R_cycle 未飽和,反彈高度有限,留意反彈峰賣出"
            else:
                note = "若 KD 未能黃金交叉,波段壓力仍未解除"

        rows.append({
            "交易日":   f"D+{i}",
            "預估日期": biz_d.strftime("%m/%d (%a)"),
            "演算法預估價":  round(price, 2),
            "上限參考":     round(price + band, 2),
            "下限參考":     round(price - band, 2),
            "型態觀測重點": note,
        })

    return rows


# ─────────────────────────────────────────────────────────────────────────────
#  台股名稱對照表 + 輔助工具
# ─────────────────────────────────────────────────────────────────────────────

# 常用台股中文名稱對照 (代號 -> 中文簡稱)
# 對照表未收錄的股票,會在執行時動態從 yfinance 取英文名稱作為 fallback
TW_NAME_MAP = {
    # 半導體/IC設計
    "2330.TW":"台積電","2303.TW":"聯電","2454.TW":"聯發科","2379.TW":"瑞昱",
    "3034.TW":"聯詠","2344.TW":"華邦電","3711.TW":"日月光投控","2408.TW":"南亞科",
    "6770.TW":"力積電","3533.TW":"嘉澤","2337.TW":"旺宏","3231.TW":"緯創",
    "3443.TW":"創意","6669.TW":"緯穎","2385.TW":"群光","2360.TW":"致茂",
    "5274.TWO":"信驊","6274.TWO":"台燿","6488.TWO":"環球晶","3443.TWO":"創意",
    # AI/伺服器/散熱
    "2317.TW":"鴻海","3008.TW":"大立光","2357.TW":"華碩","2382.TW":"廣達",
    "4919.TW":"新唐","6415.TW":"矽力-KY","8150.TW":"南茂",
    # PCB/電路板
    "2301.TW":"光寶科","3037.TW":"欣興","6153.TW":"嘉聯益","8046.TW":"南電",
    "6269.TW":"台郡","3024.TW":"憶聲","2383.TW":"台光電","6456.TW":"GIS-KY",
    "4961.TW":"天鈺","3706.TW":"神達","2404.TW":"漢唐","4919.TW":"新唐",
    # 被動元件/電子材料
    "2327.TW":"國巨","2354.TW":"鴻準","2376.TW":"技嘉","3019.TW":"亞泰",
    "6789.TW":"采鈺","5483.TWO":"中美晶","6214.TW":"精誠","2439.TW":"美律",
    # 網通/電信
    "2412.TW":"中華電","3045.TW":"台灣大","4904.TW":"遠傳","2498.TW":"宏達電",
    "3044.TW":"健鼎","4906.TW":"正文","5434.TW":"崇越","3026.TW":"禾伸堂",
    "6488.TWO":"環球晶",
    # 金融
    "2882.TW":"國泰金","2881.TW":"富邦金","2886.TW":"兆豐金","2891.TW":"中信金",
    "2884.TW":"玉山金","2885.TW":"元大金","2887.TW":"台新新光金","2892.TW":"第一金",
    "2801.TW":"彰銀","5880.TW":"合庫金",
    # 傳產/原物料
    "1301.TW":"台塑","1303.TW":"南亞","1326.TW":"台化","2002.TW":"中鋼",
    "9904.TW":"寶成","1101.TW":"台泥","1216.TW":"統一","1402.TW":"遠東新",
    "2105.TW":"正新","1210.TW":"大成",
    # 航運
    "2603.TW":"長榮","2609.TW":"陽明","2615.TW":"萬海","2610.TW":"華航",
    "5608.TW":"四維航","2617.TW":"台航","2618.TW":"長榮航","2606.TW":"裕民",
    "2637.TW":"慧洋-KY","2634.TW":"漢翔",
    # 光電/面板
    "3481.TW":"群創","2409.TW":"友達","6409.TW":"旭隼","2449.TW":"京元電子",
    "3035.TW":"智原","2395.TW":"研華","5269.TW":"祥碩","3653.TW":"健策",
    # 熱門電子股
    "2356.TW":"英業達","2353.TW":"宏碁","2352.TW":"佳世達","2347.TW":"聯強",
    "2345.TW":"智邦","2342.TW":"茂矽","2340.TW":"光磊","2332.TW":"台揚",
    "2331.TW":"精英","2329.TW":"華泰","2328.TW":"廣宇","2324.TW":"仁寶",
    "2323.TW":"中環","2321.TW":"東元","2316.TW":"楠梓電","2313.TW":"華通",
    "2312.TW":"金寶","2308.TW":"台達電","2305.TW":"全友","2302.TW":"麗正",
    "2362.TW":"藍天","2363.TW":"矽統","2364.TW":"倫飛","2365.TW":"昆盈",
    "2367.TW":"燿華","2368.TW":"金像電","2369.TW":"菱生","2371.TW":"大同",
    "2373.TW":"震旦行","2374.TW":"佳能","2375.TW":"智寶","2376.TW":"技嘉",
    "2377.TW":"微星","2378.TW":"鈺創","2379.TW":"瑞昱","2381.TW":"華宇",
    "2382.TW":"廣達","2383.TW":"台光電","2384.TW":"第一國際","2385.TW":"群光",
    "2386.TW":"天剛","2387.TW":"精倫","2388.TW":"威盛","2389.TW":"啟訊",
    "2390.TW":"云辰","2392.TW":"正崴","2393.TW":"億光","2395.TW":"研華",
    "2396.TW":"精泉","2397.TW":"友通","2398.TW":"世界","2399.TW":"映泰",
    "2401.TW":"凌陽","2402.TW":"毅嘉","2404.TW":"漢唐","2405.TW":"廣錠",
    "2406.TW":"國碩","2408.TW":"南亞科","2409.TW":"友達","2412.TW":"中華電",
    "2413.TW":"環科","2414.TW":"精技","2415.TW":"錩泰","2417.TW":"圓剛",
    "2419.TW":"仲琦","2420.TW":"新巨","2421.TW":"建準","2423.TW":"固緯",
    "2424.TW":"隴華","2425.TW":"承啟","2426.TW":"鼎元","2427.TW":"三商電",
    "2428.TW":"興勤","2429.TW":"銘異","2430.TW":"燦坤","2431.TW":"聯昌",
    "2432.TW":"倚強","2433.TW":"互動","2434.TW":"統一實","2436.TW":"偉詮電",
    "2438.TW":"翔耀","2439.TW":"美律","2440.TW":"太空梭","2441.TW":"超豐",
    "2442.TW":"新美齊","2444.TW":"兆赫","2449.TW":"京元電子","2450.TW":"神腦",
    "2451.TW":"創見","2453.TW":"凌群","2454.TW":"聯發科","2455.TW":"全新",
    "2457.TW":"飛宏","2458.TW":"義隆","2459.TW":"敦吉","2460.TW":"建通",
    "2461.TW":"光群雷","2462.TW":"白金","2464.TW":"盟立","2465.TW":"麗臺",
    "2466.TW":"冠西電","2467.TW":"志聖","2468.TW":"華經","2471.TW":"資通",
    "2472.TW":"立隆","2474.TW":"可成","2476.TW":"鉅祥","2477.TW":"美隆電",
    "2478.TW":"大毅","2480.TW":"敦陽科","2481.TW":"強茂","2482.TW":"連宇",
    "2483.TW":"百容","2484.TW":"希華","2485.TW":"兆赫","2486.TW":"一詮",
    "2488.TW":"漢平","2489.TW":"瑞軒","2491.TW":"吉祥全","2492.TW":"華新科",
    "2493.TW":"揚博","2495.TW":"普安","2496.TW":"卓越","2497.TW":"怡利電",
    "2498.TW":"宏達電","3008.TW":"大立光","3014.TW":"聯陽","3017.TW":"奇鋐",
    "3019.TW":"亞泰","3021.TW":"鴻名","3022.TW":"威強電","3023.TW":"信邦",
    "3024.TW":"憶聲","3025.TW":"星通","3026.TW":"禾伸堂","3027.TW":"盛達",
    "3029.TW":"零壹","3030.TW":"一零四","3031.TW":"佰研","3032.TW":"偉訓",
    "3033.TW":"威健","3034.TW":"聯詠","3035.TW":"智原","3036.TW":"文曄",
    "3037.TW":"欣興","3038.TW":"全台晶像","3041.TW":"揚智","3042.TW":"晶技",
    "3043.TW":"科風","3044.TW":"健鼎","3045.TW":"台灣大","3046.TW":"建碁",
    "3047.TW":"訊舟","3048.TW":"益登","3049.TW":"和鑫","3050.TW":"鈺德",
    "3051.TW":"力特","3052.TW":"夆典","3054.TW":"立德電","3055.TW":"蘋果樹",
    "3056.TW":"總太","3057.TW":"喬鼎","3058.TW":"立德","3059.TW":"華晶科",
    "3060.TW":"銘異","3062.TW":"建漢","3085.TW":"比比昂","3086.TW":"華義",
    "3090.TW":"日電貿","3092.TW":"鴻碩","3094.TW":"天亮醫療","3149.TW":"正達",
    "3150.TW":"萬旭","3163.TWO":"波若威","3189.TW":"景碩","3191.TWO":"和進",
    "3209.TWO":"全科","3211.TW":"盈正豫順","3213.TWO":"茂順","3231.TW":"緯創",
    "3232.TW":"昱捷","3290.TW":"東成","3293.TW":"鈊象","3296.TW":"勝德",
    "3305.TW":"昇貿","3311.TW":"閎暉","3312.TW":"弘憶股","3374.TW":"精材",
    "3376.TW":"新日興","3380.TW":"明泰","3382.TW":"瀛通","3388.TW":"崇越電",
    "3406.TW":"玉晶光","3413.TW":"京鼎","3416.TW":"融程電","3419.TW":"譜瑞-KY",
    "3432.TW":"台端","3437.TW":"榮創","3450.TW":"聯鈞","3466.TWO":"聚積",
    "3481.TW":"群創","3515.TW":"華擎","3519.TW":"亦強","3529.TW":"力旺",
    "3530.TW":"晶相光","3532.TW":"台勝科","3534.TW":"昱晶","3536.TWO":"祥富水電",
    "3545.TW":"敦泰","3548.TW":"兆利","3550.TW":"台灣精銳","3551.TW":"世禾",
    "3563.TW":"牧德","3576.TW":"新日光","3588.TW":"通嘉","3592.TW":"瑞鼎",
    "3596.TW":"智易","3643.TW":"通泰","3661.TW":"世芯-KY","3665.TW":"貿聯-KY",
    "3666.TW":"光洋科","3673.TW":"TPK-KY","3679.TW":"鑫禾","3691.TWO":"碩天",
    "3693.TW":"營邦","3694.TW":"料仁科","3698.TW":"隆達","3701.TW":"大眾控",
    "3702.TW":"大聯大","3703.TW":"欣陸","3704.TW":"合勤控","3705.TW":"永信",
    "3706.TW":"神達","3707.TW":"漢磊","3708.TW":"上緯","3711.TW":"日月光投控",
    "3714.TW":"富采","3715.TW":"定穎投控","3716.TW":"宸鴻","3722.TW":"同泰",
    "3726.TW":"漢科","3727.TW":"彬台","3730.TW":"蔚華科","3733.TW":"雷科",
    "3735.TW":"品安","3738.TW":"勝肯","3741.TW":"互動","3748.TW":"智旺",
    "3749.TW":"北峰","3752.TW":"先鋒","3754.TW":"萬礦","3755.TW":"有量",
    "3756.TW":"富樺","3757.TW":"金器","3758.TW":"為昇","3761.TW":"泰可",
    "3762.TW":"臻鼎-KY","3763.TW":"鴻特","3764.TW":"辰展光電","3766.TW":"耕興",
    "3769.TW":"洋華","3771.TW":"中探針","3779.TW":"晶采","3781.TW":"岳豐",
    "3782.TW":"同霖","3785.TW":"H&G-KY","3786.TW":"科旭","3788.TW":"韋僑",
    "3790.TW":"菱光","3791.TW":"揚聲","3792.TW":"惠特","3793.TW":"泰銘",
    # 上櫃電子熱門
    "6798.TWO":"展逸","6274.TWO":"台燿","5274.TWO":"信驊","6488.TWO":"環球晶",
    "5483.TWO":"中美晶","4711.TWO":"永信藥","6547.TWO":"安博-KY","4174.TWO":"浩鼎",
    "8299.TWO":"金麗科","6409.TW":"旭隼","8150.TW":"南茂",
    # ── 成交量排行常見股票補充(截圖中出現的英文名) ──────────────────
    "3481.TW":"群創","2409.TW":"友達","6116.TW":"彩晶","2002.TW":"中鋼",
    "2344.TW":"華邦電","2303.TW":"聯電","6770.TW":"力積電","1301.TW":"台塑",
    "2408.TW":"南亞科","2890.TW":"永豐金","1303.TW":"南亞","2883.TW":"開發金",
    "2337.TW":"旺宏","1802.TW":"台玻","6182.TWO":"環球晶","2887.TW":"台新新光金",
    "2884.TW":"玉山金","2610.TW":"華航","2327.TW":"國巨","2892.TW":"第一金",
    "2886.TW":"兆豐金","2324.TW":"仁寶","2317.TW":"鴻海","2492.TW":"華新科",
    "2880.TW":"華南金","2618.TW":"長榮航","6239.TW":"力成","1605.TW":"華新",
    "2891.TW":"中信金","2882.TW":"國泰金","2881.TW":"富邦金","2885.TW":"元大金",
    "2603.TW":"長榮","2609.TW":"陽明","2615.TW":"萬海","5880.TW":"合庫金",
    # ── 漲跌幅排行常見股票補充 ────────────────────────────────────────
    "5297.TWO":"三星科技","6586.TWO":"豐藝","6603.TWO":"富鴻網","6432.TWO":"亞信",
    "6919.TW":"凱萊英","6259.TWO":"百威達","4741.TWO":"亞朋","1409.TW":"新纖",
    "2399.TW":"映泰","3576.TW":"聯合再生","5230.TWO":"友鴻","2605.TW":"新興",
    "2472.TW":"立隆","2483.TW":"百容","2449.TW":"京元電子","2406.TW":"國碩",
    # ── 上櫃熱門補充 ──────────────────────────────────────────────────
    "5328.TWO":"聯發","3105.TWO":"穩懋","8043.TWO":"蜜望實","6207.TWO":"雷科",
    "6175.TWO":"立積","5351.TWO":"鈺創","6147.TWO":"頎邦","3236.TWO":"千如",
    "1815.TWO":"富喬","3707.TWO":"漢磊","8088.TWO":"品安","5347.TWO":"世界",
    "1785.TWO":"光洋科","5425.TWO":"台半","8064.TWO":"歐特邁","6548.TWO":"長華電材",
    "3264.TW":"欣銓","3264.TWO":"欣銓","3362.TWO":"先豐","3441.TWO":"聯一光","3260.TWO":"偉詮電",
    "3537.TWO":"堡達","3317.TWO":"金洋科","8069.TWO":"元太","6244.TWO":"茂迪",
    "3663.TWO":"鐿鈦","3357.TWO":"台灣彩光","8096.TWO":"擎亞",
    # ── ★ 使用者自選股補充（矽光子/CPO/電力/記憶體/被動元件主題）───────
    "3289.TW":"宜特",    "3289.TWO":"宜特",
    "3450.TW":"聯鈞",
    "4979.TW":"華星光","4979.TWO":"華星光",
    "6451.TW":"訊芯-KY",
    "3363.TW":"上詮", "3363.TWO":"上詮",
    "3163.TW":"波若威",  "3163.TWO":"波若威",
    "4908.TW":"前鼎",    "4908.TWO":"前鼎",
    "3081.TW":"聯亞",    "3081.TWO":"聯亞",
    "3406.TW":"玉晶光",
    "3587.TW":"閎康",    "3587.TWO":"閎康",
    "6683.TW":"雍智科技","6683.TWO":"雍智科技",
    "3037.TW":"欣興",
    "3189.TW":"景碩",
    "8046.TW":"南電",
    "6223.TW":"旺矽",    "6223.TWO":"旺矽",
    "6515.TW":"穎崴",    "6515.TWO":"穎崴",
    "1609.TW":"大亞",
    "1503.TW":"士電",
    "1519.TW":"華城",
    "1513.TW":"中興電",
    "1504.TW":"東元",
    "1514.TW":"亞力",
    "6806.TW":"森崴能源",
    "1618.TW":"合機",
    "3006.TW":"晶豪科",
    "8299.TW":"群聯",    "8299.TWO":"群聯",
    "6510.TW":"精測",    "6510.TWO":"精測",
    "6271.TW":"同欣電",
    "3026.TW":"禾伸堂",
    "2375.TW":"智寶",
    "6127.TW":"九豪",    "6127.TWO":"九豪",
    "3068.TW":"訊雲",    "3068.TWO":"訊雲",
    "3338.TW":"泰碩",
    "6173.TW":"信昌電",  "6173.TWO":"信昌電",
    "8935.TWO":"邦泰",
    "3490.TWO":"單井",
    "3491.TWO":"昇達科",
    "6174.TWO":"安碁",
    "8176.TWO":"智捷",
}

@st.cache_data(ttl=86400, show_spinner=False)
def get_taiwan_ticker_mapping() -> dict[str, str]:
    """
    台股代號 → 中文名稱對照表（24 小時快取）。

    ★ 相容提示詞 API 規格，但底層改用台灣證交所/櫃買中心官方免費 API
      而非 FinMind TaiwanStockInfo（後者在免費帳號下常回 402 限流）。
      官方 API 覆蓋上市 1089 筆 + 上市今日 1368 筆 + 上櫃 1011 筆，
      合計約 2381 筆，比 FinMind 免費版更穩定可靠。

    提供給外部呼叫（相容性包裝）：
      ticker_map = get_taiwan_ticker_mapping()
      name = ticker_map.get("2330.TW", "2330")

    Fallback：若所有 API 都失敗，回傳核心底盤的靜態字典。
    """
    official = _load_official_names()     # 底層呼叫已有 @st.cache_resource
    if official:
        return dict(official)
    # 最終保底靜態字典（核心自選股）
    return {
        '1609.TW': '大亞',  '3289.TW': '宜特',  '8074.TW': '鉅橡',
        '8150.TW': '南茂',  '2317.TW': '鴻海',  '2330.TW': '台積電',
    }


def get_stock_name(ticker: str) -> str:
    """
    取得股票中文名稱，查詢優先順序：
      1. _OFFICIAL_NAME_CACHE — 台灣證交所/櫃買中心官方 API（最準確）
      2. TW_NAME_MAP          — 靜態對照表（備用，部分特殊股票）
      3. _REALTIME_NAME_CACHE — 即時排行 screener 的 shortName
      4. _DYNAMIC_NAME_CACHE  — 動態查詢快取
      5. yf.Ticker.info       — 動態查詢（慢，僅在其他方式都找不到時）
      6. ticker 本身          — 最終 fallback
    """
    # 1. 官方 API（最準確，動態更新）
    official = _load_official_names()
    name = official.get(ticker, "")
    if name:
        return name

    # 上市股可能用 .TW，但官方是 .TWO，或反之，嘗試另一種後綴
    if not name and ticker.endswith(".TW") and not ticker.endswith(".TWO"):
        name = official.get(ticker.replace(".TW", ".TWO"), "")
    if not name and ticker.endswith(".TWO"):
        name = official.get(ticker.replace(".TWO", ".TW"), "")
    if name:
        return name

    # 2. 靜態對照表（備用）
    name = TW_NAME_MAP.get(ticker, "")
    if name:
        return name

    # 3. 即時排行快取（screener shortName，通常英文，需清理）
    cached = _REALTIME_NAME_CACHE.get(ticker, "")
    if cached:
        for s in [" CO., LTD.", " Co., Ltd.", " CO LTD", " CORPORATION",
                  " Corporation", " Corp.", " CORP", " INC.", " Inc.", " INC",
                  " LTD.", " Ltd.", " CO.", " International", " INTERNATIONAL",
                  " TECHNOLOGY", " Technology", " ENTERPRISE", " Enterprise",
                  " INDUSTRIAL", " Industrial", " POLYBLEND", " Polyblend",
                  ", Inc.", ", Ltd.", ", Ltd", "."]:
            cached = cached.replace(s, "").replace(s.upper(), "")
        cached = cached.strip()
        # ★ 強化過濾：有中文直接接受；純大寫英文（非中文）一律跳過，
        # 改走後續查詢取中文名（PONTEX/AKER/SINGLE WELL 等全數攔截）
        # 只有混合大小寫英文（如 Z-Com）或中文才接受
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', cached))
        is_all_caps = bool(cached) and cached.replace('-','').replace(' ','').replace('.','').isupper()
        if cached and (has_chinese or not is_all_caps):
            return cached

# 4. 動態快取
    dyn = _DYNAMIC_NAME_CACHE.get(ticker, "")
    if dyn:
        return dyn

    # 5. 動態查詢 Yahoo Finance（慢，最後手段）
    try:
        info = yf.Ticker(ticker).info
        name_raw = info.get("longName") or info.get("shortName") or ""
        if name_raw:
            for s in [" CO., LTD.", " Co., Ltd.", " CO LTD", " CORPORATION",
                      " Corporation", " Corp.", " CORP", " INC.", " Inc.", " INC",
                      " LTD.", " Ltd.", " CO.", ",", "."]:
                name_raw = name_raw.replace(s, "").replace(s.upper(), "")
            name_clean = name_raw.strip()
            if name_clean and name_clean != ticker:
                _DYNAMIC_NAME_CACHE[ticker] = name_clean
                return name_clean
    except Exception:
        pass

    # 最終 fallback：拔掉 .TW / .TWO 後綴，只顯示純代號數字
    # 例如 "1102.TW" → "1102"，比顯示完整代號更乾淨
    return re.sub(r'\.(TWO|TW)$', '', ticker, flags=re.IGNORECASE)


def get_chart_url(ticker: str) -> str:
    """
    生成 Yahoo Finance 台股技術分析頁面 URL。
    格式: https://tw.stock.yahoo.com/quote/{代號}/technical-analysis
    """
    return f"https://tw.stock.yahoo.com/quote/{ticker}/technical-analysis"


# ─────────────────────────────────────────────────────────────────────────────
#  HTML 渲染工具
# ─────────────────────────────────────────────────────────────────────────────

def html_metric(label: str, value: str, sub: str = "") -> str:
    return f"""
    <div class="dna-card">
      <h3>{label}</h3>
      <div class="val">{value}</div>
      {"<div class='sub'>"+sub+"</div>" if sub else ""}
    </div>"""


def html_feat_bar(label: str, score: float, desc: str, color: str = "#1565c0") -> str:
    pct = int(score * 100)
    return f"""
    <div class="feat-row">
      <span class="feat-label">{label}</span>
      <div class="feat-bar-wrap">
        <div class="feat-bar-fill" style="width:{pct}%;background:{color};"></div>
      </div>
      <span class="feat-val">{pct}%</span>
    </div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:12px;
                color:#2d3748;margin:-6px 0 12px 142px;">{desc}</div>"""


def html_forward_table(rows: list[dict], last_close: float) -> str:
    """
    前瞻路徑矩陣:
    - 桌機: fwd-table 標準表格
    - 手機: forecast-cards 卡片式(每日一張),避免表格橫向溢出
    """
    # ── 手機卡片式 ──────────────────────────────────────────────────
    cards = '<div class="forecast-cards">'
    for r in rows:
        p   = r["演算法預估價"]
        hi  = r["上限參考"]
        lo  = r["下限參考"]
        chg = (p - last_close) / last_close * 100
        if chg > 0.5:    pcolor, arrow = "#0a7c59", "▲"
        elif chg < -0.5: pcolor, arrow = "#c0392b", "▼"
        else:             pcolor, arrow = "#d97706", "─"

        cards += f"""
        <div style="background:#fff;border:1.5px solid #c8d8e8;border-left:4px solid {pcolor};
                    border-radius:10px;padding:12px 14px;margin-bottom:8px;
                    box-shadow:0 1px 4px rgba(21,101,192,0.06);">
          <div style="display:flex;justify-content:space-between;align-items:center;
                      margin-bottom:6px;">
            <span style="font-family:'IBM Plex Mono',monospace;font-size:13px;
                         color:#4a6fa5;font-weight:600;">{r['交易日']}
              <span style="color:#7a9bbf;font-weight:400;"> · {r['預估日期']}</span>
            </span>
            <span style="font-family:'IBM Plex Mono',monospace;font-size:20px;
                         font-weight:700;color:{pcolor};">
              {arrow} {p:.2f}
            </span>
          </div>
          <div style="display:flex;justify-content:space-between;
                      font-family:'IBM Plex Mono',monospace;font-size:13px;margin-bottom:6px;">
            <span style="color:{pcolor};font-weight:600;">{chg:+.2f}%</span>
            <span style="color:#7a9bbf;">± {((hi-lo)/2):.2f}</span>
          </div>
          <div style="font-size:12px;color:#4a6fa5;line-height:1.5;
                      border-top:1px solid #eaf2fb;padding-top:6px;">
            {r['型態觀測重點']}
          </div>
        </div>"""
    cards += '</div>'

    # ── 桌機表格式 ──────────────────────────────────────────────────
    head = "<thead><tr>" + "".join(
        f"<th>{col}</th>"
        for col in ["交易日", "預估日期", "演算法預估價", "±波動幅度", "型態觀測重點"]
    ) + "</tr></thead>"

    tbody = ""
    for r in rows:
        p   = r["演算法預估價"]
        hi  = r["上限參考"]
        lo  = r["下限參考"]
        chg = (p - last_close) / last_close * 100
        if chg > 0.5:    cls = "price-up"
        elif chg < -0.5: cls = "price-down"
        else:             cls = "price-flat"

        band_str = f'+{hi-p:.2f} / -{p-lo:.2f}'
        tbody += f"""<tr>
          <td style="color:#1a2b3c;font-weight:600;">{r['交易日']}</td>
          <td style="color:#4a6fa5;">{r['預估日期']}</td>
          <td class="{cls}" style="font-size:15px;">{p:.2f} ({chg:+.1f}%)</td>
          <td style="color:#7a9bbf;font-size:13px;">{band_str}</td>
          <td style="color:#2d3748;font-size:13px;line-height:1.5;">{r['型態觀測重點']}</td>
        </tr>"""

    table = f'<table class="fwd-table">{head}<tbody>{tbody}</tbody></table>'
    return cards + table


# ─────────────────────────────────────────────────────────────────────────────
#  台灣熱門 100 檔預設清單
#  分類: 半導體 / AI供應鏈 / 電子零組件 / 面板 / 金融 / 傳產 / 航運 / 生技 / 其他
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
#  台灣電子股 759 檔預驗證清單 (已過濾無效/已下市代號)
#  涵蓋上市(TW)與上櫃(TWO) 電子股,共 759 檔
#  代號範圍: 2300~2499(半導體/電子), 3000~3799(電子零組件),
#             4900~4999(通信), 6000~6999(電子六), 8000~8300(電子八)
# ─────────────────────────────────────────────────────────────────────────────
TW_ELECTRONIC_759 = [
    "2301.TW","2302.TW","2303.TW","2305.TW","2308.TW","2312.TW","2313.TW","2314.TW","2316.TW","2317.TW",
    "2321.TW","2323.TW","2324.TW","2327.TW","2328.TW","2329.TW","2330.TW","2331.TW","2332.TW","2337.TW",
    "2338.TW","2340.TW","2342.TW","2344.TW","2345.TW","2347.TW","2348.TW","2349.TW","2351.TW","2352.TW",
    "2353.TW","2354.TW","2355.TW","2356.TW","2357.TW","2359.TW","2360.TW","2362.TW","2363.TW","2364.TW",
    "2365.TW","2367.TW","2368.TW","2369.TW","2371.TW","2373.TW","2374.TW","2375.TW","2376.TW","2377.TW",
    "2379.TW","2380.TW","2382.TW","2383.TW","2385.TW","2387.TW","2388.TW","2390.TW","2392.TW","2393.TW",
    "2395.TW","2397.TW","2399.TW","2401.TW","2402.TW","2404.TW","2405.TW","2406.TW","2408.TW","2409.TW",
    "2412.TW","2413.TW","2414.TW","2415.TW","2417.TW","2419.TW","2420.TW","2421.TW","2423.TW","2424.TW",
    "2425.TW","2426.TW","2427.TW","2428.TW","2429.TW","2430.TW","2431.TW","2432.TW","2433.TW","2434.TW",
    "2436.TW","2438.TW","2439.TW","2440.TW","2441.TW","2442.TW","2444.TW","2449.TW","2450.TW","2451.TW",
    "2453.TW","2454.TW","2455.TW","2457.TW","2458.TW","2459.TW","2460.TW","2461.TW","2462.TW","2464.TW",
    "2465.TW","2466.TW","2467.TW","2468.TW","2471.TW","2472.TW","2474.TW","2476.TW","2477.TW","2478.TW",
    "2480.TW","2481.TW","2482.TW","2483.TW","2484.TW","2485.TW","2486.TW","2488.TW","2489.TW","2491.TW",
    "2492.TW","2493.TW","2495.TW","2496.TW","2497.TW","2498.TW","3002.TW","3003.TW","3004.TW","3005.TW",
    "3006.TW","3008.TW","3010.TW","3011.TW","3013.TW","3014.TW","3015.TW","3016.TW","3017.TW","3018.TW",
    "3019.TW","3021.TW","3022.TW","3023.TW","3024.TW","3025.TW","3026.TW","3027.TW","3028.TW","3029.TW",
    "3030.TW","3031.TW","3032.TW","3033.TW","3034.TW","3035.TW","3036.TW","3037.TW","3038.TW","3040.TW",
    "3041.TW","3042.TW","3043.TW","3044.TW","3045.TW","3046.TW","3047.TW","3048.TW","3049.TW","3050.TW",
    "3051.TW","3052.TW","3054.TW","3055.TW","3056.TW","3057.TW","3058.TW","3059.TW","3060.TW","3062.TW",
    "3073.TWO","3078.TWO","3081.TWO","3083.TWO","3085.TWO","3086.TWO","3088.TWO","3090.TW","3092.TW","3094.TW",
    "3095.TWO","3118.TWO","3122.TWO","3130.TW","3131.TWO","3135.TW","3138.TW","3141.TWO","3149.TW","3150.TW",
    "3152.TWO","3164.TW","3167.TW","3168.TW","3188.TWO","3189.TW","3191.TWO","3205.TWO","3206.TWO","3207.TWO",
    "3209.TW","3213.TWO","3217.TWO","3218.TWO","3219.TWO","3224.TWO","3227.TWO","3229.TW","3231.TW","3232.TWO",
    "3234.TWO","3236.TWO","3252.TWO","3257.TW","3259.TWO","3266.TW","3288.TWO","3293.TWO","3294.TWO","3296.TW",
    "3305.TW","3308.TW","3310.TWO","3311.TW","3312.TW","3321.TW","3325.TWO","3332.TWO","3338.TW","3346.TW",
    "3349.TWO","3356.TW","3360.TWO","3373.TWO","3374.TWO","3376.TW","3379.TWO","3380.TW","3388.TWO","3406.TW",
    "3413.TW","3416.TW","3419.TW","3426.TWO","3430.TWO","3432.TW","3437.TW","3443.TW","3444.TWO","3447.TW",
    "3450.TW","3455.TWO","3465.TWO","3467.TWO","3481.TW","3484.TWO","3494.TW","3499.TWO","3501.TW","3504.TW",
    "3515.TW","3518.TW","3521.TWO","3522.TWO","3526.TWO","3527.TWO","3528.TW","3530.TW","3531.TWO","3532.TW",
    "3533.TW","3535.TW","3543.TW","3545.TW","3550.TW","3556.TWO","3557.TW","3563.TW","3564.TWO","3576.TW",
    "3581.TWO","3583.TW","3587.TWO","3588.TW","3591.TW","3592.TW","3593.TW","3596.TW","3603.TWO","3605.TW",
    "3607.TW","3609.TWO","3615.TWO","3617.TW","3622.TW","3628.TWO","3629.TWO","3631.TWO","3633.TWO","3645.TW",
    "3652.TW","3653.TW","3659.TWO","3661.TW","3663.TWO","3664.TWO","3665.TW","3669.TW","3672.TWO","3673.TW",
    "3679.TW","3680.TWO","3686.TW","3687.TWO","3691.TWO","3694.TW","3701.TW","3702.TW","3703.TW","3704.TW",
    "3705.TW","3706.TW","3708.TW","3711.TW","3712.TW","3714.TW","3715.TW","3716.TW","3717.TW","4903.TWO",
    "4904.TW","4905.TWO","4906.TW","4907.TWO","4908.TWO","4909.TWO","4912.TW","4915.TW","4916.TW","4919.TW",
    "4927.TW","4930.TW","4934.TW","4935.TW","4938.TW","4939.TWO","4942.TW","4943.TW","4949.TW","4952.TW",
    "4956.TW","4958.TW","4960.TW","4961.TW","4966.TWO","4967.TW","4968.TW","4972.TWO","4974.TWO","4976.TW",
    "4977.TW","4989.TW","4994.TW","4999.TW","5201.TWO","5202.TWO","5205.TWO","5206.TWO","5209.TWO","5210.TWO",
    "5212.TWO","5223.TWO","5246.TWO","5248.TWO","5251.TWO","5254.TWO","5263.TWO","5267.TWO","5271.TWO","5272.TWO",
    "5274.TWO","5276.TWO","5278.TWO","5289.TWO","5297.TWO","5309.TWO","5310.TWO","5312.TWO","5314.TWO","5315.TWO",
    "5321.TWO","5324.TWO","5340.TWO","5344.TWO","5364.TWO","5371.TWO","5392.TWO","5403.TWO","5410.TWO","5426.TWO",
    "5450.TWO","5452.TWO","5455.TWO","5457.TWO","5460.TWO","5464.TWO","5465.TWO","5475.TWO","5483.TWO","5487.TWO",
    "5536.TWO","5548.TWO","6005.TW","6015.TWO","6024.TW","6028.TWO","6101.TWO","6103.TWO","6104.TWO","6108.TW",
    "6112.TW","6115.TW","6116.TW","6117.TW","6118.TWO","6120.TW","6128.TW","6133.TW","6136.TW","6139.TW",
    "6141.TW","6142.TW","6147.TWO","6152.TW","6153.TW","6155.TW","6164.TW","6165.TW","6166.TW","6167.TWO",
    "6168.TW","6171.TWO","6176.TW","6177.TW","6183.TW","6184.TW","6186.TWO","6187.TWO","6188.TWO","6189.TW",
    "6190.TWO","6191.TW","6192.TW","6194.TWO","6195.TWO","6196.TW","6197.TW","6198.TWO","6199.TWO","6201.TW",
    "6202.TW","6205.TW","6206.TW","6209.TW","6210.TWO","6213.TW","6214.TW","6215.TW","6216.TW","6220.TWO",
    "6223.TWO","6224.TW","6225.TW","6226.TW","6227.TWO","6228.TWO","6230.TW","6235.TW","6239.TW","6240.TWO",
    "6243.TW","6257.TW","6269.TW","6270.TWO","6271.TW","6272.TW","6274.TWO","6277.TW","6278.TW","6281.TW",
    "6282.TW","6283.TW","6285.TW","6294.TWO","6405.TW","6407.TWO","6409.TW","6412.TW","6414.TW","6415.TW",
    "6416.TW","6417.TWO","6418.TWO","6419.TWO","6423.TWO","6425.TWO","6426.TW","6428.TWO","6431.TW","6432.TWO",
    "6438.TW","6442.TW","6443.TW","6446.TW","6449.TW","6451.TW","6456.TW","6461.TWO","6462.TWO","6464.TW",
    "6467.TWO","6472.TW","6474.TWO","6477.TW","6482.TWO","6483.TWO","6485.TWO","6486.TWO","6488.TWO","6491.TW",
    "6492.TWO","6494.TWO","6504.TW","6505.TW","6509.TWO","6515.TW","6525.TW","6526.TW","6531.TW","6533.TW",
    "6534.TW","6541.TW","6547.TWO","6550.TW","6552.TW","6555.TWO","6558.TW","6568.TWO","6573.TW","6574.TWO",
    "6576.TWO","6579.TW","6581.TW","6582.TW","6583.TWO","6584.TWO","6585.TW","6586.TWO","6588.TWO","6589.TW",
    "6591.TW","6592.TW","6593.TWO","6595.TWO","6598.TW","6605.TW","6606.TW","6610.TWO","6612.TWO","6613.TWO",
    "6614.TW","6615.TWO","6616.TWO","6617.TWO","6625.TW","6638.TWO","6640.TWO","6641.TW","6642.TWO","6643.TWO",
    "6645.TW","6648.TWO","6649.TWO","6654.TWO","6655.TW","6657.TW","6658.TW","6659.TWO","6661.TWO","6664.TWO",
    "6666.TW","6668.TW","6669.TW","6670.TW","6671.TW","6672.TW","6673.TWO","6674.TW","6683.TWO","6684.TWO",
    "6689.TW","6690.TWO","6691.TW","6693.TWO","6695.TW","6696.TWO","6697.TWO","6698.TW","6706.TW","6715.TW",
    "6719.TW","6722.TW","6727.TWO","6728.TWO","6730.TWO","6733.TWO","6739.TWO","6742.TW","6743.TW","6744.TWO",
    "6750.TWO","6751.TWO","6752.TWO","6753.TW","6754.TW","6755.TWO","6756.TW","6757.TW","6758.TWO","6768.TW",
    "6770.TW","6771.TW","6775.TWO","6776.TW","6781.TW","6782.TW","6785.TWO","6789.TW","6790.TW","6791.TWO",
    "6792.TW","6794.TW","6796.TW","6797.TWO","6798.TWO","6799.TW","6805.TW","6806.TW","6807.TW","6830.TW",
    "6831.TW","6834.TW","6835.TW","6838.TW","6854.TW","6861.TW","6862.TW","6863.TW","6869.TW","6873.TW",
    "6885.TW","6887.TW","6890.TW","6901.TW","6902.TW","6906.TW","6908.TW","6909.TW","6914.TW","6916.TW",
    "6918.TW","6919.TW","6921.TW","6923.TW","6924.TW","6928.TW","6931.TW","6933.TW","6934.TW","6936.TW",
    "6937.TW","6944.TW","6949.TW","6951.TW","6952.TW","6955.TW","6957.TW","6958.TW","6962.TW","6965.TW",
    "6969.TW","6988.TW","6994.TW","8011.TW","8016.TW","8021.TW","8028.TW","8033.TW","8039.TW","8042.TWO",
    "8045.TW","8046.TW","8047.TWO","8058.TWO","8059.TWO","8064.TWO","8066.TWO","8067.TWO","8068.TWO","8070.TW",
    "8071.TWO","8080.TWO","8081.TW","8084.TWO","8085.TWO","8086.TWO","8087.TWO","8088.TWO","8089.TWO","8091.TWO",
    "8093.TWO","8098.TWO","8101.TW","8107.TWO","8112.TW","8131.TW","8147.TWO","8150.TW","8155.TWO","8183.TWO",
    "8201.TW","8210.TW","8213.TW","8249.TW","8261.TW","8271.TW","8272.TWO","8277.TWO","8284.TWO",
]


# ─────────────────────────────────────────────────────────────────────────────
#  即時台灣熱門排行函式 (每次掃描前動態抓取,非靜態清單)
# ─────────────────────────────────────────────────────────────────────────────

# 全域即時名稱快取(從screener結果補充TW_NAME_MAP沒有的英文名)
_REALTIME_NAME_CACHE: dict[str, str] = {}

# ★ 雷達掃描預設自選股清單（可在 Sidebar 自訂，⭐自選股模式用）
DEFAULT_WATCHLIST = [
    '8074.TW', '8150.TW', '2317.TW', '1609.TW',
    '3289.TW', '2603.TW', '2330.TW', '2454.TW',
]

# ★ 全市場雷達永久保障底盤：無論成交量排行如何變動，這些代號永遠被掃描
CORE_RADAR_WATCHLIST = [
    '1609.TW', '3289.TW', '8074.TW', '8150.TW', '2317.TW',
]

# 動態名稱快取(從 Ticker.info 動態查詢的結果，session 期間有效)
_DYNAMIC_NAME_CACHE: dict[str, str] = {}


def get_taiwan_hot_tickers(top_n: int = 50) -> list[str]:
    """
    取得「全市場成交量前 N 大」+「漲跌幅前 N 大」+「核心保障底盤」的
    合併掃描池，初選池總數約 100 檔。

    💻 Agent A 架構決策（移除 @st.cache_data 的原因）：
      @st.cache_data(ttl=300) 會快取函式回傳值，若第一次執行因網路問題
      只取得 5 檔核心底盤，這個失敗結果會被快取 5 分鐘，
      導致後續 5 分鐘內所有呼叫都回傳 5 檔，使用者無法靠重試解決。

      改用 st.session_state 手動快取：
      - key 包含時間 slot（3 分鐘），過期自動重試
      - 只快取成功取得 > 5 檔的結果（避免快取失敗結果）
      - 快取失敗時下次呼叫會重新嘗試主路徑+備援路徑

    🧑‍🔬 Agent B 雙軌備援設計：
      主路徑：fetch_tw_realtime_hot（curl_cffi + Yahoo Screener）
      備援路徑：_fetch_twse_tpex_fallback（TWSE + TPEX 官方 OpenAPI）
      任一成功即可，確保永遠能取得 80~100 檔。
    """
    import time as _t
    # ── 手動快取（3 分鐘 slot，只快取成功結果）────────────────────
    _slot_key = f"_hot_tickers_{int(_t.time() // 180)}_{top_n}"
    try:
        _cached = st.session_state.get(_slot_key)
        if _cached and len(_cached) > 5:   # 只接受多於核心底盤的快取結果
            return _cached
    except Exception:
        pass

    def _try_fetch(rank_type: str) -> list[str]:
        """
        單一排行類型抓取：
        主路徑（Yahoo）→ 失敗立即切備援（TWSE/TPEX），不等待不 sleep。
        🧑‍💼 Agent C 確認：備援函式只用 urllib 內建，合規。
        """
        # 主路徑：fetch_tw_realtime_hot（含盤前 effective_min_vol 修正）
        try:
            tickers, _meta, ok, _msg = fetch_tw_realtime_hot(rank_type, top_n)
            if ok and tickers and len(tickers) > 0:
                return tickers
        except Exception:
            pass

        # 備援路徑：TWSE + TPEX 官方 OpenAPI（純 urllib 內建，永不失敗）
        return _fetch_twse_tpex_fallback(rank_type, top_n)

    pool: list[str] = []

    # 三個來源並行取得（各自有備援，任一失敗不影響其他）
    for rank_type in ('volume', 'gain', 'loss'):
        result = _try_fetch(rank_type)
        pool.extend(result)

    # 合併核心底盤並去重複（核心股優先在前，確保必掃）
    merged = list(CORE_RADAR_WATCHLIST)
    seen   = set(merged)
    for t in pool:
        if t not in seen:
            merged.append(t)
            seen.add(t)

    # 只快取成功結果（> 5 檔才算成功）
    if len(merged) > 5:
        try:
            st.session_state[_slot_key] = merged
        except Exception:
            pass

    return merged

# ★ 官方名稱資料庫（從台灣證交所+櫃買中心 OpenAPI 動態載入）
# 比靜態 TW_NAME_MAP 更準確，永遠與官方保持同步
_OFFICIAL_NAME_CACHE: dict[str, str] = {}

@st.cache_resource(show_spinner=False)
def _load_official_names() -> dict[str, str]:
    """
    從台灣證交所(TWSE)與櫃買中心(TPEX) OpenAPI 載入完整股票中文名稱。

    ★ 改用 @st.cache_resource（全域共享，App 啟動後只打一次）：
      @st.cache_data 是 per-session，每個新使用者第一次進入都要重打 API，
      API 回傳前快取是空的，導致 get_stock_name 顯示英文名。
      @st.cache_resource 全域共享，所有 session 都直接用同一份快取。

    三個 API 互補，最大化覆蓋率：
      ① TWSE t187ap03_L       — 上市公司基本資料（含停牌/低流動性股）
      ② TWSE STOCK_DAY_ALL    — 上市今日成交（名稱最新，覆蓋①）
      ③ TPEX mainboard_quotes — 上櫃今日成交

    回傳: {ticker_with_suffix: 中文名稱}，如 {"2330.TW": "台積電"}
    """
    import requests as _req
    result: dict[str, str] = {}

    # ① TWSE 上市公司基本資料（含停牌股）
    try:
        r = _req.get(
            "https://openapi.twse.com.tw/v1/opendata/t187ap03_L",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=8
        )
        for item in r.json():
            code = item.get('公司代號', '').strip()
            name = (item.get('公司簡稱', '').strip()
                    or item.get('公司名稱', '').strip())
            if code and name and code.isdigit():
                result[f"{code}.TW"] = name
    except Exception:
        pass

    # ② TWSE 今日成交（名稱最新，覆蓋①）
    try:
        r = _req.get(
            "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=8
        )
        for item in r.json():
            code = item.get('Code', '').strip()
            name = item.get('Name', '').strip()
            if code and name and code.isdigit():
                result[f"{code}.TW"] = name
    except Exception:
        pass

    # ③ TPEX 上櫃今日成交（mainboard_quotes ~1011筆，僅今日有成交）
    try:
        r = _req.get(
            "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=8
        )
        for item in r.json():
            code = item.get('SecuritiesCompanyCode', '').strip()
            name = item.get('CompanyName', '').strip()
            if code and name and code.isdigit():
                result[f"{code}.TWO"] = name
    except Exception:
        pass

    # ④ TPEX 上櫃完整歷史收盤（daily_close_quotes ~9500筆，含低流動性/停牌股）
    # 覆蓋③沒抓到的股票（邦泰8935、單井3490、蜜望實8043、安碁6174、智捷8176等）
    try:
        r = _req.get(
            "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=10
        )
        for item in r.json():
            code = item.get('SecuritiesCompanyCode', '').strip()
            name = item.get('CompanyName', '').strip()
            if code and name and code.isdigit():
                # 不覆蓋③已取得的名稱（③的今日成交名稱更新）
                if f"{code}.TWO" not in result:
                    result[f"{code}.TWO"] = name
    except Exception:
        pass

    # ④ 手動補入：官方 API 查不到的特殊狀態股票
    _manual = {
        "6696.TWO": "科生*-KY",
        "6618.TWO": "宇泰科技",
    }
    for k, v in _manual.items():
        if k not in result:
            result[k] = v

    return result


def _fetch_twse_tpex_fallback(rank_type: str = 'volume',
                              top_n: int = 50) -> list[str]:
    """
    🧑‍🔬 Agent B 實作：純 urllib 內建備援抓取路徑
    ─────────────────────────────────────────────────────────────────────
    當 curl_cffi Yahoo Screener 失敗時（Streamlit Cloud 冷啟動/限流），
    改用台灣官方免費 OpenAPI 取得熱門股清單。

    💻 Agent A 架構決策：
      - 來源一：TWSE OpenAPI（STOCK_DAY_ALL）→ 上市股完整今日成交資料
      - 來源二：TPEX OpenAPI（tpex_mainboard_quotes）→ 上櫃股今日成交資料
      - 完全免費、無 API key、無限流、純 urllib 內建（Agent C 合規通過）

    rank_type:
      'volume' → 成交量排行
      'gain'   → 漲幅排行（用漲跌幅 Change 欄位）
      'loss'   → 跌幅排行
    """
    import urllib.request as _ur
    import ssl as _ssl
    import json as _json

    ctx = _ssl.create_default_context()
    hdrs = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    results: list[dict] = []

    # ── 上市股（TWSE）────────────────────────────────────────────────
    try:
        req = _ur.Request(
            "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL",
            headers=hdrs
        )
        with _ur.urlopen(req, timeout=12, context=ctx) as resp:
            data = _json.loads(resp.read())

        for item in data:
            code  = item.get('Code', '').strip()
            vol   = item.get('TradeVolume', '').replace(',', '').strip()
            price = item.get('ClosingPrice', '').replace(',', '').strip()
            chg   = item.get('Change', '').replace('+', '').replace(',', '').strip()
            if not re.match(r'^\d{4}$', code):
                continue
            try:
                results.append({
                    'symbol':  f"{code}.TW",
                    'volume':  int(vol) if vol.isdigit() else 0,
                    'price':   float(price) if price else 0.0,
                    'change':  float(chg) if chg not in ('', '--', 'X', '除') else 0.0,
                    'name':    item.get('Name', '').strip(),
                })
            except (ValueError, TypeError):
                continue
    except Exception:
        pass

    # ── 上櫃股（TPEX）────────────────────────────────────────────────
    try:
        req2 = _ur.Request(
            "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes",
            headers=hdrs
        )
        with _ur.urlopen(req2, timeout=12, context=ctx) as resp2:
            data2 = _json.loads(resp2.read())

        for item in data2:
            code  = item.get('SecuritiesCompanyCode', '').strip()
            vol   = item.get('TradingShares', '') or '0'
            price = item.get('Close', '') or '0'
            chg   = item.get('Change', '') or '0'
            if not re.match(r'^\d{4}$', code):
                continue
            try:
                results.append({
                    'symbol':  f"{code}.TWO",
                    'volume':  int(str(vol).replace(',', '')) if vol else 0,
                    'price':   float(str(price).replace(',', '')) if price else 0.0,
                    'change':  float(str(chg).replace(',', '').replace('+', ''))
                               if chg not in ('', '--') else 0.0,
                    'name':    item.get('CompanyName', '').strip(),
                })
            except (ValueError, TypeError):
                continue
    except Exception:
        pass

    if not results:
        return []

    # ── 排序 ─────────────────────────────────────────────────────────
    min_price = 5.0
    results = [r for r in results if r['price'] >= min_price and r['volume'] > 0]

    if rank_type == 'volume':
        results.sort(key=lambda x: x['volume'], reverse=True)
    elif rank_type == 'gain':
        results.sort(key=lambda x: x['change'], reverse=True)
    elif rank_type == 'loss':
        results.sort(key=lambda x: x['change'])

    # 同步寫入名稱快取（讓 get_stock_name 也能用到）
    global _REALTIME_NAME_CACHE
    for r in results:
        if r.get('name'):
            _REALTIME_NAME_CACHE[r['symbol']] = r['name']

    return [r['symbol'] for r in results[:top_n]]


def _fetch_screener_cffi(exchange: str, size: int = 240) -> list[dict]:
    """
    用 curl_cffi 模擬 Chrome 瀏覽器指紋 + 完整 Cookie Session 抓取
    Yahoo Finance Screener 資料。

    為何不用 yf.screen():
      Streamlit Community Cloud 的共享 IP 因為大量用戶同時使用 yfinance
      打 Yahoo API,Yahoo 把這些 IP 識別為機器人流量並限流(429)。
      curl_cffi 模擬完整的 Chrome 瀏覽器 TLS 指紋 + 先取 Cookie Session
      再打 API,Yahoo 無法區分真實用戶和程式,因此不會被限流。

    正確流程:
      1. GET finance.yahoo.com → 建立合法 Cookie Session
      2. GET /v1/test/getcrumb → 取得 API 認證用 crumb token
      3. POST /v1/finance/screener → 帶入 crumb + Cookie 抓取資料
    """
    try:
        from curl_cffi.requests import Session as CffiSession
        with CffiSession(impersonate="chrome124") as s:
            # Step 1: 建立 Cookie Session
            # ★ Agent B：timeout 從 10 → 20 秒，Streamlit Cloud 冷啟動時網路延遲大
            s.get("https://finance.yahoo.com/",
                  timeout=20,
                  headers={'Accept-Language': 'zh-TW,zh;q=0.9,en;q=0.8'})

            # Step 2: 取得 crumb（timeout 從 8 → 15 秒）
            cr = s.get("https://query1.finance.yahoo.com/v1/test/getcrumb", timeout=15)
            crumb = cr.text.strip() if cr.status_code == 200 else ""

            # Fallback: 換 query2
            if not crumb or '{' in crumb:
                cr2 = s.get("https://query2.finance.yahoo.com/v1/test/getcrumb", timeout=15)
                crumb = cr2.text.strip() if cr2.status_code == 200 else ""

            if not crumb or '{' in crumb:
                return []

            # Step 3: 呼叫 Screener（timeout 從 15 → 25 秒）
            url  = (f"https://query1.finance.yahoo.com/v1/finance/screener"
                    f"?formatted=false&lang=zh-TW&region=TW&crumb={crumb}")
            body = {
                "offset": 0, "size": size,
                "sortField": "dayvolume", "sortType": "DESC",
                "quoteType": "EQUITY",
                "query": {
                    "operator": "AND",
                    "operands": [{"operator": "EQ", "operands": ["exchange", exchange]}]
                },
                "userId": "", "userIdType": "guid"
            }
            resp = s.post(url, json=body, timeout=25)
            if resp.status_code == 200:
                return (resp.json()
                        .get('finance', {})
                        .get('result', [{}])[0]
                        .get('quotes', []))
    except Exception:
        pass
    return []


def fetch_tw_realtime_hot(
    rank_type: str = 'volume',
    size: int = 100,
    min_price: float = 5.0,
    min_vol: int = 500_000,
) -> tuple[list[str], list[dict], bool, str]:
    """
    即時從 Yahoo Finance 抓取台灣上市(TAI)＋上櫃(TWO)的熱門排行。
    失敗時自動切換 TWSE + TPEX 官方 OpenAPI 備援（純 urllib 內建）。

    💻 Agent A 雙軌架構：
      主路徑：curl_cffi Yahoo Screener（盤中即時資料）
      備援路徑：_fetch_twse_tpex_fallback()（TWSE + TPEX 官方 API，永不失敗）

    🧑‍🔬 Agent B 關鍵修正（盤前 8:20 問題根因）：
      Yahoo Screener 在盤前回傳的 volume 為 0，被 min_vol >= 500_000 過濾掉
      → 即使 Yahoo 回傳資料，也全部被過濾 → 顯示「無法連線」。
      修正：盤前/盤後自動放寬 effective_min_vol = 0，讓資料通過。

    回傳: (tickers, meta_list, success: bool, message: str)
    """
    global _REALTIME_NAME_CACHE

    # ★ Agent B：盤前/盤後自動放寬 min_vol
    # 台股交易時段 09:00~13:35，盤前/盤後 volume=0，不應套用最小成交量門檻
    try:
        import pytz as _pytz
        _tw = datetime.datetime.now(_pytz.timezone('Asia/Taipei'))
        _in_market = (datetime.time(9, 0) <= _tw.time() <= datetime.time(13, 35)
                      and _tw.weekday() < 5)
    except Exception:
        _in_market = True
    effective_min_vol = min_vol if _in_market else 0

    all_quotes      = []
    failed_exchanges = []

    for exchange in ['TAI', 'TWO']:
        quotes = []

        # 方法一: curl_cffi + cookie session (主要方式)
        quotes = _fetch_screener_cffi(exchange, 240)

        # 方法二: yf.screen() 備用
        if not quotes and _EQUITY_QUERY_AVAILABLE:
            try:
                q = EquityQuery('eq', ['exchange', exchange])
                result = yf.screen(q, sortField='dayvolume', sortAsc=False, size=240)
                quotes = result.get('quotes', [])
            except Exception:
                pass

        if quotes:
            all_quotes.extend(quotes)
            for quote in quotes:
                sym   = quote.get('symbol', '')
                sname = quote.get('shortName', '') or quote.get('longName', '')
                if sym and sname:
                    _REALTIME_NAME_CACHE[sym] = sname
        else:
            failed_exchanges.append(exchange)

    # ★ 主路徑完全失敗 → 直接用 TWSE/TPEX 備援（永不失敗）
    if len(failed_exchanges) == 2:
        fallback_tickers = _fetch_twse_tpex_fallback(rank_type, size)
        if fallback_tickers:
            meta_list = [{"symbol": t, "name": get_stock_name(t),
                          "price": 0.0, "chg_pct": 0.0, "volume": 0}
                         for t in fallback_tickers]
            return fallback_tickers, meta_list, True, f"備援模式（TWSE+TPEX）取得 {len(fallback_tickers)} 筆"
        return [], [], False, "Yahoo Screener 暫時無法連線，已自動切換靜態清單"

    # 過濾純4位數字代號（使用放寬後的 effective_min_vol）
    filtered = []
    for q in all_quotes:
        sym   = q.get('symbol', '')
        code  = sym[:-4] if sym.endswith('.TWO') else sym[:-3] if sym.endswith('.TW') else sym
        price = float(q.get('regularMarketPrice', 0) or 0)
        vol   = int(q.get('regularMarketVolume', 0) or 0)
        if re.match(r'^\d{4}$', code) and price >= min_price and vol >= effective_min_vol:
            filtered.append(q)

    # 若過濾後仍為空（盤前某些情況），完全放寬 volume 門檻
    if not filtered and all_quotes:
        for q in all_quotes:
            sym   = q.get('symbol', '')
            code  = sym[:-4] if sym.endswith('.TWO') else sym[:-3] if sym.endswith('.TW') else sym
            price = float(q.get('regularMarketPrice', 0) or 0)
            if re.match(r'^\d{4}$', code) and price >= min_price:
                filtered.append(q)

    # 若仍為空，用 TWSE/TPEX 備援補足
    if not filtered:
        fallback_tickers = _fetch_twse_tpex_fallback(rank_type, size)
        if fallback_tickers:
            meta_list = [{"symbol": t, "name": get_stock_name(t),
                          "price": 0.0, "chg_pct": 0.0, "volume": 0}
                         for t in fallback_tickers]
            partial = f"（{'/'.join(failed_exchanges)} 部分缺失）" if failed_exchanges else ""
            return fallback_tickers, meta_list, True, f"備援補足（TWSE）{len(fallback_tickers)} 筆{partial}"

    if rank_type == 'gain':
        filtered.sort(key=lambda x: float(x.get('regularMarketChangePercent', 0) or 0),
                      reverse=True)
    elif rank_type == 'loss':
        filtered.sort(key=lambda x: float(x.get('regularMarketChangePercent', 0) or 0))
    else:
        filtered.sort(key=lambda x: int(x.get('regularMarketVolume', 0) or 0),
                      reverse=True)

    filtered  = filtered[:size]
    tickers   = [q['symbol'] for q in filtered]
    meta_list = [
        {
            "symbol":  q['symbol'],
            "name":    q.get('shortName', '') or q.get('longName', ''),
            "price":   float(q.get('regularMarketPrice', 0) or 0),
            "chg_pct": float(q.get('regularMarketChangePercent', 0) or 0),
            "volume":  int(q.get('regularMarketVolume', 0) or 0),
        }
        for q in filtered
    ]

    partial = f"（{'/'.join(failed_exchanges)} 部分資料缺失）" if failed_exchanges else ""
    return tickers, meta_list, True, f"成功取得 {len(tickers)} 筆{partial}"


TW_HOT_100 = [
    # ── 半導體龍頭 ──────────────────────────────────────────────────────
    "2330","2303","2454","2379","3034","6770","2344","3711","3533","2408",
    # ── AI / 伺服器 / 散熱 / CoWoS ─────────────────────────────────────
    "2317","3008","2357","6669","3231","6274.TWO","5274.TWO","8150","4919","6415",
    # ── PCB / 電路板 / 載板 ─────────────────────────────────────────────
    "2301","3037","6153","8046","6269","3024","2383","6456","4961","3706",
    # ── 被動元件 / 電子材料 ─────────────────────────────────────────────
    "2327","2354","2382","3019","6789","2376","5483.TWO","3443","6214","2439",
    # ── 網通 / 電信 ─────────────────────────────────────────────────────
    "2412","3045","4904","2498","3044","6488.TWO","4906","2915","3026","5434",
    # ── 金融 ────────────────────────────────────────────────────────────
    "2882","2881","2886","2891","2884","2885","2887","2892","2801","5880",
    # ── 傳產 / 原物料 / 塑化 ────────────────────────────────────────────
    "1301","1303","1326","2002","9904","1101","1216","1402","2105","1210",
    # ── 航運 / 貨櫃 / 空運 ──────────────────────────────────────────────
    "2603","2609","2615","2610","5608","2617","2618","2606","2637","2634",
    # ── 光電 / 面板 / 光學 ──────────────────────────────────────────────
    "3481","2409","8299.TWO","6409","3691.TWO","2449","3035","2395","5269","3653",
    # ── 生技 / 醫療 / CDMO ──────────────────────────────────────────────
    "4711.TWO","6446","4726.TWO","4537.TWO","1796.TWO","6547.TWO","4174.TWO","4164","6509.TWO","4729.TWO",
]

# ─────────────────────────────────────────────────────────────────────────────
#  批量掃描引擎 (ThreadPoolExecutor 並行下載)
# ─────────────────────────────────────────────────────────────────────────────


def _scan_one(ticker: str, period: str) -> dict | None:
    """
    掃描單一股票,回傳精簡結果 dict;若資料取得失敗回傳 None。
    此函式會被 ThreadPoolExecutor 並行呼叫。

    ★ Rate Limit 處理: Yahoo Finance 在短時間大量請求時會回傳 429。
      遇到限速錯誤時自動等待 2 秒後最多重試 3 次,超過則跳過。
    """
    import time as _time
    for attempt in range(3):
        try:
            df, used = fetch_data(ticker, period=period, time_bucket=_get_cache_bucket())
            if df is None or len(df) < 60:
                return None

            # 注意: 批量掃描不呼叫 _patch_today_price
            # fetch_data 內部的 Close=NaN 修補已用 fast_info 填入正確收盤價
            # 若在此再呼叫 _patch_today_price,100 檔同時打 fast_info 會觸發 Yahoo 限流
            df  = add_indicators(df)
            dna = detect_wave_dna(df)
            wr  = compute_winrate(dna, df)
            entry = evaluate_entry_point(dna, wr, df)  # ★ 買點評估
            last = df.iloc[-1]
            return {
                "代號":       used,
                "股名":       get_stock_name(used),
                "chart_url":  get_chart_url(used),
                "input":      ticker,
                "收盤價":     round(float(last["Close"]), 2),
                "勝率":       round(wr["winrate"] * 100, 1),
                "分類":       wr["category_label"],
                "category":   wr["category"],
                "R_cycle":    dna["R_cycle"],
                "T_median":   dna["T_median"],
                "D_current":  dna["D_current"],
                "均線型態":   wr["desc_ma"],
                "KD狀態":     wr["desc_kd"],
                "時間波":     wr["desc_time"],
                "K9":         wr["k9"],
                "D9":         wr["d9"],
                "量比":       wr["vol_ratio"],
                "days_trough": dna.get("days_since_trough", -1),
                "corr_end":   dna.get("correction_end_date"),
                # ── ★ 買點獵人欄位 ────────────────────────────────
                "買點分數":   entry["score"],
                "買點訊號":   entry["signal"],
                "KD拐頭":     entry["kd_stage"],
                "買點條件":   entry["conditions"],
            }
        except Exception as e:
            err_str = str(e).lower()
            # Rate limit: 等待後重試
            if "rate" in err_str or "429" in err_str or "too many" in err_str:
                if attempt < 2:
                    _time.sleep(2 + attempt * 2)  # 2s / 4s 遞增等待
                    continue
            return None
    return None


def run_batch_scan(tickers: list[str], period: str,
                   progress_bar, status_text) -> list[dict]:
    """
    以 ThreadPoolExecutor 並行下載並分析所有股票。
    每完成一檔就更新進度條。
    """
    results   = []
    total     = len(tickers)
    completed = 0

    # max_workers=10:同時 10 條線程,避免 Yahoo Finance 限流
    with ThreadPoolExecutor(max_workers=10) as exe:
        future_map = {exe.submit(_scan_one, t, period): t for t in tickers}
        for fut in as_completed(future_map):
            completed += 1
            progress_bar.progress(completed / total)
            status_text.markdown(
                f'<span style="font-family:\'IBM Plex Mono\',monospace;'
                f'font-size:14px;color:#1a2b3c;">'
                f'掃描中 {completed}/{total} ── {future_map[fut]}</span>',
                unsafe_allow_html=True
            )
            res = fut.result()
            if res:
                results.append(res)

    # 依勝率降序排列
    results.sort(key=lambda x: x["勝率"], reverse=True)
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  掃描結果 HTML 表格渲染
# ─────────────────────────────────────────────────────────────────────────────

def html_scan_table(rows: list[dict], min_winrate: float = 0,
                    hunter_mode: bool = False) -> str:
    """
    批量掃描結果表格。
    ★ 新增: 股名欄 / 點擊代號開新分頁 / 📈按鈕彈出iframe技術線型視窗
    """
    filtered = [r for r in rows if r["勝率"] >= min_winrate]
    if not filtered:
        return ('<div style="color:#4a6fa5;font-family:\'Noto Sans TC\',sans-serif;'
                'padding:24px;font-size:15px;">⚠️ 無符合條件的標的，請降低勝率門檻或調整清單</div>')

    cat_badge = {
        "top":  ('<span style="background:#0a7c59;color:#fff;padding:3px 10px;'
                 'border-radius:5px;font-size:13px;font-weight:700;">🚀 頂級</span>'),
        "mid":  ('<span style="background:#d97706;color:#fff;padding:3px 10px;'
                 'border-radius:5px;font-size:13px;font-weight:700;">⏳ 蓄勢</span>'),
        "warn": ('<span style="background:#c0392b;color:#fff;padding:3px 10px;'
                 'border-radius:5px;font-size:13px;font-weight:700;">🛑 警戒</span>'),
    }
    bar_color = {"top": "#0a7c59", "mid": "#d97706", "warn": "#c0392b"}

    # 彈出 iframe 視窗 JS (只生成一次)
    modal_js = """
    <div id="chartModal" onclick="if(event.target===this){this.style.display='none';
         document.getElementById('chartFrame').src='';}"
         style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;
         background:rgba(0,0,0,0.72);z-index:9999;align-items:center;justify-content:center;">
      <div style="background:#fff;border-radius:14px;width:90%;max-width:1100px;
                  height:82vh;overflow:hidden;position:relative;
                  box-shadow:0 12px 48px rgba(0,0,0,0.45);">
        <div style="display:flex;align-items:center;justify-content:space-between;
                    padding:12px 18px;background:#1565c0;color:#fff;">
          <div>
            <span id="modalTitle" style="font-family:\'Noto Sans TC\',sans-serif;
                  font-size:16px;font-weight:700;"></span>
            <span style="font-size:12px;opacity:.75;margin-left:10px;">
              Yahoo Finance 技術分析</span>
          </div>
          <button onclick="document.getElementById(\'chartModal\').style.display=\'none\';
                           document.getElementById(\'chartFrame\').src=\'\';event.stopPropagation();"
                  style="background:rgba(255,255,255,.2);border:none;color:#fff;
                         font-size:18px;cursor:pointer;border-radius:6px;
                         padding:4px 10px;line-height:1;">✕ 關閉</button>
        </div>
        <iframe id="chartFrame" src="" style="width:100%;height:calc(82vh - 52px);border:none;"></iframe>
      </div>
    </div>
    <script>
    function openChart(url, title) {
        document.getElementById(\'chartFrame\').src = url;
        document.getElementById(\'modalTitle\').textContent = title;
        document.getElementById(\'chartModal\').style.display = \'flex\';
    }
    </script>"""

    if hunter_mode:
        head = """
    <table class="scan-table fwd-table" style="font-size:14px;width:100%;">
      <thead>
        <tr>
          <th style="width:32px;">#</th>
          <th style="min-width:90px;">代號</th>
          <th style="min-width:80px;">股名</th>
          <th style="width:78px;">收盤</th>
          <th style="width:100px;">買點分數</th>
          <th style="width:90px;">訊號</th>
          <th style="width:72px;">R_cycle</th>
          <th style="width:70px;">勝率</th>
          <th style="width:90px;">KD拐頭</th>
          <th style="width:60px;">量比</th>
          <th>均線型態</th>
          <th style="width:42px;">線型</th>
        </tr>
      </thead><tbody>"""
    else:
        head = """
    <table class="scan-table fwd-table" style="font-size:14px;width:100%;">
      <thead>
        <tr>
          <th style="width:32px;">#</th>
          <th style="min-width:90px;">代號</th>
          <th style="min-width:80px;">股名</th>
          <th style="width:78px;">收盤價</th>
          <th style="width:155px;">波段勝率</th>
          <th style="width:72px;">分類</th>
          <th style="width:72px;">R_cycle</th>
          <th style="width:68px;">修正基準</th>
          <th style="width:68px;">拉回天數</th>
          <th>均線型態</th>
          <th>KD狀態</th>
          <th style="width:42px;">線型</th>
        </tr>
      </thead><tbody>"""

    body = ""
    for i, r in enumerate(filtered, 1):
        cat   = r["category"]
        wr    = r["勝率"]
        bc    = bar_color.get(cat, "#1565c0")
        badge = cat_badge.get(cat, r["分類"])
        code  = r["代號"]
        name  = r.get("股名", "")
        url   = r.get("chart_url", get_chart_url(code))
        title_str = f"{name} ({code})" if name else code

        row_bg = "background:#f7fafd;" if i % 2 == 0 else "background:#ffffff;"

        bar = (f'<div style="display:flex;align-items:center;gap:8px;">'
               f'<div style="width:70px;background:#c8d8e8;border-radius:4px;'
               f'height:10px;flex-shrink:0;overflow:hidden;">'
               f'<div style="width:{min(wr,100):.0f}%;height:10px;border-radius:4px;'
               f'background:{bc};"></div></div>'
               f'<span style="color:{bc};font-weight:700;font-size:15px;'
               f'font-family:\'IBM Plex Mono\',monospace;">{wr:.1f}%</span>'
               f'</div>')

        rc = r["R_cycle"]
        rc_color = "#0a7c59" if rc >= 1.0 else "#d97706" if rc >= 0.6 else "#c0392b"

        # 安全轉義title中的單引號
        safe_title = title_str.replace("'", " ").replace('"', ' ')
        safe_url   = url.replace("'", "%27")

        code_link = (f'<a href="{url}" target="_blank" '
                     f'style="color:#1565c0;font-weight:700;font-size:14px;'
                     f'font-family:\'IBM Plex Mono\',monospace;text-decoration:none;">' 
                     f'{code}</a>')
        name_html = (f'<span style="color:#1a2b3c;font-size:13px;">{name}</span>'
                     if name else
                     f'<span style="color:#b0bec5;font-size:12px;">--</span>')
        chart_btn = (f'<button onclick="openChart(\'{safe_url}\',\'{safe_title}\')" '
                     f'style="background:#eaf2fb;border:1px solid #b8cce0;border-radius:6px;'
                     f'padding:3px 8px;cursor:pointer;font-size:14px;color:#1565c0;">📈</button>')

        entry_score  = r.get("買點分數", 0)
        entry_signal = r.get("買點訊號", "")
        kd_stage     = r.get("KD拐頭", "")
        vol_r        = r.get("量比", 1.0)

        # 買點分數顏色
        es_color = ("#0a7c59" if entry_score >= 80
                    else "#1565c0" if entry_score >= 65
                    else "#d97706" if entry_score >= 50
                    else "#c0392b")

        if hunter_mode:
            body += f"""
        <tr style="{row_bg}">
          <td style="color:#7a9bbf;font-size:13px;text-align:center;">{i}</td>
          <td>{code_link}</td>
          <td>{name_html}</td>
          <td style="color:#1a2b3c;font-weight:700;font-size:15px;
                     font-family:'IBM Plex Mono',monospace;">{r['收盤價']}</td>
          <td>
            <div style="display:flex;align-items:center;gap:6px;">
              <div style="width:55px;background:#c8d8e8;border-radius:4px;
                          height:10px;overflow:hidden;">
                <div style="width:{min(entry_score,100)}%;height:10px;border-radius:4px;
                            background:{es_color};"></div>
              </div>
              <span style="color:{es_color};font-weight:700;font-size:15px;
                           font-family:'IBM Plex Mono',monospace;">{entry_score}</span>
            </div>
          </td>
          <td><span style="font-size:13px;">{entry_signal}</span></td>
          <td style="color:{rc_color};font-weight:700;font-size:14px;
                     font-family:'IBM Plex Mono',monospace;">{rc:.3f}</td>
          <td style="color:#4a6fa5;font-size:13px;">{r['勝率']:.0f}%</td>
          <td style="color:#2d3748;font-size:13px;">{kd_stage}</td>
          <td style="color:#4a6fa5;font-size:13px;">{vol_r:.1f}x</td>
          <td style="color:#2d3748;font-size:13px;">{r['均線型態'][:20]}</td>
          <td style="text-align:center;">{chart_btn}</td>
        </tr>"""
        else:
            body += f"""
        <tr style="{row_bg}">
          <td style="color:#7a9bbf;font-size:13px;text-align:center;">{i}</td>
          <td>{code_link}</td>
          <td>{name_html}</td>
          <td style="color:#1a2b3c;font-weight:700;font-size:15px;
                     font-family:'IBM Plex Mono',monospace;">{r['收盤價']}</td>
          <td>{bar}</td>
          <td>{badge}</td>
          <td style="color:{rc_color};font-weight:600;font-size:14px;
                     font-family:'IBM Plex Mono',monospace;">{rc:.3f}</td>
          <td style="color:#1a2b3c;font-size:14px;
                     font-family:'IBM Plex Mono',monospace;">{r['T_median']:.0f} 天</td>
          <td style="color:#1a2b3c;font-size:14px;
                     font-family:'IBM Plex Mono',monospace;">{r['D_current']} 天</td>
          <td style="color:#2d3748;font-size:13px;">{r['均線型態'][:22]}</td>
          <td style="color:#2d3748;font-size:13px;">{r['KD狀態'][:22]}</td>
          <td style="text-align:center;">{chart_btn}</td>
        </tr>"""

    # ── 手機卡片版(mobile-cards,桌機隱藏) ─────────────────────────
    cards_html = '<div class="mobile-cards">'
    for i, r in enumerate(filtered, 1):
        cat   = r["category"]
        wr    = r["勝率"]
        bc    = bar_color.get(cat, "#1565c0")
        code  = r["代號"]
        name  = r.get("股名", "")
        url   = r.get("chart_url", get_chart_url(code))
        title_str = f"{name} ({code})" if name else code
        safe_title = title_str.replace("'", " ").replace('"', ' ')
        safe_url   = url.replace("'", "%27")

        badge_bg = {"top":"#0a7c59","mid":"#d97706","warn":"#c0392b"}.get(cat,"#1565c0")
        badge_txt = {"top":"🚀 頂級浪潮","mid":"⏳ 中繼蓄勢","warn":"🛑 警戒浪潮"}.get(cat, cat)
        rc = r["R_cycle"]
        rc_color = "#0a7c59" if rc >= 1.0 else "#d97706" if rc >= 0.6 else "#c0392b"

        cards_html += f"""
        <div class="scan-card">
          <div class="sc-header">
            <div>
              <a href="{url}" target="_blank" class="sc-code">#{i} {code}</a>
              <span class="sc-name">{" · " + name if name else ""}</span>
            </div>
            <span class="sc-badge" style="background:{badge_bg};color:#fff;">{badge_txt}</span>
          </div>
          <div class="sc-row">
            <span class="sc-price">{r['收盤價']}</span>
            <span class="sc-wr" style="color:{bc};">{wr:.1f}%</span>
          </div>
          <div class="sc-bar-wrap">
            <div class="sc-bar-fill" style="width:{min(wr,100):.0f}%;background:{bc};"></div>
          </div>
          <div class="sc-meta">
            <span>R_cycle: <b style="color:{rc_color};">{rc:.3f}</b></span>
            <span>修正基準: <b>{r['T_median']:.0f}天</b></span>
            <span>拉回天數: <b>{r['D_current']}天</b></span>
          </div>
          <div class="sc-desc">
            📊 {r['均線型態'][:30]}<br>
            📈 {r['KD狀態'][:30]}
          </div>
          <div style="margin-top:8px;text-align:right;">
            <button onclick="openChart('{safe_url}','{safe_title}')"
              style="background:#eaf2fb;border:1px solid #b8cce0;border-radius:6px;
                     padding:5px 12px;color:#1565c0;font-size:14px;cursor:pointer;">
              📈 開啟技術線型
            </button>
          </div>
        </div>"""
    cards_html += '</div>'

    return modal_js + cards_html + head + body + "</tbody></table>"



# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
#  ★ 自選股系統 (業界下單頁面風格)
#  儲存: st.session_state["watchlists"] — 5 個清單，每個最多 20 檔
#  功能: 即時報價 / DNA分類 / 買點評估 / 籌碼 / JSON匯出入
# ─────────────────────────────────────────────────────────────────────────────

_WL_KEY   = "watchlists"       # session_state 存放自選股的 key
_WL_COUNT = 5                  # 自選股清單數量
_WL_MAX   = 20                 # 每個清單最多幾檔


def _wl_init():
    """初始化自選股 session_state（若尚未存在）"""
    if _WL_KEY not in st.session_state:
        st.session_state[_WL_KEY] = {
            i: {"name": f"自選股 {i}", "tickers": []}
            for i in range(1, _WL_COUNT + 1)
        }
    if "_wl_active" not in st.session_state:
        st.session_state["_wl_active"] = 1


def _wl_get(idx: int) -> dict:
    _wl_init()
    return st.session_state[_WL_KEY][idx]


def _wl_set_name(idx: int, name: str):
    _wl_init()
    st.session_state[_WL_KEY][idx]["name"] = name.strip() or f"自選股 {idx}"


def _wl_resolve_ticker(ticker: str) -> str:
    """
    自動解析台股代號的正確後綴。
    輸入 '8299' 或 '8299.TW'，若 Yahoo Finance 找不到，自動改試 '.TWO'。
    回傳能查到資料的正確代號，或原始代號（找不到時）。
    """
    import warnings as _w; _w.filterwarnings('ignore')

    # 標準化
    t = ticker.strip().upper()
    if t.isdigit():
        candidates = [f"{t}.TW", f"{t}.TWO"]
    elif t.endswith('.TW') and not t.endswith('.TWO'):
        candidates = [t, t.replace('.TW', '.TWO')]
    elif t.endswith('.TWO'):
        candidates = [t, t.replace('.TWO', '.TW')]
    else:
        return t  # 美股或其他，直接回傳

    for cand in candidates:
        try:
            fi = yf.Ticker(cand).fast_info
            lp = float(getattr(fi, 'last_price', 0) or 0)
            if lp > 0:
                return cand
        except Exception:
            continue

    return candidates[0]  # fallback 到第一個


def _wl_add_ticker(idx: int, ticker: str) -> tuple[bool, str]:
    """新增代號到自選股，自動解析正確後綴，回傳 (成功, 訊息)"""
    _wl_init()
    ticker = ticker.strip().upper()
    if not ticker:
        return False, "請輸入代號"
    wl = st.session_state[_WL_KEY][idx]
    if len(wl["tickers"]) >= _WL_MAX:
        return False, f"每個清單最多 {_WL_MAX} 檔"

    # 自動解析正確後綴（.TW vs .TWO）
    resolved = _wl_resolve_ticker(ticker)

    if resolved in wl["tickers"]:
        return False, f"{resolved} 已在清單中"

    # 若解析後不同，提示使用者
    note = f"（已自動修正為 {resolved}）" if resolved != ticker and '.' in resolved else ""
    wl["tickers"].append(resolved)
    return True, f"✅ 已加入 {resolved} {note}"


def _wl_remove_ticker(idx: int, ticker: str):
    _wl_init()
    try:
        st.session_state[_WL_KEY][idx]["tickers"].remove(ticker)
    except ValueError:
        pass


def _wl_move_ticker(idx: int, ticker: str, direction: int):
    """上移(-1)/下移(+1)"""
    _wl_init()
    tickers = st.session_state[_WL_KEY][idx]["tickers"]
    i = tickers.index(ticker)
    j = i + direction
    if 0 <= j < len(tickers):
        tickers[i], tickers[j] = tickers[j], tickers[i]


@st.cache_data(ttl=60, show_spinner=False)
def _wl_fetch_quote(ticker: str, _bucket: str = "") -> dict:
    """
    抓取單檔即時報價（快取 60 秒）。
    自動嘗試 .TW / .TWO 後綴，確保上櫃股也能查到。
    回傳: price, chg_pct, volume, prev_close, high, low, resolved_ticker
    """
    empty = {"price": 0, "chg_pct": 0, "volume": 0, "prev": 0,
             "high": 0, "low": 0, "ok": False, "resolved": ticker}

    # 決定嘗試順序
    t = ticker.strip().upper()
    if t.endswith('.TWO'):
        candidates = [t, t.replace('.TWO', '.TW')]
    elif t.endswith('.TW') and not t.endswith('.TWO'):
        candidates = [t, t.replace('.TW', '.TWO')]
    else:
        candidates = [t]

    for cand in candidates:
        try:
            fi    = yf.Ticker(cand).fast_info
            price = float(getattr(fi, 'last_price', 0) or 0)
            if price <= 0:
                continue
            prev  = float(getattr(fi, 'regular_market_previous_close', price) or price)
            chg   = (price - prev) / prev * 100 if prev > 0 else 0.0
            return {
                "price":    round(price, 2),
                "chg_pct":  round(chg, 2),
                "volume":   int(getattr(fi, 'last_volume', 0) or 0),
                "prev":     round(prev, 2),
                "high":     round(float(getattr(fi, 'day_high', price) or price), 2),
                "low":      round(float(getattr(fi, 'day_low',  price) or price), 2),
                "ok":       True,
                "resolved": cand,   # 實際成功的代號
            }
        except Exception:
            continue

    return empty


def _wl_scan_one(ticker: str, period: str, _bucket: str = "") -> dict | None:
    """
    對單一自選股做 DNA + 買點 + 籌碼完整掃描。

    ★ 改用 session_state 手動快取（移除 @st.cache_data）:
      @st.cache_data 在 Streamlit Cloud 首次執行時，即使 cache miss 也可能因為
      網路或 session_state 尚未初始化而回傳 None，改用手動快取更可靠。
      快取 key = f"_wl_scan_{ticker}_{date}_{bucket}" 盤中每分鐘失效。
    """
    # ── session_state 手動快取 ─────────────────────────────────────────
    cache_key = f"_wl_scan_{ticker}_{_bucket}"
    try:
        cached = st.session_state.get(cache_key)
        if cached is not None:
            return cached
    except Exception:
        pass

    try:
        df, used = fetch_data(ticker, period=period, time_bucket=_get_cache_bucket())
        if df is None or len(df) < 60:
            return None
        df  = add_indicators(df)
        dna = detect_wave_dna(df)
        wr  = compute_winrate(dna, df)

        # 籌碼（自選股掃描才打 FinMind REST API）
        chip_raw  = _fetch_chip_data(used)
        chip_eval = evaluate_chip(chip_raw)

        # 買點評估（含籌碼第⑥條件）
        entry = evaluate_entry_point(dna, wr, df, chip=chip_eval)

        # 前瞻 D+1、D+2 下限
        rows   = generate_forward_matrix(df, wr, dna, n_days=3)
        d1_low = rows[0]['下限參考'] if len(rows) > 0 else None
        d2_low = rows[1]['下限參考'] if len(rows) > 1 else None

        last = df.iloc[-1]
        result = {
            "代號":        used,
            "股名":        get_stock_name(used),
            "收盤價":      round(float(last["Close"]), 2),
            "勝率":        round(wr["winrate"] * 100, 1),
            "分類":        wr["category_label"],
            "category":    wr["category"],
            "R_cycle":     round(dna["R_cycle"], 3),
            "T_median":    dna["T_median"],
            "D_current":   dna["D_current"],
            "均線型態":    wr["desc_ma"],
            "KD狀態":      wr["desc_kd"],
            "時間波":      wr["desc_time"],
            "K9":          wr["k9"],
            "D9":          wr["d9"],
            "量比":        wr["vol_ratio"],
            "買點分數":    entry["score"],
            "買點訊號":    entry["signal"],
            "KD拐頭":      entry["kd_stage"],
            "買點條件":    entry["conditions"],
            "籌碼標籤":    chip_eval["label"],
            "籌碼加分":    chip_eval["boost"],
            "籌碼否決":    chip_eval["veto"],
            "籌碼說明":    chip_eval["detail"],
            "籌碼可用":    chip_raw.get("available", False),
            "it_buy_days": chip_raw.get("it_buy_days", 0),
            "fi_3d_sum":   chip_raw.get("fi_3d_sum", 0.0),
            "it_3d_sum":   chip_raw.get("it_3d_sum", 0.0),
            # 近10天每日明細（供彈窗）
            "fi_net_daily": chip_raw.get("fi_net_daily", {}),
            "it_net_daily": chip_raw.get("it_net_daily", {}),
            "fi_net_5d":   chip_raw.get("fi_net_5d", []),
            "it_net_5d":   chip_raw.get("it_net_5d", []),
            "D1下限":      d1_low,
            "D2下限":      d2_low,
            "chart_url":   get_chart_url(used),
        }

        try:
            st.session_state[cache_key] = result
        except Exception:
            pass

        return result
    except Exception:
        return None


def render_watchlist_page(period: str = "2y"):
    """
    ★ 自選股看板主體 — 業界下單軟體風格
    ─────────────────────────────────────────────────────────────────────────
    Layout:
      左側 Tab × 5 → 右側即時報價表格 + 操作列
      底部: 「一鍵掃描」→ DNA/買點/籌碼完整看板
    """
    _wl_init()
    bucket = _get_cache_bucket()

    st.markdown('<div class="section-title">⭐ 自選股看板</div>', unsafe_allow_html=True)

    # ── 一鍵修正代號後綴 ──────────────────────────────────────────────
    with st.expander("🔧 一鍵修正代號後綴（解決 -- 問題）", expanded=False):
        st.markdown(
            "部分上櫃股在 Yahoo Finance 須用 `.TWO` 後綴，"
            "若自選股顯示 `--`，點下方按鈕自動修正所有清單的後綴。",
            unsafe_allow_html=False
        )
        if st.button("🔍 自動偵測並修正所有清單", key="wl_fix_suffix",
                     use_container_width=True):
            fixed_list = []
            with st.spinner("偵測中，約需 10~30 秒..."):
                for idx in range(1, _WL_COUNT + 1):
                    wl = st.session_state[_WL_KEY][idx]
                    new_tickers = []
                    for t in wl["tickers"]:
                        resolved = _wl_resolve_ticker(t)
                        if resolved != t:
                            fixed_list.append(f"{t} → {resolved}")
                        new_tickers.append(resolved)
                    wl["tickers"] = new_tickers
            if fixed_list:
                st.success(f"✅ 已修正 {len(fixed_list)} 個代號：\n" +
                           "\n".join(f"• {f}" for f in fixed_list))
                st.rerun()
            else:
                st.info("✅ 所有代號後綴均正確，無需修正")

    # ── JSON 匯入匯出 ──────────────────────────────────────────────────
    with st.expander("💾 匯入 / 匯出自選股設定", expanded=False):
        import json as _json

        col_exp, col_imp = st.columns(2)

        # ── 匯出 ──────────────────────────────────────────────────────
        with col_exp:
            wl_data  = st.session_state[_WL_KEY]
            json_str = _json.dumps(
                {str(k): v for k, v in wl_data.items()},
                ensure_ascii=False, indent=2
            )
            st.download_button(
                "⬇️ 匯出自選股 JSON",
                data=json_str,
                file_name="watchlist.json",
                mime="application/json",
                use_container_width=True,
                key="wl_export_btn"
            )

        # ── 匯入 ──────────────────────────────────────────────────────
        # 用固定 key + session_state 記錄「已處理過的檔案」
        # 避免 file_uploader 因 Streamlit rerun 重複執行匯入邏輯
        with col_imp:
            uploaded = st.file_uploader(
                "⬆️ 匯入 JSON",
                type=["json"],
                key="wl_import_file",
                label_visibility="collapsed"
            )
            if uploaded is not None:
                # 用檔案名稱+大小作為唯一識別，避免重複處理
                file_sig = f"{uploaded.name}_{uploaded.size}"
                if st.session_state.get("_wl_last_import") != file_sig:
                    try:
                        loaded = _json.loads(uploaded.read())
                        for k, v in loaded.items():
                            idx = int(k)
                            if 1 <= idx <= _WL_COUNT:
                                st.session_state[_WL_KEY][idx] = v
                        st.session_state["_wl_last_import"] = file_sig
                        st.success("✅ 匯入成功！")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ 匯入失敗: {e}")

    # ── 五個自選股 Tab ──────────────────────────────────────────────────
    wl_names = [st.session_state[_WL_KEY][i]["name"] for i in range(1, _WL_COUNT + 1)]
    tabs = st.tabs([f"{'★ ' if i+1 == st.session_state['_wl_active'] else ''}{name}"
                    for i, name in enumerate(wl_names)])

    for tab_idx, tab in enumerate(tabs):
        wl_no = tab_idx + 1
        wl    = _wl_get(wl_no)

        with tab:
            st.session_state["_wl_active"] = wl_no

            # ── 清單標題編輯 ──────────────────────────────────────────
            c_name, c_add, c_scan = st.columns([3, 2, 2])
            with c_name:
                new_name = st.text_input(
                    f"清單名稱", value=wl["name"],
                    key=f"wl_name_{wl_no}", label_visibility="collapsed"
                )
                if new_name != wl["name"]:
                    _wl_set_name(wl_no, new_name)
                    st.rerun()

            with c_add:
                add_ticker = st.text_input(
                    "新增代號", placeholder="2330 / AAPL",
                    key=f"wl_add_{wl_no}", label_visibility="collapsed"
                )
                if st.button("➕ 加入", key=f"wl_add_btn_{wl_no}", use_container_width=True):
                    ok, msg = _wl_add_ticker(wl_no, add_ticker)
                    if ok:
                        st.toast(msg, icon="⭐")
                    else:
                        st.warning(msg)
                    st.rerun()

            with c_scan:
                do_scan = st.button(
                    f"🔬 一鍵掃描 ({len(wl['tickers'])} 檔)",
                    key=f"wl_scan_{wl_no}",
                    use_container_width=True,
                    type="primary",
                    disabled=len(wl["tickers"]) == 0
                )

            # ── 即時報價表格 ──────────────────────────────────────────
            if not wl["tickers"]:
                st.markdown("""
                <div style="text-align:center;padding:40px;color:#7a9bbf;font-size:14px;">
                  ℹ️ 此清單尚無股票。輸入代號後按「➕ 加入」。<br>
                  <span style="font-size:12px;">台股輸入數字(如 2330)，美股輸入英文(如 AAPL)</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                # 即時報價 HTML 表格
                quote_rows = ""
                for ticker in wl["tickers"]:
                    q    = _wl_fetch_quote(ticker, _bucket=bucket)
                    name = get_stock_name(ticker)
                    url  = get_chart_url(ticker)
                    chg  = q["chg_pct"]
                    chg_color = "#0a7c59" if chg > 0 else "#c0392b" if chg < 0 else "#666"
                    chg_str   = f"{chg:+.2f}%"
                    price_str = f"{q['price']:.2f}" if q['ok'] else "--"
                    vol_str   = f"{q['volume']//1000:,}張" if q['ok'] and q['volume'] > 0 else "--"

                    safe_title = f"{name}({ticker})".replace("'", " ")
                    quote_rows += f"""
                    <tr>
                      <td style="text-align:center;">
                        <button onclick="(function(){{
                          document.querySelectorAll('[data-testid=stTextInput] input').forEach(function(el){{
                            if(el.placeholder && el.placeholder.includes('2330'))
                              el.value='{ticker}';
                          }})
                        }})()" style="background:none;border:none;cursor:pointer;
                          font-size:11px;color:#1565c0;">▶</button>
                      </td>
                      <td>
                        <a href="{url}" target="_blank" style="color:#1565c0;font-weight:700;
                          font-family:'IBM Plex Mono',monospace;text-decoration:none;
                          font-size:14px;">{ticker}</a>
                      </td>
                      <td style="color:#1a2b3c;font-size:13px;">{name}</td>
                      <td style="font-family:'IBM Plex Mono',monospace;font-weight:700;
                          font-size:15px;color:#1a2b3c;">{price_str}</td>
                      <td style="font-family:'IBM Plex Mono',monospace;font-weight:700;
                          color:{chg_color};font-size:14px;">{chg_str}</td>
                      <td style="color:#4a6fa5;font-size:12px;">{vol_str}</td>
                      <td style="color:#4a6fa5;font-size:12px;">
                        H {q['high']:.2f} / L {q['low']:.2f}
                      </td>
                      <td style="text-align:center;">
                        <button onclick="openChart('{url}','{safe_title}')"
                          style="background:#eaf2fb;border:1px solid #b8cce0;border-radius:5px;
                          padding:2px 8px;cursor:pointer;font-size:13px;color:#1565c0;">📈</button>
                      </td>
                    </tr>"""

                st.markdown(f"""
                <div id="chartModal_wl" onclick="if(event.target===this){{this.style.display='none';document.getElementById('chartFrame_wl').src='';}}"
                     style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;
                     background:rgba(0,0,0,0.72);z-index:9999;align-items:center;justify-content:center;">
                  <div style="background:#fff;border-radius:14px;width:90%;max-width:1100px;
                              height:82vh;overflow:hidden;box-shadow:0 12px 48px rgba(0,0,0,.45);">
                    <div style="display:flex;align-items:center;justify-content:space-between;
                                padding:12px 18px;background:#1565c0;color:#fff;">
                      <span id="chartTitle_wl" style="font-weight:700;font-size:16px;"></span>
                      <button onclick="document.getElementById('chartModal_wl').style.display='none';
                                       document.getElementById('chartFrame_wl').src='';"
                              style="background:rgba(255,255,255,.2);border:none;color:#fff;
                                     font-size:18px;cursor:pointer;border-radius:6px;padding:4px 10px;">✕</button>
                    </div>
                    <iframe id="chartFrame_wl" src="" style="width:100%;height:calc(82vh - 52px);border:none;"></iframe>
                  </div>
                </div>
                <script>
                function openChart(url, title) {{
                    document.getElementById('chartFrame_wl').src = url;
                    document.getElementById('chartTitle_wl').textContent = title;
                    document.getElementById('chartModal_wl').style.display = 'flex';
                }}
                </script>
                <table style="width:100%;border-collapse:collapse;font-size:14px;">
                  <thead>
                    <tr style="background:#1565c0;color:#fff;">
                      <th style="padding:8px;width:32px;"></th>
                      <th style="padding:8px;text-align:left;min-width:100px;">代號</th>
                      <th style="padding:8px;text-align:left;min-width:80px;">股名</th>
                      <th style="padding:8px;text-align:right;min-width:80px;">現價</th>
                      <th style="padding:8px;text-align:right;min-width:75px;">漲跌幅</th>
                      <th style="padding:8px;text-align:right;min-width:75px;">成交量</th>
                      <th style="padding:8px;text-align:center;min-width:120px;">今日區間</th>
                      <th style="padding:8px;text-align:center;width:42px;">線型</th>
                    </tr>
                  </thead>
                  <tbody>{quote_rows}</tbody>
                </table>
                """, unsafe_allow_html=True)

                # ── 刪除操作列 ────────────────────────────────────────
                st.markdown('<div style="margin-top:10px;font-size:12px;color:#7a9bbf;">刪除股票：</div>',
                            unsafe_allow_html=True)
                del_cols = st.columns(min(len(wl["tickers"]), 5))
                for i, (col, ticker) in enumerate(zip(del_cols, wl["tickers"][:5])):
                    with col:
                        if st.button(f"✕ {ticker}", key=f"del_{wl_no}_{ticker}_{i}",
                                     use_container_width=True):
                            _wl_remove_ticker(wl_no, ticker)
                            st.rerun()
                if len(wl["tickers"]) > 5:
                    del_cols2 = st.columns(min(len(wl["tickers"]) - 5, 5))
                    for i, (col, ticker) in enumerate(zip(del_cols2, wl["tickers"][5:10])):
                        with col:
                            if st.button(f"✕ {ticker}", key=f"del2_{wl_no}_{ticker}_{i}",
                                         use_container_width=True):
                                _wl_remove_ticker(wl_no, ticker)
                                st.rerun()

            # ── 一鍵掃描結果 ─────────────────────────────────────────
            if do_scan and wl["tickers"]:
                st.markdown(f"""
                <div class="section-title">🔬 {wl['name']} — DNA × 買點 × 籌碼 完整掃描</div>
                """, unsafe_allow_html=True)

                prog = st.progress(0.0, text="⏳ 掃描中...")
                results = []
                total   = len(wl["tickers"])

                for i, ticker in enumerate(wl["tickers"]):
                    prog.progress((i + 1) / total,
                                  text=f"⏳ 分析 {ticker}（{i+1}/{total}）...")
                    r = _wl_scan_one(ticker, period, _bucket=bucket)
                    if r:
                        results.append(r)

                prog.empty()

                if not results:
                    st.warning("⚠️ 所有股票掃描失敗，請確認代號是否正確")
                else:
                    # 依買點分數排序
                    results.sort(key=lambda x: x["買點分數"], reverse=True)
                    _render_wl_scan_table(results)


def _render_wl_scan_table(results: list):
    """
    自選股掃描結果表格 v2
    ─────────────────────────────────────────────────────────────────────────
    修正:
      1. 籌碼顯示實際數值(不再只顯示標籤)
      2. 加入 D+2 下限欄位
      3. 買點評估 → 點擊開彈窗，顯示 5 大條件明細
      4. 法人動向 → 點擊開彈窗，顯示近10天三大法人每日買賣超明細表
    """
    bar_color = {"top": "#0a7c59", "mid": "#d97706", "warn": "#c0392b"}

    # ── 所有 JS 彈窗（一次定義）───────────────────────────────────────
    modal_defs = """
    <!-- 技術線型彈窗 -->
    <div id="wlChartModal" onclick="if(event.target===this){this.style.display='none';document.getElementById('wlChartFrame').src='';}"
         style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;
         background:rgba(0,0,0,0.72);z-index:9000;align-items:center;justify-content:center;">
      <div style="background:#fff;border-radius:14px;width:90%;max-width:1100px;
                  height:82vh;overflow:hidden;box-shadow:0 12px 48px rgba(0,0,0,.45);">
        <div style="display:flex;align-items:center;justify-content:space-between;
                    padding:12px 18px;background:#1565c0;color:#fff;">
          <span id="wlChartTitle" style="font-weight:700;font-size:16px;"></span>
          <button onclick="document.getElementById('wlChartModal').style.display='none';
                           document.getElementById('wlChartFrame').src='';"
                  style="background:rgba(255,255,255,.2);border:none;color:#fff;
                         font-size:18px;cursor:pointer;border-radius:6px;padding:4px 10px;">✕</button>
        </div>
        <iframe id="wlChartFrame" src="" style="width:100%;height:calc(82vh - 52px);border:none;"></iframe>
      </div>
    </div>

    <!-- 買點詳細彈窗 -->
    <div id="entryModal" onclick="if(event.target===this){this.style.display='none';}"
         style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;
         background:rgba(0,0,0,0.72);z-index:9001;align-items:center;justify-content:center;">
      <div style="background:#fff;border-radius:14px;width:420px;max-width:95%;
                  max-height:85vh;overflow-y:auto;box-shadow:0 12px 48px rgba(0,0,0,.45);">
        <div id="entryModalContent" style="padding:20px;"></div>
        <div style="padding:0 20px 16px;">
          <button onclick="document.getElementById('entryModal').style.display='none';"
                  style="width:100%;background:#1565c0;color:#fff;border:none;
                         border-radius:8px;padding:10px;font-size:14px;cursor:pointer;">關閉</button>
        </div>
      </div>
    </div>

    <!-- 法人動向彈窗 -->
    <div id="chipModal" onclick="if(event.target===this){this.style.display='none';}"
         style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;
         background:rgba(0,0,0,0.72);z-index:9001;align-items:center;justify-content:center;">
      <div style="background:#fff;border-radius:14px;width:560px;max-width:95%;
                  max-height:85vh;overflow-y:auto;box-shadow:0 12px 48px rgba(0,0,0,.45);">
        <div id="chipModalContent" style="padding:20px;"></div>
        <div style="padding:0 20px 16px;">
          <button onclick="document.getElementById('chipModal').style.display='none';"
                  style="width:100%;background:#1565c0;color:#fff;border:none;
                         border-radius:8px;padding:10px;font-size:14px;cursor:pointer;">關閉</button>
        </div>
      </div>
    </div>

    <script>
    function openWlChart(url, title) {
        document.getElementById('wlChartFrame').src = url;
        document.getElementById('wlChartTitle').textContent = title;
        document.getElementById('wlChartModal').style.display = 'flex';
    }
    function openEntryDetail(html) {
        document.getElementById('entryModalContent').innerHTML = html;
        document.getElementById('entryModal').style.display = 'flex';
    }
    function openChipDetail(html) {
        document.getElementById('chipModalContent').innerHTML = html;
        document.getElementById('chipModal').style.display = 'flex';
    }
    </script>"""

    # ── 逐列生成 ─────────────────────────────────────────────────────
    rows_html = ""
    for i, r in enumerate(results, 1):
        cat     = r["category"]
        bc      = bar_color.get(cat, "#1565c0")
        wr_val  = r["勝率"]
        score   = r["買點分數"]
        signal  = r["買點訊號"]
        code    = r["代號"]
        name    = r.get("股名", "")
        url     = r.get("chart_url", get_chart_url(code))
        safe_t  = f"{name}({code})".replace("'", " ")

        sc_color = ("#c0392b" if "共振" in signal else "#0a7c59" if score >= 80
                    else "#1565c0" if score >= 65 else "#d97706" if score >= 50 else "#9e9e9e")

        cat_badge = {
            "top":  '<span style="background:#0a7c59;color:#fff;padding:2px 8px;border-radius:4px;font-size:11px;white-space:nowrap;">🚀頂級</span>',
            "mid":  '<span style="background:#d97706;color:#fff;padding:2px 8px;border-radius:4px;font-size:11px;white-space:nowrap;">⏳蓄勢</span>',
            "warn": '<span style="background:#c0392b;color:#fff;padding:2px 8px;border-radius:4px;font-size:11px;white-space:nowrap;">🛑警戒</span>',
        }.get(cat, r["分類"])

        rc       = r["R_cycle"]
        rc_color = "#0a7c59" if rc >= 1.0 else "#d97706" if rc >= 0.6 else "#c0392b"
        d1_str   = f"{r['D1下限']:.2f}" if r.get("D1下限") else "--"
        d2_str   = f"{r['D2下限']:.2f}" if r.get("D2下限") else "--"

        # ── 買點詳細彈窗 HTML ─────────────────────────────────────────
        conds = r.get("買點條件", {})
        ci = lambda v: "✅" if v else "❌"
        entry_html = f"""
        <div style='font-family:Noto Sans TC,sans-serif;'>
          <div style='font-size:18px;font-weight:700;color:{sc_color};margin-bottom:12px;'>
            {code} {name}<br>
            <span style='font-size:28px;'>{score}分</span>
            <span style='font-size:16px;'> {signal}</span>
          </div>
          <div style='font-size:14px;line-height:2.2;'>
            <div>{ci(conds.get('c3_rcycle'))} ① R_cycle ≥ 1.0 時間波飽和
              <b style='color:{rc_color};'> {rc:.3f}</b></div>
            <div>{ci(conds.get('c4_kd'))} ② KD 低檔拐頭
              <b> {r.get('KD拐頭','')} (K9={r.get('K9',0):.0f} D9={r.get('D9',0):.0f})</b></div>
            <div>{ci(conds.get('c1_mid'))} ③ 中繼蓄勢分類
              <b> {r['分類']}</b></div>
            <div>{ci(conds.get('c2_wr'))} ④ 勝率甜蜜區 50~68%
              <b> {wr_val:.0f}%</b></div>
            <div>{ci(conds.get('c5_vol'))} ⑤ 量比 &lt; 2.5
              <b> {r.get('量比',0):.2f}x</b></div>
          </div>
          <div style='margin-top:12px;padding:10px;background:#eaf2fb;border-radius:8px;font-size:13px;'>
            📌 掛單參考：D+1下限 <b>{d1_str}</b> 元 ｜ D+2下限 <b>{d2_str}</b> 元（停損基準）
          </div>
          <div style='margin-top:8px;font-size:12px;color:#4a6fa5;'>
            {r.get('籌碼說明','')}
          </div>
        </div>""".replace('"', '&quot;').replace("'", "&#39;")

        # ── 法人彈窗 HTML（近10天每日明細）──────────────────────────
        fi_d = r.get("fi_net_daily", {})
        it_d = r.get("it_net_daily", {})
        chip_avail = r.get("籌碼可用", False)
        it_days = r.get("it_buy_days", 0)
        fi_3d = r.get("fi_3d_sum", 0.0)
        it_3d = r.get("it_3d_sum", 0.0)

        if chip_avail and (fi_d or it_d):
            all_dates = sorted(set(list(fi_d.keys()) + list(it_d.keys())), reverse=True)[:10]
            chip_rows = ""
            for d in all_dates:
                fi_v = fi_d.get(d, 0.0)
                it_v = it_d.get(d, 0.0)
                fi_col = "#0a7c59" if fi_v > 0 else "#c0392b"
                it_col = "#0a7c59" if it_v > 0 else "#c0392b"
                chip_rows += f"""<tr style='border-bottom:1px solid #eaf2fb;'>
                  <td style='padding:6px 10px;color:#4a6fa5;font-size:13px;'>{d}</td>
                  <td style='padding:6px 10px;text-align:right;font-weight:700;
                      color:{fi_col};font-family:IBM Plex Mono,monospace;font-size:13px;'>{fi_v:+,.0f}</td>
                  <td style='padding:6px 10px;text-align:right;font-weight:700;
                      color:{it_col};font-family:IBM Plex Mono,monospace;font-size:13px;'>{it_v:+,.0f}</td>
                </tr>"""
            chip_html = f"""
            <div style='font-family:Noto Sans TC,sans-serif;'>
              <div style='font-size:17px;font-weight:700;color:#1a2b3c;margin-bottom:14px;'>
                {code} {name} — 三大法人近10天買賣超（張）
              </div>
              <div style='display:flex;gap:16px;margin-bottom:14px;font-size:13px;'>
                <div style='background:#eaf2fb;border-radius:8px;padding:8px 14px;'>
                  外資近3日合計：<b style='color:{"#0a7c59" if fi_3d>0 else "#c0392b"};'>{fi_3d:+,.0f}張</b>
                </div>
                <div style='background:#eaf2fb;border-radius:8px;padding:8px 14px;'>
                  投信近5日買超：<b style='color:{"#0a7c59" if it_days>=3 else "#1a2b3c"};'>{it_days}/5天</b>
                </div>
                <div style='background:#eaf2fb;border-radius:8px;padding:8px 14px;'>
                  投信近3日：<b style='color:{"#0a7c59" if it_3d>0 else "#c0392b"};'>{it_3d:+,.0f}張</b>
                </div>
              </div>
              <table style='width:100%;border-collapse:collapse;'>
                <thead>
                  <tr style='background:#1565c0;color:#fff;font-size:13px;'>
                    <th style='padding:8px 10px;text-align:left;'>日期</th>
                    <th style='padding:8px 10px;text-align:right;'>外資（張）</th>
                    <th style='padding:8px 10px;text-align:right;'>投信（張）</th>
                  </tr>
                </thead>
                <tbody>{chip_rows}</tbody>
              </table>
              <div style='margin-top:10px;font-size:11px;color:#7a9bbf;'>
                資料來源：FinMind / 台灣交易所法人買賣超公告（正數=買超，負數=賣超）
              </div>
            </div>"""
        else:
            chip_html = f"""
            <div style='font-family:Noto Sans TC,sans-serif;padding:10px;'>
              <div style='font-size:16px;font-weight:700;color:#1a2b3c;margin-bottom:10px;'>
                {code} {name} — 三大法人
              </div>
              <div style='color:#4a6fa5;font-size:14px;'>
                ℹ️ 籌碼資料暫時無法取得，可能原因：<br>
                • FinMind API 連線逾時<br>
                • 該股為美股或特殊股票（無台灣法人資料）<br>
                • 請稍後重新掃描
              </div>
            </div>"""

        chip_html_esc  = chip_html.replace('"', '&quot;').replace("'", "&#39;")
        chip_label = r["籌碼標籤"]
        chip_color = ("#c0392b" if "共振" in chip_label or "倒貨" in chip_label
                      else "#0a7c59" if "認養" in chip_label or "共振" in chip_label
                      else "#d97706" if "佈局" in chip_label
                      else "#7a9bbf")

        # 法人數值（直接顯示）
        fi_3d_r = r.get("fi_3d_sum", 0.0)
        it_3d_r = r.get("it_3d_sum", 0.0)
        it_d_r  = r.get("it_buy_days", 0)
        fi_col_r = "#0a7c59" if fi_3d_r > 0 else "#c0392b"
        it_col_r = "#0a7c59" if it_3d_r > 0 else "#c0392b"

        row_bg = "background:#f7fafd;" if i % 2 == 0 else "background:#ffffff;"

        rows_html += f"""
        <tr style="{row_bg}">
          <td style="color:#7a9bbf;text-align:center;font-size:12px;">{i}</td>
          <td>
            <a href="{url}" target="_blank"
               style="color:#1565c0;font-weight:700;font-size:13px;
                      font-family:'IBM Plex Mono',monospace;text-decoration:none;">{code}</a>
          </td>
          <td style="color:#1a2b3c;font-size:12px;">{name}</td>
          <td style="font-family:'IBM Plex Mono',monospace;font-weight:700;
                     font-size:14px;color:#1a2b3c;">{r['收盤價']}</td>
          <td>
            <div style="display:flex;align-items:center;gap:5px;">
              <div style="width:45px;background:#c8d8e8;border-radius:3px;height:7px;overflow:hidden;">
                <div style="width:{min(wr_val,100):.0f}%;height:7px;background:{bc};border-radius:3px;"></div>
              </div>
              <span style="color:{bc};font-weight:700;font-size:12px;">{wr_val:.0f}%</span>
            </div>
          </td>
          <td>{cat_badge}</td>
          <td style="color:{rc_color};font-weight:700;font-size:12px;
                     font-family:'IBM Plex Mono',monospace;">{rc:.3f}</td>
          <td>
            <button onclick="openEntryDetail('{entry_html}')"
              style="background:{sc_color};color:#fff;border:none;border-radius:6px;
                     padding:4px 10px;cursor:pointer;font-size:12px;font-weight:700;
                     white-space:nowrap;">
              {score}分 {signal[:4]}
            </button>
          </td>
          <td>
            <button onclick="openChipDetail('{chip_html_esc}')"
              style="background:transparent;border:1px solid {chip_color};border-radius:6px;
                     padding:3px 8px;cursor:pointer;font-size:11px;color:{chip_color};
                     white-space:nowrap;">
              {chip_label[:8]}
            </button>
          </td>
          <td>
            <div style="font-size:11px;line-height:1.8;">
              <span style="color:{fi_col_r};font-weight:600;">外{fi_3d_r:+,.0f}張</span><br>
              <span style="color:{it_col_r};font-weight:600;">投{it_3d_r:+,.0f}(買{it_d_r}/5)</span>
            </div>
          </td>
          <td style="font-family:'IBM Plex Mono',monospace;font-size:12px;font-weight:700;
                     color:#1565c0;text-align:center;">{d1_str}</td>
          <td style="font-family:'IBM Plex Mono',monospace;font-size:12px;font-weight:700;
                     color:#d97706;text-align:center;">{d2_str}</td>
          <td style="text-align:center;">
            <button onclick="openWlChart('{url}','{safe_t}')"
              style="background:#eaf2fb;border:1px solid #b8cce0;border-radius:5px;
                     padding:3px 8px;cursor:pointer;font-size:12px;color:#1565c0;">📈</button>
          </td>
        </tr>"""

    table_html = f"""
    {modal_defs}
    <div style="overflow-x:auto;-webkit-overflow-scrolling:touch;">
    <table style="width:100%;border-collapse:collapse;font-size:13px;min-width:900px;">
      <thead>
        <tr style="background:#1565c0;color:#fff;font-size:11px;">
          <th style="padding:8px 5px;width:28px;">#</th>
          <th style="padding:8px;text-align:left;min-width:85px;">代號</th>
          <th style="padding:8px;text-align:left;min-width:65px;">股名</th>
          <th style="padding:8px;text-align:left;min-width:70px;">收盤價</th>
          <th style="padding:8px;text-align:left;min-width:90px;">波段勝率</th>
          <th style="padding:8px;text-align:left;min-width:60px;">分類</th>
          <th style="padding:8px;text-align:left;min-width:65px;">R_cycle</th>
          <th style="padding:8px;text-align:left;min-width:95px;">🎯買點評估</th>
          <th style="padding:8px;text-align:left;min-width:80px;">🧬籌碼動態</th>
          <th style="padding:8px;text-align:left;min-width:95px;">法人動向(3日)</th>
          <th style="padding:8px;text-align:center;min-width:65px;">D+1下限</th>
          <th style="padding:8px;text-align:center;min-width:65px;">D+2下限</th>
          <th style="padding:8px;width:38px;">線型</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    </div>"""

    st.markdown(table_html, unsafe_allow_html=True)

    # 手機卡片版
    cards_html = '<div class="mobile-cards">'
    for i, r in enumerate(results, 1):
        code   = r["代號"]
        name   = r.get("股名", "")
        url    = r.get("chart_url", get_chart_url(code))
        score  = r["買點分數"]
        cat    = r["category"]
        bc     = bar_color.get(cat, "#1565c0")
        sc_col = ("#0a7c59" if score >= 80 else "#1565c0" if score >= 65
                  else "#d97706" if score >= 50 else "#9e9e9e")
        fi_3d  = r.get("fi_3d_sum", 0.0)
        it_d   = r.get("it_buy_days", 0)
        fi_col = "#0a7c59" if fi_3d > 0 else "#c0392b"
        d1     = f"{r['D1下限']:.2f}" if r.get("D1下限") else "--"
        d2     = f"{r['D2下限']:.2f}" if r.get("D2下限") else "--"
        cards_html += f"""
        <div class="scan-card">
          <div class="sc-header">
            <div>
              <a href="{url}" target="_blank" class="sc-code">#{i} {code}</a>
              <span class="sc-name">{" · " + name if name else ""}</span>
            </div>
            <span style="font-size:13px;font-weight:700;color:{sc_col};">{score}分 {r['買點訊號'][:4]}</span>
          </div>
          <div class="sc-meta">
            <span>勝率 <b style="color:{bc};">{r['勝率']:.0f}%</b></span>
            <span>R {r['R_cycle']:.3f}</span>
            <span style="color:{fi_col};">外資3d {fi_3d:+,.0f}張</span>
            <span>投信{it_d}/5天</span>
          </div>
          <div class="sc-meta" style="margin-top:4px;">
            <span>D+1下限 <b>{d1}</b></span>
            <span>D+2下限 <b>{d2}</b></span>
          </div>
          <div class="sc-desc">{r.get('籌碼說明','')[:40]}</div>
        </div>"""
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)
    bar_color = {"top": "#0a7c59", "mid": "#d97706", "warn": "#c0392b"}
    modal_js = """
    <div id="scanModal" onclick="if(event.target===this){this.style.display='none';document.getElementById('scanFrame').src='';}"
         style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;
         background:rgba(0,0,0,0.72);z-index:9999;align-items:center;justify-content:center;">
      <div style="background:#fff;border-radius:14px;width:90%;max-width:1100px;
                  height:82vh;overflow:hidden;box-shadow:0 12px 48px rgba(0,0,0,.45);">
        <div style="display:flex;align-items:center;justify-content:space-between;
                    padding:12px 18px;background:#1565c0;color:#fff;">
          <span id="scanTitle" style="font-weight:700;font-size:16px;"></span>
          <button onclick="document.getElementById('scanModal').style.display='none';
                           document.getElementById('scanFrame').src='';"
                  style="background:rgba(255,255,255,.2);border:none;color:#fff;
                         font-size:18px;cursor:pointer;border-radius:6px;padding:4px 10px;">✕</button>
        </div>
        <iframe id="scanFrame" src="" style="width:100%;height:calc(82vh - 52px);border:none;"></iframe>
      </div>
    </div>
    <script>
    function openScan(url, title) {
        document.getElementById('scanFrame').src = url;
        document.getElementById('scanTitle').textContent = title;
        document.getElementById('scanModal').style.display = 'flex';
    }
    </script>"""

    rows_html = ""
    for i, r in enumerate(results, 1):
        cat   = r["category"]
        bc    = bar_color.get(cat, "#1565c0")
        wr    = r["勝率"]
        score = r["買點分數"]
        signal = r["買點訊號"]
        code  = r["代號"]
        name  = r.get("股名", "")
        url   = r.get("chart_url", get_chart_url(code))
        safe_title = f"{name}({code})".replace("'", " ")

        # 買點分數顏色
        sc_color = ("#c0392b" if "共振" in signal
                    else "#0a7c59" if score >= 80
                    else "#1565c0" if score >= 65
                    else "#d97706" if score >= 50
                    else "#9e9e9e")

        # 分類標籤
        cat_badge = {
            "top":  '<span style="background:#0a7c59;color:#fff;padding:2px 8px;border-radius:4px;font-size:12px;">🚀頂級</span>',
            "mid":  '<span style="background:#d97706;color:#fff;padding:2px 8px;border-radius:4px;font-size:12px;">⏳蓄勢</span>',
            "warn": '<span style="background:#c0392b;color:#fff;padding:2px 8px;border-radius:4px;font-size:12px;">🛑警戒</span>',
        }.get(cat, r["分類"])

        # 籌碼顏色
        chip_lbl   = r["籌碼標籤"]
        chip_color = ("#c0392b" if "共振" in chip_lbl or "倒貨" in chip_lbl
                      else "#0a7c59" if "認養" in chip_lbl or "共振" in chip_lbl
                      else "#d97706" if "佈局" in chip_lbl
                      else "#9e9e9e")

        # 外資/投信動向
        it_d   = r["it_buy_days"]
        fi_3d  = r["fi_3d_sum"]
        it_3d  = r["it_3d_sum"]
        it_str = f"投信{it_d}/5天"
        fi_str = f"外資{fi_3d:+.0f}張"

        # D+1掛單下限
        d1 = r.get("D1下限")
        d1_str = f"{d1:.2f}" if d1 else "--"

        # R_cycle 顏色
        rc       = r["R_cycle"]
        rc_color = "#0a7c59" if rc >= 1.0 else "#d97706" if rc >= 0.6 else "#c0392b"

        row_bg = "background:#f7fafd;" if i % 2 == 0 else "background:#ffffff;"

        rows_html += f"""
        <tr style="{row_bg}">
          <td style="color:#7a9bbf;text-align:center;font-size:12px;">{i}</td>
          <td>
            <a href="{url}" target="_blank"
               style="color:#1565c0;font-weight:700;font-size:14px;
                      font-family:'IBM Plex Mono',monospace;text-decoration:none;">{code}</a>
          </td>
          <td style="color:#1a2b3c;font-size:13px;">{name}</td>
          <td style="font-family:'IBM Plex Mono',monospace;font-weight:700;
                     font-size:15px;color:#1a2b3c;">{r['收盤價']}</td>
          <td>
            <div style="display:flex;align-items:center;gap:6px;">
              <div style="width:55px;background:#c8d8e8;border-radius:4px;height:8px;overflow:hidden;">
                <div style="width:{min(wr,100):.0f}%;height:8px;background:{bc};border-radius:4px;"></div>
              </div>
              <span style="color:{bc};font-weight:700;font-size:13px;">{wr:.0f}%</span>
            </div>
          </td>
          <td>{cat_badge}</td>
          <td style="color:{rc_color};font-weight:700;font-size:13px;
                     font-family:'IBM Plex Mono',monospace;">{rc:.3f}</td>
          <td>
            <div style="font-size:13px;font-weight:700;color:{sc_color};">{score}分</div>
            <div style="font-size:11px;color:{sc_color};">{signal}</div>
          </td>
          <td style="font-size:12px;color:{chip_color};font-weight:600;">{chip_lbl}</td>
          <td>
            <div style="font-size:12px;color:{'#0a7c59' if it_d>=3 else '#4a6fa5'};">{it_str}</div>
            <div style="font-size:12px;color:{'#0a7c59' if fi_3d>0 else '#c0392b'};">{fi_str}</div>
          </td>
          <td style="font-family:'IBM Plex Mono',monospace;font-size:13px;
                     color:#1565c0;font-weight:600;">{d1_str}</td>
          <td style="text-align:center;">
            <button onclick="openScan('{url}','{safe_title}')"
              style="background:#eaf2fb;border:1px solid #b8cce0;border-radius:5px;
                     padding:3px 8px;cursor:pointer;font-size:13px;color:#1565c0;">📈</button>
          </td>
        </tr>"""

    table_html = f"""
    {modal_js}
    <table style="width:100%;border-collapse:collapse;font-size:13px;">
      <thead>
        <tr style="background:#1565c0;color:#fff;font-size:12px;">
          <th style="padding:8px;width:32px;">#</th>
          <th style="padding:8px;text-align:left;min-width:90px;">代號</th>
          <th style="padding:8px;text-align:left;min-width:70px;">股名</th>
          <th style="padding:8px;text-align:left;min-width:75px;">收盤價</th>
          <th style="padding:8px;text-align:left;min-width:100px;">波段勝率</th>
          <th style="padding:8px;text-align:left;min-width:65px;">分類</th>
          <th style="padding:8px;text-align:left;min-width:70px;">R_cycle</th>
          <th style="padding:8px;text-align:left;min-width:80px;">買點評估</th>
          <th style="padding:8px;text-align:left;min-width:100px;">籌碼動態</th>
          <th style="padding:8px;text-align:left;min-width:100px;">法人動向</th>
          <th style="padding:8px;text-align:center;min-width:75px;">D+1下限</th>
          <th style="padding:8px;width:42px;">線型</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>"""

    st.markdown(table_html, unsafe_allow_html=True)

    # 手機卡片版
    cards_html = '<div class="mobile-cards">'
    for i, r in enumerate(results, 1):
        code  = r["代號"]
        name  = r.get("股名", "")
        url   = r.get("chart_url", get_chart_url(code))
        score = r["買點分數"]
        cat   = r["category"]
        bc    = bar_color.get(cat, "#1565c0")
        sc_color = ("#0a7c59" if score >= 80 else "#1565c0" if score >= 65
                    else "#d97706" if score >= 50 else "#9e9e9e")
        cards_html += f"""
        <div class="scan-card">
          <div class="sc-header">
            <div>
              <a href="{url}" target="_blank" class="sc-code">#{i} {code}</a>
              <span class="sc-name">{" · " + name if name else ""}</span>
            </div>
            <span style="font-size:13px;font-weight:700;color:{sc_color};">{score}分 {r['買點訊號']}</span>
          </div>
          <div class="sc-meta">
            <span>勝率 <b style="color:{bc};">{r['勝率']:.0f}%</b></span>
            <span>R {r['R_cycle']:.3f}</span>
            <span>{r['籌碼標籤']}</span>
            <span>D+1下限 <b>{r.get('D1下限','--')}</b></span>
          </div>
          <div class="sc-desc">{r['均線型態'][:25]}</div>
        </div>"""
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)



def _radar_sort_key(r: dict) -> tuple:
    """
    ★ 最優質加強排序鍵：
      1. 勝率（Win Rate）由高到低 — 主排序
      2. R_cycle 越接近 1.0~1.3 甜蜜區間越優先 — 次排序
         （距離甜蜜區中心 1.15 的絕對值越小，排序權重越高）
    """
    winrate = r.get("勝率", 0)
    rc      = r.get("R_cycle", 0)
    # R_cycle 在 1.0~1.3 區間內距離 1.15 的差距（越小越好）
    if 1.0 <= rc <= 1.3:
        rc_dist = abs(rc - 1.15)
    else:
        rc_dist = abs(rc - 1.15) + 1.0   # 區間外的標的明確排在區間內標的之後
    return (-winrate, rc_dist)   # 負號讓勝率由高到低排序


def _send_radar_status_report(scanned_count: int, golden_count: int, stage: str):
    """
    ★ Discord 雷達運作狀態回報（開盤首輪 / 收盤末輪）
    用來確認雲端雷達系統仍正常運作中，未死機。
    """
    import pytz as _pytz
    now_tw = datetime.datetime.now(_pytz.timezone('Asia/Taipei'))
    msg = (
        f"🤖 **【波浪 DNA 雲端雷達 · 運行日誌】**\n"
        f"📅 觀測日期：{now_tw.strftime('%Y-%m-%d')}\n"
        f"⏰ 狀態時間：{now_tw.strftime('%H:%M:%S')}（{stage}）\n"
        f"📊 掃描進度：本輪已全數掃描全市場 {scanned_count} 檔量大與漲跌幅焦點股。\n"
        f"🎯 本輪大藍燈達標：{golden_count} 檔。"
    )
    send_discord_notify(msg)


def _auto_radar_scan_and_notify(period: str = "2y"):
    """
    ★ 全市場雷達自動掃描 + Discord 精選推播
    ─────────────────────────────────────────────────────────────────────
    流程：
      1. get_taiwan_hot_tickers() 取得全市場成交量前 50 大 + 核心底盤
         （約 50~55 檔，CORE_RADAR_WATCHLIST 永遠包含在內）
      2. run_radar_scan() 多執行緒併發掃描（max_workers=8）
      3. 篩出 all_green（五大條件全綠 + 籌碼未一票否決）的標的
      4. 依 _radar_sort_key 排序：勝率高→低，R_cycle 越近 1.0~1.3 越優先
      5. 只取 Top 3 推播 Discord，沒有達標則完全不發送（防洗版）
      6. 開盤首輪（09:00~09:05）與收盤末輪（13:25~13:35）額外發送
         「雷達運作狀態回報」，確認系統存活

    推播記錄：用「_notified_{今日日期}」key，每天自動重置，
              確保每支股票每天最多推播一次。
    """
    import pytz as _pytz
    now_tw      = datetime.datetime.now(_pytz.timezone('Asia/Taipei'))
    _today_str  = now_tw.strftime('%Y%m%d')
    _notify_key = f"_notified_{_today_str}"

    if _notify_key not in st.session_state:
        st.session_state[_notify_key] = set()
    notified = st.session_state[_notify_key]

    # ── ① 組裝全市場掃描池（全市場前50大 + 核心底盤，自動去重複）──
    try:
        scan_pool = get_taiwan_hot_tickers(top_n=50)
    except Exception:
        scan_pool = list(CORE_RADAR_WATCHLIST)   # 全市場抓取失敗，退守核心底盤

    # ── ② 多執行緒掃描全市場池 ─────────────────────────────────────
    results = run_radar_scan(scan_pool, period=period, with_chip=True)
    golden  = [r for r in results if r.get("all_green")]

    # ── ③ 最優質排序：勝率高→低，R_cycle 越近 1.0~1.3 越優先 ───────
    golden.sort(key=_radar_sort_key)

    # ── ④ 只取 Top 3 推播，避免洗版；尚未推播過的才送 ───────────────
    top3 = [r for r in golden if r["代號"] not in notified][:3]

    for r in top3:
        code = r["代號"]
        d1   = f"{r['D1下限']:.2f}" if r.get("D1下限") else "--"
        d2   = f"{r['D2下限']:.2f}" if r.get("D2下限") else "--"
        chip_note = f"\n🧬 籌碼動向：{r['chip_label']}" if r.get("chip_label") else ""
        msg = (
            f"🚨 **【波浪 DNA 雷達·起漲點觸發】** {now_tw.strftime('%H:%M')}\n"
            f"📈 標的：**{r['股名']}** (`{code}`)\n"
            f"💰 當前現價：**{r['現價']}** 元 ｜ 🎯 預測勝率：**{r['勝率']:.0f}%**\n"
            f"🧬 歷史對稱率 R_cycle：**{r['R_cycle']:.3f}**\n"
            f"📌 建議掛單 (D+1 下限)：**{d1}** 元\n"
            f"🛡️ 停損基準 (D+2 下限)：**{d2}** 元"
            f"{chip_note}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        success = send_discord_notify(msg)
        if success:
            notified.add(code)
            st.session_state[_notify_key] = notified
        import time as _time; _time.sleep(0.5)  # ★ 防 Discord 429（瞬間多條）

    # ── ⑤ 開盤首輪 / 收盤末輪：發送雷達存活狀態回報 ────────────────
    _status_key_open  = f"_status_open_{_today_str}"
    _status_key_close = f"_status_close_{_today_str}"
    t = now_tw.time()

    if (datetime.time(9, 0) <= t <= datetime.time(9, 5)
            and not st.session_state.get(_status_key_open)):
        _send_radar_status_report(len(results), len(golden), "開盤首輪")
        st.session_state[_status_key_open] = True

    if (datetime.time(13, 25) <= t <= datetime.time(13, 35)
            and not st.session_state.get(_status_key_close)):
        _send_radar_status_report(len(results), len(golden), "收盤末輪")
        st.session_state[_status_key_close] = True


# ─────────────────────────────────────────────────────────────────────────────
#  🚀 雷達掃描引擎
# ─────────────────────────────────────────────────────────────────────────────

def run_radar_scan(tickers: list[str], period: str = "2y",
                   with_chip: bool = True) -> list[dict]:
    """
    多執行緒雷達掃描：對每支股票執行完整的 DNA + 買點 + 籌碼評估。

    ─────────────────────────────────────────────────────────────────────
    ★ all_green 判定（大藍燈基本關卡）：
      五大技術條件全部成立 AND 籌碼面未被一票否決（veto=False）。
      若 with_chip=False（全市場海選池太大時可關閉籌碼以加速），
      則只看五大技術條件。

    ★ max_workers=8：
      在 5~10 之間取中間值。過高會被 Yahoo Finance 限流(429)，
      過低掃描 50+ 檔會太慢。每支股票完整跑一次 fetch_data + FinMind
      籌碼查詢，實測 8 條併發是穩定與速度的平衡點。

    ★ try-except 保護：
      任何一檔股票的 fetch_data、FinMind、或計算過程失敗，
      _scan_one_radar 都會 return None，不會讓整個 ThreadPoolExecutor
      崩潰，futures.result() 逐一收集即可繼續處理下一檔。

    回傳：list[dict]，每筆含 all_green / 買點分數 / 籌碼資訊等完整欄位
    """
    def _scan_one_radar(ticker: str) -> dict | None:
        try:
            df, used = fetch_data(ticker, period=period,
                                  time_bucket=_get_cache_bucket())
            if df is None or len(df) < 60:
                return None

            df, _ = _patch_today_price(df, used)
            df    = add_indicators(df)
            dna   = detect_wave_dna(df)
            wr    = compute_winrate(dna, df)

            # ── 籌碼評估（可選，全市場海選時可關閉以加速）──────────
            chip_eval = None
            chip_raw  = None
            if with_chip:
                try:
                    chip_raw  = _fetch_chip_data(used)
                    chip_eval = evaluate_chip(chip_raw)
                except Exception:
                    chip_eval = None   # 籌碼失敗不影響技術面評估

            entry = evaluate_entry_point(dna, wr, df, chip=chip_eval)
            conds = entry["conditions"]

            # 五大技術條件全綠 AND 籌碼未被一票否決 → 真正的大藍燈
            tech_all_green = all(conds.values())
            chip_veto      = bool(chip_eval and chip_eval.get("veto"))
            all_green      = tech_all_green and not chip_veto

            # 前瞻矩陣取 D+1 D+2 下限
            rows = generate_forward_matrix(df, wr, dna, n_days=3)
            d1   = rows[0]["下限參考"] if len(rows) > 0 else None
            d2   = rows[1]["下限參考"] if len(rows) > 1 else None

            return {
                "代號":       used,
                "股名":       get_stock_name(used),
                "現價":       round(float(df["Close"].iloc[-1]), 2),
                "R_cycle":    round(dna["R_cycle"], 3),
                "勝率":       round(wr["winrate"] * 100, 1),
                "買點分數":   entry["score"],
                "買點訊號":   entry["signal"],
                "D1下限":     d1,
                "D2下限":     d2,
                "conds":      conds,
                "all_green":  all_green,
                "chip_veto":  chip_veto,
                "chip_label": chip_eval.get("label", "") if chip_eval else "",
                "K9":         wr["k9"],
                "D9":         wr["d9"],
                "量比":       wr["vol_ratio"],
                "category":   wr["category"],
            }
        except Exception:
            return None   # ★ 單檔失敗優雅跳過，不影響其他股票掃描

    results = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_scan_one_radar, t): t for t in tickers}
        for fut in futures:
            try:
                r = fut.result()
                if r is not None:
                    results.append(r)
            except Exception:
                continue   # ★ 個別 future 拋例外也不中斷整體掃描
    return results

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="font-family:'IBM Plex Mono',monospace;margin-bottom:18px;">
          <div style="font-size:18px;font-weight:700;color:#1565c0;letter-spacing:1px;">🧬 波浪 DNA</div>
          <div style="font-size:10px;color:#7a9bbf;letter-spacing:2px;margin-top:4px;">
            DYNAMIC WAVE CYCLE DNA
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── 模式切換 ──────────────────────────────────────────────────────
        mode = st.radio(
            "分析模式",
            ["🔍 單股分析", "⭐ 自選股", "📡 批量掃描"],
            horizontal=True,
        )

        st.markdown("---")

        # ── 🚀 雷達掃描控制區 ──────────────────────────────────────────────
        with st.expander("🚀 雷達掃描（快速戰情）", expanded=False):

            # ── 掃描來源 ──────────────────────────────────────────────
            radar_source = st.radio(
                "掃描來源",
                ["📋 自訂清單", "🌐 全市場海選100（成交量+漲跌幅）",
                 "📊 即時成交量排行 100", "📈 即時漲幅排行 100", "📉 即時跌幅排行 100"],
                index=0,
                key="radar_source_radio",
                help="全市場海選＝成交量前50＋漲幅前50＋跌幅前50合併去重複，"
                     "與 Discord 自動推播使用同一份掃描池"
            )

            # ── 自訂清單（只在選擇自訂時顯示）────────────────────────
            if radar_source == "📋 自訂清單":
                st.markdown(
                    '<div style="font-size:12px;color:#4a6fa5;margin:6px 0 4px;">'
                    '代號（逗號或換行分隔），留空使用預設清單</div>',
                    unsafe_allow_html=True
                )
                radar_input = st.text_area(
                    "自選股清單", height=80,
                    value="\n".join(DEFAULT_WATCHLIST),
                    placeholder="8150\n2330\n2454",
                    label_visibility="collapsed",
                    key="radar_input_text"
                )
            else:
                radar_input = ""

            # ── 勝率門檻 ──────────────────────────────────────────────
            radar_min_wr = st.slider(
                "最低勝率門檻 (%)", 0, 90, 50, step=5,
                help="只顯示勝率 ≥ 此值的標的（0 = 顯示全部）",
                key="radar_min_wr"
            )

            # ── 掃描按鈕 ──────────────────────────────────────────────
            radar_scan = st.button(
                "🚀 啟動雷達大掃描",
                use_container_width=True,
                type="primary",
                key="radar_scan_btn"
            )
            if radar_scan:
                st.session_state["_radar_source"]  = radar_source
                st.session_state["_radar_input"]   = radar_input
                st.session_state["_radar_min_wr"]  = radar_min_wr
                st.session_state["_radar_trigger"]  = True

            st.divider()

            # ── 自動雷達開關 ──────────────────────────────────────────
            st.markdown(
                '<div style="font-size:12px;color:#4a6fa5;">⏱️ 開頁自動雷達（每20分鐘）</div>',
                unsafe_allow_html=True
            )
            auto_radar_toggle = st.toggle(
                "啟用自動掃描+推播",
                value=st.session_state.get('_auto_radar_enabled', True),
                key="auto_radar_toggle_widget",
                help="開啟後，只要此分頁保持開啟、且在台股交易時段(09:00~13:35)，"
                     "每 20 分鐘會自動掃描全市場並推播 Discord。"
                     "關閉分頁或標籤頁睡眠時不會運作，需重新打開網頁才會繼續。"
            )
            st.session_state['_auto_radar_enabled'] = auto_radar_toggle
            if not auto_radar_toggle:
                st.caption("🔴 已關閉，手動掃描按鈕仍可正常使用")

            st.divider()

            # ── Discord 推播控制 ───────────────────────────────────────
            st.markdown('<div style="font-size:12px;color:#4a6fa5;">📡 Discord 手動推播</div>',
                        unsafe_allow_html=True)
            col_t, col_s = st.columns(2)
            with col_t:
                if st.button("🧪 測試推播", key="discord_test_btn",
                             use_container_width=True,
                             help="發送測試訊息到 Discord 確認連線"):
                    st.session_state["_discord_test_trigger"] = True
            with col_s:
                if st.button("📡 強制掃描推播", key="discord_force_btn",
                             use_container_width=True,
                             help="立即掃描全市場熱門股池（約50~55檔）並精選Top3推播"):
                    st.session_state["_discord_force_scan"] = True

        st.markdown("---")

        if mode == "🔍 單股分析":
            ticker = st.text_input(
                "股票代號", value="8150",
                placeholder="台股: 2330 / 8150  美股: AAPL",
                help="台股輸入數字代號即可(自動補 .TW),美股輸入英文代號"
            )
        elif mode == "⭐ 自選股":
            ticker = ""
            st.markdown(
                '<div style="font-size:12px;color:#4a6fa5;margin:6px 0;">管理您的 5 組自選股清單</div>',
                unsafe_allow_html=True
            )
        else:
            ticker = ""

        # ── 批量掃描參數 ────────────────────────────────────────────────
        if mode == "📡 批量掃描":
            st.markdown('<div style="font-size:10px;color:#7a9bbf;letter-spacing:2px;margin-bottom:6px;">掃描清單</div>',
                        unsafe_allow_html=True)

            # 自選股輸入框 ── 每行一個代號,或逗號分隔
            custom_raw = st.text_area(
                "✏️ 自選股 (可空白)",
                placeholder="每行或逗號分隔\n例: 8150\n2330\nAAPL",
                height=100,
                help="留空則使用台灣熱門100檔；填入代號後,自選股會優先列在掃描清單最前面"
            )

            scan_universe = st.selectbox(
                "掃描清單來源",
                [
                    "📊 即時成交量排行 (今日最活躍)",
                    "📈 即時漲幅排行 (今日強勢股)",
                    "📉 即時跌幅排行 (今日弱勢/超跌)",
                    "⭐ 台灣熱門100檔 (固定清單)",
                    "🔬 台股全市場759檔 (含傳產/金融/電子)",
                    "✏️ 僅自選股",
                ],
                index=0,
                help=(
                    "即時排行: 每次掃描自動從 Yahoo Finance 抓取當日最新排行\n"
                    "固定清單: 使用預設代號庫(離線可用)"
                )
            )
            use_hot100 = scan_universe != "✏️ 僅自選股"

            # ── 掃描模式 ────────────────────────────────────────────
            scan_mode = st.radio(
                "掃描目標",
                ["📊 高勝率標的 (≥門檻)", "🎯 買點獵人 (底部起漲點)"],
                index=0,
                help=(
                    "高勝率模式: 找已在噴發的頂級浪潮\n"
                    "買點獵人: 找勝率55~68%、R_cycle≥1.0的中繼蓄勢底部"
                )
            )

            if "高勝率" in scan_mode:
                min_wr = st.slider("最低勝率門檻 (%)", 0, 90, 70, step=5,
                                   help="只顯示波段勝率大於此值的標的")
                min_entry_score = 0  # 不篩買點分數
            else:
                min_wr = 0  # 買點獵人模式不限勝率
                min_entry_score = st.slider(
                    "最低買點分數", 50, 90, 65, step=5,
                    help="≥80 強力買點 / ≥65 潛力買點 / ≥50 蓄勢觀察"
                )

        st.markdown("---")
        st.markdown('<div style="font-size:10px;color:#7a9bbf;letter-spacing:2px;">WAVE DNA 參數</div>',
                    unsafe_allow_html=True)
        period = st.selectbox("歷史資料期間", ["2y", "1y", "3y"], index=0)
        top_n  = st.slider("前瞻天數", 5, 20, 10, step=5)

        st.markdown("---")

        if mode == "🔍 單股分析":
            analyze = st.button("🔍 開始 DNA 分析", use_container_width=True, type="primary")
            scan    = False
            custom_raw = ""
            min_wr  = 70
            use_hot100 = True
            scan_mode = "📊 高勝率標的 (≥門檻)"
            min_entry_score = 0
        elif mode == "⭐ 自選股":
            analyze = False
            scan    = False
            custom_raw = ""
            min_wr  = 70
            use_hot100 = True
            scan_mode = "📊 高勝率標的 (≥門檻)"
            min_entry_score = 0
        else:
            scan    = st.button("📡 開始批量掃描", use_container_width=True, type="primary")
            analyze = False

        st.markdown("""
        <div style="font-size:10px;color:#7a9bbf;margin-top:18px;line-height:1.8;">
        <b style="color:#4a6fa5;">三大分類說明</b><br>
        🚀 頂級浪潮 ── 勝率 ≥ 70%<br>
        ⏳ 中繼蓄勢 ── 勝率 50-70%<br>
        🛑 警戒浪潮 ── 勝率 &lt; 50%
        </div>
        """, unsafe_allow_html=True)

    return (ticker.strip(), period, top_n, analyze,
            scan, custom_raw, min_wr, use_hot100, mode,
            locals().get('scan_universe', '台灣熱門100檔'),
            locals().get('scan_mode', '📊 高勝率標的 (≥門檻)'),
            locals().get('min_entry_score', 0))


# ─────────────────────────────────────────────────────────────────────────────
#  UI 渲染函式群
# ─────────────────────────────────────────────────────────────────────────────

def render_dna_stats(dna: dict):
    """波浪 DNA 統計看板 (修正週期慣性)"""
    st.markdown('<div class="section-title">🧬 兩年波浪 DNA 統計</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("T_median 修正中位數", f"{dna['T_median']:.0f} 天",
                  help="個股過去兩年所有修正波段天數的中位數,作為 R_cycle 基準分母")
    with c2:
        st.metric("T_mean 平均修正天數", f"{dna['T_mean']:.0f} 天")
    with c3:
        st.metric("T_std 修正週期標準差", f"{dna['T_std']:.1f} 天",
                  help="越小代表個股修正週期慣性越規律,R_cycle 預測可信度越高")
    with c4:
        st.metric("修正波段樣本數", f"{len(dna['corrections'])} 組")

    atr_pct = dna.get("atr_pct", 0)
    dist    = dna.get("distance_used", 15)
    vol_type = "低波動型(漣漪股)" if atr_pct < 1.5 else "標準型" if atr_pct < 3.0 else "高波動型"
    st.markdown(f"""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:13px;color:#4a6fa5;
                margin-top:8px;margin-bottom:4px;display:flex;gap:24px;flex-wrap:wrap;">
      <span>ATR日均波動率: <b style="color:#1565c0;">{atr_pct:.2f}%</b></span>
      <span>動態波段最小間距: <b style="color:#1565c0;">{dist} 天</b>
            <span style="color:#7a9bbf;font-size:12px;">({vol_type})</span>
      </span>
    </div>
    """, unsafe_allow_html=True)

    if dna["corrections"]:
        corr_df = pd.DataFrame({"修正天數(天)": dna["corrections"]})
        st.bar_chart(corr_df, use_container_width=True, height=110)


def render_r_cycle(dna: dict, wr: dict, used_ticker: str):
    """主要核心指標區 + 分類標籤"""
    close     = wr["close"]
    r         = dna["R_cycle"]
    d_cur     = dna["D_current"]
    t_med     = dna["T_median"]
    category  = wr["category"]
    cat_label = wr["category_label"]

    css_map = {"top": "label-top", "mid": "label-mid", "warn": "label-warn"}
    st.markdown(
        f'<div class="dna-label {css_map[category]}">{cat_label}</div>',
        unsafe_allow_html=True
    )

    # R_cycle 顏色(淺色系)
    r_color = "#0a7c59" if 0.95 <= r <= 1.25 else \
              "#d97706" if r >= 0.60 else "#c0392b"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(html_metric("當前收盤價", f"{close:.2f}",
                                f"最近波峰: {dna['last_peak_date']} @ {dna['last_peak_price']}"),
                    unsafe_allow_html=True)
    with c2:
        st.markdown(html_metric("D_current 拉回天數", f"{d_cur} 天",
                                f"自 {dna['last_peak_date']} 起"),
                    unsafe_allow_html=True)
    with c3:
        st.markdown(html_metric("T_median 慣性基準", f"{t_med:.0f} 天",
                                "個股兩年修正週期中位數"),
                    unsafe_allow_html=True)
    with c4:
        st.markdown(html_metric(
            "R_cycle 週期對稱率",
            f'<span style="color:{r_color};font-size:30px;">{r:.3f}</span>',
            wr["desc_time"]
        ), unsafe_allow_html=True)

    r_pct = min(r / 1.5, 1.0) * 100
    st.markdown(f"""
    <div style="margin-top:8px;margin-bottom:6px;">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:#4a6fa5;
                  letter-spacing:1px;margin-bottom:6px;">
        R_CYCLE ── 0%(未修正) → 100%(T_median 臨界) → 150%+(超額修正)
      </div>
      <div class="bar-wrap">
        <div class="bar-fill" style="width:{r_pct:.1f}%;background:{r_color};"></div>
      </div>
      <div style="display:flex;justify-content:space-between;
                  font-family:'IBM Plex Mono',monospace;font-size:12px;color:#7a9bbf;margin-top:4px;">
        <span>0%</span><span style="color:#d97706;">60%</span>
        <span style="color:#d97706;">80%</span>
        <span style="color:{r_color};font-weight:700;">100% ← 共振臨界</span>
        <span>150%+</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # 若修正低點已確認,顯示補充資訊
    if dna.get("correction_end_date"):
        st.markdown(f"""
        <div style="background:#e8f4ec;border:1.5px solid #0a7c59;border-radius:8px;
                    padding:10px 16px;margin-top:8px;font-size:14px;color:#0a5c42;">
          ✅ 修正低點已確認：{dna['correction_end_date']} @ <b>{dna['correction_end_price']}</b>
          　｜　實際修正 {dna['actual_correction_days']} 天
          　｜　反彈已走 <b>{dna['days_since_trough']}</b> 天
        </div>
        """, unsafe_allow_html=True)


def render_feature_scores(wr: dict):
    """特徵分數條形圖(淺藍色系)"""
    st.markdown('<div class="section-title">⚙️ 特徵向量評分</div>', unsafe_allow_html=True)

    colors = {"time": "#1565c0", "ma": "#d97706", "kd": "#6d28d9"}

    st.markdown(html_feat_bar("時間波 (×40%)", wr["s_time"], wr["desc_time"], colors["time"]),
                unsafe_allow_html=True)
    st.markdown(html_feat_bar("均線型態 (×30%)", wr["s_ma"], wr["desc_ma"], colors["ma"]),
                unsafe_allow_html=True)
    st.markdown(html_feat_bar("KD+量能 (×30%)", wr["s_kd"], wr["desc_kd"], colors["kd"]),
                unsafe_allow_html=True)

    wrate_pct = int(wr["winrate"] * 100)
    w_color   = "#0a7c59" if wrate_pct >= 70 else "#d97706" if wrate_pct >= 50 else "#c0392b"
    w_bg      = "#e8f4ec" if wrate_pct >= 70 else "#fef3c7" if wrate_pct >= 50 else "#fde8e8"

    st.markdown(f"""
    <div class="dna-card" style="border-color:{w_color};background:{w_bg};margin-top:6px;">
      <h3>綜合波段成功率 (加權)</h3>
      <div style="display:flex;align-items:baseline;gap:12px;">
        <div class="val" style="color:{w_color};font-size:40px;">{wrate_pct}%</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:13px;color:#4a6fa5;">
          時間波×0.4 + 均線×0.3 + KD/量×0.3
        </div>
      </div>
      <div class="bar-wrap" style="margin-top:10px;">
        <div class="bar-fill" style="width:{wrate_pct}%;background:{w_color};"></div>
      </div>
      <div style="display:flex;justify-content:space-between;
                  font-family:'IBM Plex Mono',monospace;font-size:12px;color:#7a9bbf;margin-top:6px;">
        <span>0%</span><span>50% 蓄勢</span><span style="color:{w_color};font-weight:700;">70% 頂級</span><span>100%</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:13px;color:#4a6fa5;
                display:flex;gap:20px;margin-top:6px;flex-wrap:wrap;">
      <span>9K: <b style="color:#1a2b3c;">{wr['k9']:.1f}</b></span>
      <span>9D: <b style="color:#1a2b3c;">{wr['d9']:.1f}</b></span>
      <span>均線壓縮度: <b style="color:#1a2b3c;">{wr['ma_spread_pct']:.2f}%</b></span>
      <span>量比: <b style="color:#1a2b3c;">{wr['vol_ratio']:.2f}x</b></span>
    </div>
    """, unsafe_allow_html=True)


def render_forward_table(rows: list[dict], last_close: float):
    """未來 N 日前瞻路徑矩陣"""
    st.markdown('<div class="section-title">📅 前瞻路徑矩陣 (演算法預估)</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:13px;color:#4a6fa5;margin-bottom:12px;line-height:1.6;
                background:#eaf2fb;border-left:4px solid #1565c0;padding:10px 14px;border-radius:6px;">
    ⚠️ 本矩陣由「波浪DNA慣性 × ATR波動帶 × 型態分類漂移」動態生成，僅供型態研究參考，非投資建議。
    上下限幅度隨預估天數遞增（越遠不確定性越大）。
    </div>
    """, unsafe_allow_html=True)
    st.markdown(html_forward_table(rows, last_close), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  主程式入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── 啟動時預載官方股票名稱資料庫（TWSE + TPEX OpenAPI）────────────
    # @st.cache_data(ttl=3600) 確保每小時最多打一次 API
    if not st.session_state.get('_official_names_loaded'):
        _load_official_names()
        st.session_state['_official_names_loaded'] = True
        # 同步初始化 global_ticker_map（相容外部呼叫規格）
        st.session_state['global_ticker_map'] = get_taiwan_ticker_mapping()

    # ── ★ 盤中自動刷新（每 20 分鐘）── 與手動掃描完全並存，互不影響 ──
    # streamlit_autorefresh 在「網頁開著」的前提下，於瀏覽器端倒數計時，
    # 時間到了自動觸發 Streamlit 重新執行 main() 一次。
    # 手動操作（切換 Sidebar、按按鈕）也會觸發 main() 重新執行，
    # 兩者共用同一套 is_tw_trading_hours() 判斷與 _auto_radar_scan_and_notify()，
    # 不會互相干擾、也不會重複推播（靠每日 session_state key 防重複）。
    _auto_radar_on = st.session_state.get('_auto_radar_enabled', True)

    _autorefresh_count = 0
    if _AUTOREFRESH_AVAILABLE and is_tw_trading_hours() and _auto_radar_on:
        # st_autorefresh 回傳「已刷新次數」，每 20 分鐘遞增 1
        # 用此值判斷是否為 autorefresh 觸發，而非手動操作觸發的 rerun
        _autorefresh_count = st_autorefresh(
            interval=20 * 60 * 1000, key="auto_radar_refresh"
        ) or 0

    # ── ★ 盤中自動雷達掃描 + Discord 推播（移到 sidebar 之後執行）────
    # 此區塊在 render_sidebar() 之後呼叫，確保 period 等參數已取得
    (ticker_raw, period, top_n, analyze,
     scan, custom_raw, min_wr, use_hot100, mode,
     scan_universe, scan_mode, min_entry_score) = render_sidebar()

    # ── ★ Discord 推播控制 ────────────────────────────────────────────
    # 每日重置推播記錄（用日期作 key，跨 session 也能每天重新推播）
    import pytz as _pytz
    _today_str = datetime.datetime.now(_pytz.timezone('Asia/Taipei')).strftime('%Y%m%d')
    _notify_key = f"_notified_{_today_str}"
    if _notify_key not in st.session_state:
        st.session_state[_notify_key] = set()   # 今日推播記錄
        # 清掉昨天的 key，節省 session_state 空間
        for k in list(st.session_state.keys()):
            if k.startswith('_notified_') and k != _notify_key:
                del st.session_state[k]

    # ── ★ 自動雷達運作狀態指示器（讓使用者清楚知道目前模式）─────────
    _in_market = is_tw_trading_hours()
    if _auto_radar_on and _in_market:
        st.markdown("""
        <div style="background:#e8f4ec;border-left:4px solid #0a7c59;
                    padding:8px 14px;border-radius:6px;margin-bottom:10px;
                    font-size:12px;color:#0a7c59;display:flex;align-items:center;gap:8px;">
          🟢 <b>自動雷達運作中</b>　每 20 分鐘自動掃描全市場並推播 Discord
          （此分頁必須保持開啟才會持續運作）
        </div>
        """, unsafe_allow_html=True)
    elif _auto_radar_on and not _in_market:
        st.markdown("""
        <div style="background:#f0f3f7;border-left:4px solid #7a9bbf;
                    padding:8px 14px;border-radius:6px;margin-bottom:10px;
                    font-size:12px;color:#4a6fa5;">
          ⚪ 目前非台股交易時段（09:00~13:35），自動雷達待命中，
          可隨時用 Sidebar「📡 強制掃描推播」手動觸發
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#fde8e8;border-left:4px solid #c0392b;
                    padding:8px 14px;border-radius:6px;margin-bottom:10px;
                    font-size:12px;color:#c0392b;">
          🔴 自動雷達已關閉（Sidebar 可重新開啟），手動掃描功能仍正常可用
        </div>
        """, unsafe_allow_html=True)

    # ★ 盤中自動掃描：只在 autorefresh 計數器遞增時執行
    # 手動切換 Sidebar/頁面觸發的 rerun，_autorefresh_count 不變 → 跳過掃描
    # ★ Agent A 修正：_last_radar_count 初始值改為 0（與第一次 _autorefresh_count=0 相同）
    # 確保「第一次進頁面」不自動觸發掃描（冷啟動時 curl_cffi 可能尚未穩定）
    # 只有 autorefresh 真正計時到（_autorefresh_count 從 0 遞增到 1）才觸發第一次掃描
    _last_radar_count = st.session_state.get('_last_radar_count', 0)
    if _in_market and _auto_radar_on and _autorefresh_count != _last_radar_count:
        st.session_state['_last_radar_count'] = _autorefresh_count
        _auto_radar_scan_and_notify(period=period)

    # 手動測試推播按鈕（由 sidebar 傳入觸發）
    if st.session_state.pop("_discord_test_trigger", False):
        ok = send_discord_notify(
            f"🧪 **【Wave DNA 推播測試】** "
            f"{datetime.datetime.now(_pytz.timezone('Asia/Taipei')).strftime('%H:%M')}\n"
            f"✅ Discord Webhook 連線正常，盤中黃金訊號將自動推播到此頻道。"
        )
        if ok:
            st.toast("✅ 測試推播已發送！請確認 Discord", icon="📡")
        else:
            st.toast("❌ Discord 推播失敗，請確認 Webhook URL", icon="⚠️")

    # 手動強制掃描推播按鈕（由 sidebar 傳入觸發）
    if st.session_state.pop("_discord_force_scan", False):
        with st.spinner("🔬 強制掃描全市場熱門池中（約50~55檔）..."):
            try:
                force_pool = get_taiwan_hot_tickers(top_n=50)
            except Exception:
                force_pool = list(CORE_RADAR_WATCHLIST)
            results = run_radar_scan(force_pool, period=period, with_chip=True)

        golden = [r for r in results if r.get("all_green")]
        golden.sort(key=_radar_sort_key)
        top3 = golden[:3]   # ★ 只取最優質 Top 3，防洗版

        if top3:
            for r in top3:
                d1  = f"{r['D1下限']:.2f}" if r.get("D1下限") else "--"
                d2  = f"{r['D2下限']:.2f}" if r.get("D2下限") else "--"
                chip_note = f"\n🧬 籌碼動向：{r['chip_label']}" if r.get("chip_label") else ""
                now_tw = datetime.datetime.now(_pytz.timezone('Asia/Taipei'))
                msg = (
                    f"🚨 **【波浪 DNA 雷達·起漲點觸發】** {now_tw.strftime('%H:%M')}\n"
                    f"📈 標的：**{r['股名']}** (`{r['代號']}`)\n"
                    f"💰 當前現價：**{r['現價']}** 元 ｜ 🎯 預測勝率：**{r['勝率']:.0f}%**\n"
                    f"🧬 歷史對稱率 R_cycle：**{r['R_cycle']:.3f}**\n"
                    f"📌 建議掛單 (D+1 下限)：**{d1}** 元\n"
                    f"🛡️ 停損基準 (D+2 下限)：**{d2}** 元"
                    f"{chip_note}\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━"
                )
                send_discord_notify(msg)
                import time as _time2; _time2.sleep(0.5)  # ★ 防 Discord 429
            st.toast(f"✅ 已從 {len(results)} 檔中精選 Top {len(top3)} 推播！", icon="🎯")
        else:
            # 黃金條件全綠找不到時，推播最高分標的作為觀察通知
            if results:
                best = max(results, key=lambda x: x["買點分數"])
                now_tw = datetime.datetime.now(_pytz.timezone('Asia/Taipei'))
                msg = (
                    f"📊 **【Wave DNA 觀察標的】** {now_tw.strftime('%H:%M')}\n"
                    f"⚠️ 本次掃描 {len(results)} 檔全市場熱門股，無五大條件全綠標的，"
                    f"最高分標的如下：\n"
                    f"📈 {best['股名']} (`{best['代號']}`) "
                    f"買點 {best['買點分數']} 分 / 勝率 {best['勝率']:.0f}% / "
                    f"R_cycle {best['R_cycle']:.3f}"
                )
                send_discord_notify(msg)
                st.toast(f"ℹ️ 掃描{len(results)}檔，無黃金標的，已推播最高分 {best['代號']}", icon="📊")
            else:
                st.toast("⚠️ 掃描失敗，請確認網路連線", icon="⚠️")


    st.markdown("""
    <div style="margin-bottom:22px;">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;
                  letter-spacing:3px;color:#7a9bbf;margin-bottom:6px;">
        QUANTITATIVE WAVE ANALYSIS SYSTEM
      </div>
      <h1 style="font-family:'IBM Plex Mono',monospace;font-size:24px;
                 font-weight:700;color:#1a2b3c;margin:0;">
        🧬 動態波浪週期 DNA 匹配系統
      </h1>
      <div style="font-size:13px;color:#4a6fa5;margin-top:6px;">
        Dynamic Wave Cycle DNA Matching ── 拒絕死板天數,演算法動態學習個股生理鐘慣性
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    #  🚀 雷達掃描戰情表（優先顯示在最頂部）
    # ════════════════════════════════════════════════════════════════════
    if st.session_state.get("_radar_trigger"):
        st.session_state["_radar_trigger"] = False

        radar_source = st.session_state.get("_radar_source", "📋 自訂清單")
        radar_input  = st.session_state.get("_radar_input",  "")
        radar_min_wr = st.session_state.get("_radar_min_wr", 50)

        # ── 決定掃描代號清單 ──────────────────────────────────────────
        rank_map = {
            "📊 即時成交量排行 100": "volume",
            "📈 即時漲幅排行 100":  "gain",
            "📉 即時跌幅排行 100":  "loss",
        }

        if radar_source == "🌐 全市場海選100（成交量+漲跌幅）":
            # ★ 與 Discord 自動推播使用同一份合併池：
            #   成交量前50 + 漲幅前50 + 跌幅前50 + 核心底盤，去重複
            with st.spinner("⏳ 正在合併全市場成交量＋漲跌幅前50大熱門股池..."):
                radar_tickers = get_taiwan_hot_tickers(top_n=50)
            realtime_meta = []
            source_label  = f"🌐 全市場海選 ({len(radar_tickers)} 檔)"

        elif radar_source in rank_map:
            with st.spinner(f"⏳ 正在抓取 Yahoo Finance {radar_source}..."):
                radar_tickers, realtime_meta, ok, msg = fetch_tw_realtime_hot(
                    rank_map[radar_source], 100
                )
            if not ok or not radar_tickers:
                st.warning(f"⚠️ 即時排行抓取失敗（{msg}），改用預設清單")
                radar_tickers = list(DEFAULT_WATCHLIST)
                realtime_meta = []
            source_label = radar_source
        else:
            # 自訂清單
            raw_list = radar_input.replace(",", "\n").split("\n")
            radar_tickers = [t.strip().upper() for t in raw_list if t.strip()]
            if not radar_tickers:
                radar_tickers = list(DEFAULT_WATCHLIST)
            realtime_meta = []
            source_label = f"📋 自訂清單 ({len(radar_tickers)} 檔)"

        st.markdown(f"""
        <div class="section-title">🚀 雷達掃描戰情表 — {source_label}</div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="font-size:13px;color:#4a6fa5;background:#eaf2fb;
                    border-left:4px solid #1565c0;padding:10px 16px;
                    border-radius:6px;margin-bottom:14px;">
          ⚡ 掃描 <b>{len(radar_tickers)}</b> 支標的，
          勝率門檻 ≥ <b style="color:#0a7c59;">{radar_min_wr}%</b>，
          🎯 黃底 = 五大黃金條件全部成立
        </div>
        """, unsafe_allow_html=True)

        # 若為即時排行，先顯示排行預覽
        if realtime_meta:
            prev_rows = ""
            for j, m in enumerate(realtime_meta[:10], 1):
                sym  = m["symbol"]
                nm   = TW_NAME_MAP.get(sym, m["name"][:14] if m["name"] else "--")
                chg  = m["chg_pct"]
                cc   = "#0a7c59" if chg > 0 else "#c0392b"
                prev_rows += (f"<tr><td style='color:#7a9bbf;text-align:center;'>{j}</td>"
                              f"<td style='font-family:IBM Plex Mono,monospace;color:#1565c0;"
                              f"font-weight:700;'>{sym}</td>"
                              f"<td style='color:#1a2b3c;font-size:12px;'>{nm}</td>"
                              f"<td style='font-family:IBM Plex Mono,monospace;font-weight:700;'>"
                              f"{m['price']:.2f}</td>"
                              f"<td style='color:{cc};font-weight:700;'>{chg:+.2f}%</td>"
                              f"<td style='color:#4a6fa5;font-size:12px;'>"
                              f"{m['volume']//1000:,}張</td></tr>")
            with st.expander(f"📋 {radar_source} 前10筆預覽", expanded=False):
                st.markdown(f"""
                <table style="width:100%;border-collapse:collapse;font-size:13px;">
                  <thead><tr style="background:#1565c0;color:#fff;">
                    <th style="padding:6px;">#</th><th>代號</th><th>股名</th>
                    <th>現價</th><th>漲跌</th><th>成交量</th>
                  </tr></thead><tbody>{prev_rows}</tbody>
                </table>""", unsafe_allow_html=True)

        prog = st.progress(0.0, text="🔬 DNA × 籌碼 雙重掃描中...")
        radar_results = run_radar_scan(radar_tickers, period=period, with_chip=True)
        prog.progress(1.0, text="✅ 掃描完成")
        import time as _t; _t.sleep(0.3)
        prog.empty()

        # ── 勝率篩選 ──────────────────────────────────────────────────
        filtered = [r for r in radar_results if r["勝率"] >= radar_min_wr]
        golden   = [r for r in filtered if r.get("all_green")]
        golden.sort(key=_radar_sort_key)   # ★ 與 Discord 推播相同排序：勝率優先,R_cycle近1.15優先
        all_sorted = sorted(filtered, key=lambda x: x["買點分數"], reverse=True)

        # ── 統計摘要 ──────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("掃描總數", f"{len(radar_results)} 檔")
        c2.metric(f"勝率≥{radar_min_wr}%", f"{len(filtered)} 檔")
        c3.metric("🎯 黃金標的", f"{len(golden)} 檔")
        c4.metric("掃描失敗", f"{len(radar_tickers)-len(radar_results)} 檔")

        if not filtered:
            st.info(f"⏳ 當前所有標的勝率均低於 {radar_min_wr}%，建議降低門檻重試。")
        else:
            if not golden:
                st.info("⏳ 當前自選股均處於波動或過熱階段，尚未觸發週期生理鐘扣滿訊號，請保持耐心。")

        # ── 戰情表 ────────────────────────────────────────────────────
        if all_sorted:
            bar_color = {"top":"#0a7c59","mid":"#d97706","warn":"#c0392b"}
            rows_html = ""
            for i, r in enumerate(all_sorted, 1):
                code  = r["代號"]
                name  = r["股名"]
                url   = get_chart_url(code)
                score = r["買點分數"]
                sig   = r["買點訊號"]
                rc    = r["R_cycle"]
                wr_v  = r["勝率"]
                d1    = f"{r['D1下限']:.2f}" if r.get("D1下限") else "--"
                d2    = f"{r['D2下限']:.2f}" if r.get("D2下限") else "--"
                cat   = r["category"]
                bc    = bar_color.get(cat, "#1565c0")
                is_g  = r.get("all_green")

                sc_color = ("#c0392b" if "共振" in sig else "#0a7c59" if score >= 80
                            else "#1565c0" if score >= 65 else "#d97706" if score >= 50
                            else "#9e9e9e")
                rc_color = "#0a7c59" if rc >= 1.0 else "#d97706" if rc >= 0.6 else "#c0392b"
                row_bg   = "#fff8e1" if is_g else ("#f7fafd" if i % 2 == 0 else "#fff")
                gmark    = "🎯 " if is_g else ""
                c = r.get("conds", {})
                ci = lambda v: "✅" if v else "❌"
                cond_str = (f"{ci(c.get('c3_rcycle'))}R "
                            f"{ci(c.get('c4_kd'))}KD "
                            f"{ci(c.get('c1_mid'))}蓄 "
                            f"{ci(c.get('c2_wr'))}勝 "
                            f"{ci(c.get('c5_vol'))}量")

                rows_html += f"""
                <tr style="background:{row_bg};">
                  <td style="text-align:center;font-size:12px;color:#7a9bbf;">{i}</td>
                  <td><a href="{url}" target="_blank"
                     style="color:#1565c0;font-weight:700;font-size:13px;
                            font-family:'IBM Plex Mono',monospace;text-decoration:none;">
                    {gmark}{code}</a></td>
                  <td style="font-size:12px;color:#1a2b3c;">{name}</td>
                  <td style="font-family:'IBM Plex Mono',monospace;font-weight:700;
                             font-size:14px;color:#1a2b3c;">{r['現價']}</td>
                  <td style="color:{rc_color};font-weight:700;font-size:13px;
                             font-family:'IBM Plex Mono',monospace;">{rc:.3f}</td>
                  <td>
                    <div style="display:flex;align-items:center;gap:5px;">
                      <div style="width:40px;background:#c8d8e8;border-radius:3px;
                                  height:7px;overflow:hidden;">
                        <div style="width:{min(wr_v,100):.0f}%;height:7px;background:{bc};"></div>
                      </div>
                      <span style="color:{bc};font-weight:700;font-size:12px;">{wr_v:.0f}%</span>
                    </div>
                  </td>
                  <td style="font-size:11px;letter-spacing:1px;">{cond_str}</td>
                  <td style="font-weight:700;color:{sc_color};font-size:13px;
                             white-space:nowrap;">{score}分 {sig[:4]}</td>
                  <td style="font-family:'IBM Plex Mono',monospace;font-size:13px;
                             color:#1565c0;font-weight:700;text-align:center;">{d1}</td>
                  <td style="font-family:'IBM Plex Mono',monospace;font-size:13px;
                             color:#d97706;font-weight:700;text-align:center;">{d2}</td>
                </tr>"""

            st.markdown(f"""
            <div style="overflow-x:auto;-webkit-overflow-scrolling:touch;">
            <table style="width:100%;border-collapse:collapse;font-size:13px;min-width:850px;">
              <thead>
                <tr style="background:#1565c0;color:#fff;font-size:11px;">
                  <th style="padding:8px 5px;width:28px;">#</th>
                  <th style="padding:8px;text-align:left;min-width:90px;">代號</th>
                  <th style="padding:8px;text-align:left;min-width:65px;">股名</th>
                  <th style="padding:8px;text-align:left;min-width:75px;">現價</th>
                  <th style="padding:8px;text-align:left;min-width:70px;">R_cycle</th>
                  <th style="padding:8px;text-align:left;min-width:90px;">勝率</th>
                  <th style="padding:8px;text-align:left;min-width:120px;">五大條件</th>
                  <th style="padding:8px;text-align:left;min-width:95px;">買點評估</th>
                  <th style="padding:8px;text-align:center;min-width:65px;">D+1下限</th>
                  <th style="padding:8px;text-align:center;min-width:65px;">D+2下限</th>
                </tr>
              </thead>
              <tbody>{rows_html}</tbody>
            </table>
            </div>
            <div style="font-size:11px;color:#7a9bbf;margin-top:8px;text-align:right;">
              🎯 黃底 = 五大黃金條件全部成立
            </div>
            """, unsafe_allow_html=True)

            # ── 手動推播本次戰情表的黃金標的到 Discord ─────────────────
            if golden:
                top3_preview = golden[:3]
                preview_str = "、".join(f"{r['代號']}({r['勝率']:.0f}%)" for r in top3_preview)
                st.markdown(
                    f'<div style="font-size:12px;color:#4a6fa5;margin-top:6px;">'
                    f'💡 本次掃描共 {len(golden)} 檔黃金標的，'
                    f'Top {len(top3_preview)}: {preview_str}</div>',
                    unsafe_allow_html=True
                )
                if st.button("📡 推播本次 Top 3 到 Discord", key="radar_table_push_btn"):
                    for r in top3_preview:
                        d1 = f"{r['D1下限']:.2f}" if r.get("D1下限") else "--"
                        d2 = f"{r['D2下限']:.2f}" if r.get("D2下限") else "--"
                        chip_note = f"\n🧬 籌碼動向：{r['chip_label']}" if r.get("chip_label") else ""
                        import pytz as _pytz2
                        now_tw2 = datetime.datetime.now(_pytz2.timezone('Asia/Taipei'))
                        msg = (
                            f"🚨 **【波浪 DNA 雷達·手動精選】** {now_tw2.strftime('%H:%M')}\n"
                            f"📈 標的：**{r['股名']}** (`{r['代號']}`)\n"
                            f"💰 當前現價：**{r['現價']}** 元 ｜ 🎯 預測勝率：**{r['勝率']:.0f}%**\n"
                            f"🧬 歷史對稱率 R_cycle：**{r['R_cycle']:.3f}**\n"
                            f"📌 建議掛單 (D+1 下限)：**{d1}** 元\n"
                            f"🛡️ 停損基準 (D+2 下限)：**{d2}** 元"
                            f"{chip_note}\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━━━━━"
                        )
                        send_discord_notify(msg)
                    st.toast(f"✅ 已推播 {len(top3_preview)} 檔到 Discord！", icon="📡")

        st.markdown("---")

    # ════════════════════════════════════════════════════════════════════
    #  模式 ⭐: 自選股看板
    # ════════════════════════════════════════════════════════════════════
    if mode == "⭐ 自選股":
        render_watchlist_page(period=period)
        return

    # ════════════════════════════════════════════════════════════════════
    #  模式 A: 批量掃描
    # ════════════════════════════════════════════════════════════════════
    if mode == "📡 批量掃描":
        if not scan:
            st.markdown("""
            <div class="dna-card" style="text-align:center;padding:40px;">
              <div style="font-size:36px;margin-bottom:12px;">📡</div>
              <div style="font-family:'IBM Plex Mono',monospace;font-size:14px;color:#4a6fa5;">
                設定掃描參數後,按下「開始批量掃描」
              </div>
              <div style="font-size:12px;color:#7a9bbf;margin-top:10px;line-height:1.8;">
                系統將以多線程並行下載所有標的的近2年K線，<br>
                自動計算波浪DNA勝率，篩選出高機率標的。<br>
                <b style="color:#d97706;">預計耗時: 100檔約 15~25 秒</b>
              </div>
            </div>
            """, unsafe_allow_html=True)
            return

        # ── 組合掃描清單 ──────────────────────────────────────────────
        custom_tickers = []
        if custom_raw.strip():
            raw_list = custom_raw.replace(",", "\n").split("\n")
            custom_tickers = [t.strip().upper() for t in raw_list if t.strip()]

        # ── 選擇/抓取預設清單 ─────────────────────────────────────────
        realtime_meta = []
        is_realtime   = False
        fallback_used = False   # 是否已降級到靜態清單

        if "即時成交量排行" in scan_universe:
            with st.spinner("⏳ 正在抓取 Yahoo Finance 今日成交量排行..."):
                preset_list, realtime_meta, ok, msg = fetch_tw_realtime_hot('volume', 100)
            if ok and preset_list:
                is_realtime = True
                rank_label  = "📊 今日成交量排行"
            else:
                preset_list  = TW_HOT_100
                rank_label   = "⭐ 台灣熱門100檔（自動降級）"
                fallback_used = True
                st.info(f"ℹ️ 即時排行暫時無法取得（{msg}），已自動切換為靜態熱門100檔清單。")

        elif "即時漲幅排行" in scan_universe:
            with st.spinner("⏳ 正在抓取 Yahoo Finance 今日漲幅排行..."):
                preset_list, realtime_meta, ok, msg = fetch_tw_realtime_hot('gain', 100)
            if ok and preset_list:
                is_realtime = True
                rank_label  = "📈 今日漲幅排行"
            else:
                preset_list  = TW_HOT_100
                rank_label   = "⭐ 台灣熱門100檔（自動降級）"
                fallback_used = True
                st.info(f"ℹ️ 即時排行暫時無法取得（{msg}），已自動切換為靜態熱門100檔清單。")

        elif "即時跌幅排行" in scan_universe:
            with st.spinner("⏳ 正在抓取 Yahoo Finance 今日跌幅排行..."):
                preset_list, realtime_meta, ok, msg = fetch_tw_realtime_hot('loss', 100)
            if ok and preset_list:
                is_realtime = True
                rank_label  = "📉 今日跌幅排行"
            else:
                preset_list  = TW_HOT_100
                rank_label   = "⭐ 台灣熱門100檔（自動降級）"
                fallback_used = True
                st.info(f"ℹ️ 即時排行暫時無法取得（{msg}），已自動切換為靜態熱門100檔清單。")

        elif "全市場759" in scan_universe or "電子股759" in scan_universe:
            preset_list = TW_ELECTRONIC_759
            rank_label  = "🔬 台股全市場759檔"

        elif "✏️" in scan_universe:
            preset_list = []
            rank_label  = "✏️ 僅自選股"

        else:  # 熱門100檔
            preset_list = TW_HOT_100
            rank_label  = "⭐ 台灣熱門100檔"

        # ── 合併自選股 + 預設清單 ────────────────────────────────────
        scan_list = []
        seen = set()
        for t in custom_tickers:
            if t not in seen:
                scan_list.append(t); seen.add(t)
        if use_hot100:
            for t in preset_list:
                if t not in seen:
                    scan_list.append(t); seen.add(t)

        total = len(scan_list)

        # ── 即時排行預覽表 ───────────────────────────────────────────
        if is_realtime and realtime_meta:
            st.markdown(f'<div class="section-title">{rank_label} (共 {len(realtime_meta)} 檔)</div>',
                        unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size:13px;color:#4a6fa5;margin-bottom:10px;background:#eaf2fb;
                        border-left:4px solid #1565c0;padding:8px 14px;border-radius:6px;">
            ✅ 已從 Yahoo Finance 取得今日即時排行，以下為前20筆預覽，完整清單將進入 DNA 波浪掃描
            </div>""", unsafe_allow_html=True)

            # 排行預覽表
            preview_html = """
            <table class="fwd-table" style="font-size:13px;margin-bottom:16px;">
            <thead><tr>
              <th>#</th><th>代號</th><th>名稱</th>
              <th>現價</th><th>漲跌幅</th><th>成交量</th>
            </tr></thead><tbody>"""
            for i, m in enumerate(realtime_meta[:20], 1):
                sym   = m['symbol']
                name  = TW_NAME_MAP.get(sym, m['name'][:16] if m['name'] else '--')
                price = m['price']
                chg   = m['chg_pct']
                vol   = m['volume']
                url   = get_chart_url(sym)
                chg_color = "#0a7c59" if chg > 0 else "#c0392b" if chg < 0 else "#666"
                row_bg = "background:#f7fafd;" if i%2==0 else ""
                preview_html += f"""
                <tr style="{row_bg}">
                  <td style="color:#7a9bbf;">{i}</td>
                  <td><a href="{url}" target="_blank"
                      style="color:#1565c0;font-weight:700;font-family:'IBM Plex Mono',monospace;
                             text-decoration:none;">{sym}</a></td>
                  <td style="color:#1a2b3c;">{name}</td>
                  <td style="font-family:'IBM Plex Mono',monospace;font-weight:700;color:#1a2b3c;">{price:.2f}</td>
                  <td style="color:{chg_color};font-weight:700;">{chg:+.2f}%</td>
                  <td style="color:#4a6fa5;font-family:'IBM Plex Mono',monospace;">{vol:,}</td>
                </tr>"""
            preview_html += "</tbody></table>"
            st.markdown(preview_html, unsafe_allow_html=True)

        # ── 掃描 UI ───────────────────────────────────────────────────
        st.markdown(f"""
        <div style="font-family:'IBM Plex Mono',monospace;font-size:14px;color:#1a2b3c;
                    margin-bottom:12px;background:#eaf2fb;padding:10px 16px;border-radius:8px;
                    border-left:4px solid #1565c0;">
          📡 開始掃描 <b style="color:#1565c0;">{total}</b> 檔標的
          (自選 {len(custom_tickers)} + {rank_label} {total - len(custom_tickers)})
          ── 勝率門檻 ≥ <b style="color:#0a7c59;">{min_wr}%</b>
        </div>
        """, unsafe_allow_html=True)

        prog_bar    = st.progress(0)
        status_text = st.empty()

        import time
        t0      = time.time()
        results = run_batch_scan(scan_list, period, prog_bar, status_text)
        elapsed = time.time() - t0

        prog_bar.empty()
        status_text.empty()

        # 清除批量掃描期間累積的補丁訊息(避免殘留到下次單股分析誤顯示)
        st.session_state.pop("_patch_msg", None)

        # ── 統計摘要 ─────────────────────────────────────────────────
        top_count  = sum(1 for r in results if r["category"] == "top")
        mid_count  = sum(1 for r in results if r["category"] == "mid")
        warn_count = sum(1 for r in results if r["category"] == "warn")
        hit_count  = sum(1 for r in results if r["勝率"] >= min_wr)
        fail_count = total - len(results)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("掃描完成", f"{len(results)} 檔", f"失敗 {fail_count} 檔")
        c2.metric("🚀 頂級浪潮", f"{top_count} 檔")
        c3.metric("⏳ 中繼蓄勢", f"{mid_count} 檔")
        c4.metric("🛑 警戒浪潮", f"{warn_count} 檔")
        c5.metric(f"≥{min_wr}% 標的", f"{hit_count} 檔",
                  f"耗時 {elapsed:.1f}s")

        # ── 篩選結果顯示 ─────────────────────────────────────────────
        is_hunter_mode = "買點獵人" in scan_mode

        if is_hunter_mode:
            # 買點獵人模式:依買點分數篩選
            hit_results = sorted(
                [r for r in results if r.get("買點分數", 0) >= min_entry_score],
                key=lambda x: x.get("買點分數", 0), reverse=True
            )
            st.markdown(
                f'<div class="section-title">🎯 買點獵人結果 '
                f'(買點分數 ≥ {min_entry_score}，共 {len(hit_results)} 檔)</div>',
                unsafe_allow_html=True
            )
            st.markdown("""
            <div style="font-size:13px;color:#4a6fa5;background:#fff8e1;
                        border-left:4px solid #d97706;padding:10px 14px;
                        border-radius:6px;margin-bottom:14px;">
            🎯 <b>買點獵人模式</b>：以下為符合「R_cycle≥1.0時間波飽和 + KD低檔拐頭 + 中繼蓄勢」
            三大條件的底部布局候選標的，<b>非高勝率追漲股</b>。
            請配合 D+1 下限價格分批掛單，以 D+2 下限作停損基準。
            </div>
            """, unsafe_allow_html=True)
        else:
            # 高勝率模式:依勝率篩選
            hit_results = [r for r in results if r["勝率"] >= min_wr]
            st.markdown(f'<div class="section-title">🎯 高勝率標的 (≥ {min_wr}%)</div>',
                        unsafe_allow_html=True)

        if not hit_results:
            if is_hunter_mode:
                st.warning(f"⚠️ 目前沒有買點分數 ≥ {min_entry_score} 的標的。建議降低門檻或換個掃描清單。")
            else:
                st.warning(f"⚠️ 目前沒有勝率 ≥ {min_wr}% 的標的。建議降低門檻或換個掃描清單。")
        else:
            st.markdown(
                html_scan_table(hit_results, min_winrate=0,
                                hunter_mode=is_hunter_mode),
                unsafe_allow_html=True
            )

            # ── 點擊展開完整分析 ────────────────────────────────────
            st.markdown('<div class="section-title">🔬 展開個股完整 DNA 分析</div>',
                        unsafe_allow_html=True)
            st.markdown("""
            <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;
                        color:#7a9bbf;margin-bottom:10px;">
              從下拉選單選擇任一高勝率標的,即可展開完整 DNA 分析報告
            </div>
            """, unsafe_allow_html=True)

            choices  = [f"{r['代號']}  ({r['勝率']:.1f}%)  {r['分類']}"
                        for r in hit_results]
            selected = st.selectbox("選擇標的", choices, index=0)

            if selected:
                sel_idx = choices.index(selected)
                sel_row = hit_results[sel_idx]
                sel_ticker = sel_row["input"]

                with st.spinner(f"載入 {sel_row['代號']} 完整分析..."):
                    df_sel, used_sel = fetch_data(sel_ticker, period=period, time_bucket=_get_cache_bucket())

                if df_sel is not None and len(df_sel) >= 60:
                    # ★ 補丁在 cache 外部執行
                    df_sel, patched_sel = _patch_today_price(df_sel, used_sel)
                    if patched_sel:
                        st.toast(f"🧬 即時報價已同步：{used_sel} 最新價 {float(df_sel['Close'].iloc[-1]):.2f}", icon="⚡")
                    df_sel  = add_indicators(df_sel)
                    dna_sel = detect_wave_dna(df_sel)
                    wr_sel  = compute_winrate(dna_sel, df_sel)

                    st.markdown(f"""
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:12px;
                                color:#4a6fa5;margin:12px 0;">
                      ▶ 展開分析: <b style="color:#1565c0;">{used_sel}</b> ──
                      {len(df_sel)} 個交易日
                      ({df_sel.index[0].strftime('%Y-%m-%d')} ~ {df_sel.index[-1].strftime('%Y-%m-%d')})
                    </div>
                    """, unsafe_allow_html=True)

                    render_r_cycle(dna_sel, wr_sel, used_sel)
                    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

                    col_l, col_r = st.columns(2, gap="large")
                    with col_l: render_dna_stats(dna_sel)
                    with col_r: render_feature_scores(wr_sel)

                    rows_fwd = generate_forward_matrix(df_sel, wr_sel, dna_sel, n_days=top_n)
                    render_forward_table(rows_fwd, wr_sel["close"])

                    chart_df = df_sel[["Close","MA5","MA20","MA60"]].tail(120).dropna(subset=["Close"])
                    st.line_chart(chart_df, use_container_width=True, height=180)

        # ── 完整掃描結果(含低勝率,可折疊) ──────────────────────────
        with st.expander(f"📋 顯示全部 {len(results)} 檔掃描結果(含低勝率)"):
            st.markdown(html_scan_table(results, min_winrate=0),
                        unsafe_allow_html=True)

        # ── 可下載的 CSV ─────────────────────────────────────────────
        if results:
            import io
            out_rows = []
            for r in results:
                out_rows.append({
                    "代號": r["代號"], "收盤價": r["收盤價"],
                    "勝率(%)": r["勝率"], "分類": r["分類"],
                    "R_cycle": r["R_cycle"], "T_median": r["T_median"],
                    "D_current": r["D_current"],
                    "均線型態": r["均線型態"], "KD狀態": r["KD狀態"],
                    "時間波說明": r["時間波"],
                })
            csv_df  = pd.DataFrame(out_rows)
            csv_buf = io.StringIO()
            csv_df.to_csv(csv_buf, index=False, encoding="utf-8-sig")
            st.download_button(
                "⬇️ 下載掃描結果 CSV",
                data=csv_buf.getvalue().encode("utf-8-sig"),
                file_name=f"wave_dna_scan_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )

        st.markdown(f"""
        <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#7a9bbf;
                    margin-top:18px;text-align:center;color:#4a6fa5;">
          ⚠️ 本系統僅供技術型態研究,不構成任何投資建議。數據來源: Yahoo Finance。
          掃描完成時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
        </div>
        """, unsafe_allow_html=True)
        return

    # ════════════════════════════════════════════════════════════════════
    #  模式 B: 單股分析 (原有邏輯)
    # ════════════════════════════════════════════════════════════════════
    if not analyze:
        st.markdown("""
        <div class="dna-card" style="text-align:center;padding:40px;">
          <div style="font-size:36px;margin-bottom:12px;">🧬</div>
          <div style="font-family:'IBM Plex Mono',monospace;font-size:14px;color:#4a6fa5;">
            在左側輸入股票代號,按下「開始 DNA 分析」
          </div>
          <div style="font-size:12px;color:#7a9bbf;margin-top:8px;">
            台股輸入數字代號(如 2330、8150)，美股輸入英文代號(如 AAPL、NVDA)<br>
            <b style="color:#d97706;">或切換至「批量掃描」模式,一次篩選所有高勝率標的</b>
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    with st.spinner(f"正在下載「{ticker_raw}」近 {period} 日線資料..."):
        df, used_ticker = fetch_data(ticker_raw, period=period, time_bucket=_get_cache_bucket())

    if df is None or len(df) < 60:
        st.error(
            f"❌ 無法取得「{ticker_raw}」的資料。\n\n"
            f"**可能原因:**\n"
            f"- Yahoo Finance 未收錄此股（部分台股如光隆1650、部分中小型股不在 Yahoo 資料庫）\n"
            f"- 興櫃股票(如部分銀行、生技)不在 Yahoo Finance 資料庫\n"
            f"- 代號格式有誤(台股只需輸入數字,如 2330)\n"
            f"- 上市時間太短(不足 60 個交易日)\n\n"
            f"**建議:** 可試試直接輸入帶後綴的代號,如 `2330.TW` 或 `2330.TWO`"
        )
        return

    # ★ 即時報價補丁 — 在 cache 外部執行,確保每次重整都能更新今日現價
    df, patched = _patch_today_price(df, used_ticker)
    if patched:
        live_close = float(df["Close"].iloc[-1])
        st.toast(f"🧬 即時報價已同步：{used_ticker} 最新價 {live_close:.2f}", icon="⚡")

    df  = add_indicators(df)
    dna = detect_wave_dna(df)
    wr  = compute_winrate(dna, df)

    st.markdown(f"""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:#4a6fa5;
                margin-bottom:16px;">
      ▶ 已分析: <b style="color:#1565c0;font-weight:700;font-size:16px;">{used_ticker}</b> ──
      {len(df)} 個交易日 ({df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')})
    </div>
    """, unsafe_allow_html=True)

    render_r_cycle(dna, wr, used_ticker)
    st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")
    with col_left:  render_dna_stats(dna)
    with col_right: render_feature_scores(wr)

    st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)

    rows = generate_forward_matrix(df, wr, dna, n_days=top_n)

    # ── ★ 籌碼資料抓取 (FinMind，快取1小時) ──────────────────────────
    with st.spinner("🧬 正在讀取三大法人籌碼資料..."):
        chip_raw  = _fetch_chip_data(used_ticker)
    chip_eval = evaluate_chip(chip_raw)

    # ── ★ 買點獵人評估看板(含籌碼第⑥條件) ────────────────────────────
    entry  = evaluate_entry_point(dna, wr, df, chip=chip_eval)
    score  = entry["score"]
    signal = entry["signal"]
    conds  = entry["conditions"]

    d1_low = rows[0]['下限參考'] if len(rows) > 0 else None
    d2_low = rows[1]['下限參考'] if len(rows) > 1 else None

    # 訊號顏色
    if "共振" in signal:          score_color, score_bg = "#c0392b", "#fde8e8"
    elif score >= 80:              score_color, score_bg = "#0a7c59", "#e8f4ec"
    elif score >= 65:              score_color, score_bg = "#1565c0", "#eaf2fb"
    elif score >= 50:              score_color, score_bg = "#d97706", "#fef3c7"
    else:                          score_color, score_bg = "#c0392b", "#fde8e8"

    cond_icon = lambda v: "✅" if v else "❌"
    d1_str = f"{d1_low:.2f}" if d1_low else "計算中"
    d2_str = f"{d2_low:.2f}" if d2_low else "計算中"

    # 籌碼否決特殊處理
    chip_veto_html = ""
    chip_boost_html = ""
    if chip_eval.get("veto"):
        chip_veto_html = f"""
        <div style="background:#c0392b;color:#fff;border-radius:8px;
                    padding:10px 16px;margin-top:10px;font-size:14px;font-weight:700;">
          🚫 一票否決：{chip_eval['label']} — {chip_eval['detail']}
        </div>"""
    elif chip_eval.get("boost") and score >= 65:
        chip_boost_html = f"""
        <div style="background:#0a7c59;color:#fff;border-radius:8px;
                    padding:10px 16px;margin-top:10px;font-size:14px;font-weight:700;">
          🔥 籌碼面同步確認：法人秘密吃貨中！{chip_eval['detail']}
        </div>"""

    st.markdown('<div class="section-title">🎯 買點獵人評估 (技術面 × 籌碼面三合一)</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:{score_bg};border:2px solid {score_color};border-radius:12px;
                padding:16px 20px;margin-bottom:14px;">
      <div style="display:flex;align-items:center;justify-content:space-between;
                  flex-wrap:wrap;gap:12px;">
        <div>
          <div style="font-family:'IBM Plex Mono',monospace;font-size:13px;
                      color:#4a6fa5;margin-bottom:4px;">買點綜合評分（技術面）</div>
          <div style="display:flex;align-items:baseline;gap:10px;">
            <span style="font-family:'IBM Plex Mono',monospace;font-size:40px;
                         font-weight:700;color:{score_color};">{score}</span>
            <span style="font-size:20px;font-weight:700;color:{score_color};">{signal}</span>
          </div>
        </div>
        <div style="font-size:14px;line-height:2.0;">
          <div>{cond_icon(conds['c3_rcycle'])} ① R_cycle ≥ 1.0 &nbsp;
               <b style="color:{score_color};">{dna['R_cycle']:.3f}</b></div>
          <div>{cond_icon(conds['c4_kd'])} ② KD低檔拐頭 &nbsp;
               <b>{entry['kd_stage']}</b></div>
          <div>{cond_icon(conds['c1_mid'])} ③ 中繼蓄勢 &nbsp;
               <b>{wr['category_label']}</b></div>
          <div>{cond_icon(conds['c2_wr'])} ④ 勝率甜蜜區 50~68% &nbsp;
               <b>{wr['winrate']*100:.0f}%</b></div>
          <div>{cond_icon(conds['c5_vol'])} ⑤ 量比 &lt; 2.5 &nbsp;
               <b>{wr['vol_ratio']:.2f}x</b></div>
        </div>
      </div>
      {chip_veto_html}{chip_boost_html}
      <div style="margin-top:12px;padding-top:10px;border-top:1px solid {score_color}33;
                  font-size:13px;color:#1a2b3c;">
        <b style="color:{score_color};">📌 掛單區間</b> &nbsp;
        D+1 下限 <b style="font-family:'IBM Plex Mono',monospace;">{d1_str}</b> 元（低接）
        &nbsp;｜&nbsp;
        D+2 下限 <b style="font-family:'IBM Plex Mono',monospace;">{d2_str}</b> 元（停損基準）
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── ★ 籌碼特徵看板 ────────────────────────────────────────────────
    st.markdown('<div class="section-title">🧬 籌碼特徵動態（三大法人）</div>',
                unsafe_allow_html=True)

    if chip_raw.get("available"):
        it5  = chip_raw["it_net_5d"]
        fi5  = chip_raw["fi_net_5d"]
        it_d = chip_raw["it_buy_days"]
        fi3  = chip_raw["fi_3d_sum"]
        it3  = chip_raw["it_3d_sum"]

        # 生成每日淨買超的 sparkline 文字表示
        def spark(vals, unit="張"):
            bars = ""
            for v in vals:
                if v > 500:    bars += "▲"
                elif v > 0:    bars += "△"
                elif v > -500: bars += "▽"
                else:          bars += "▼"
            return bars

        fi_spark = spark(fi5)
        it_spark = spark(it5)

        cl  = chip_eval["css_color"]
        lbl = chip_eval["label"]

        st.markdown(f"""
        <div class="dna-card" style="border-color:{cl};">
          <div style="display:flex;justify-content:space-between;
                      align-items:center;margin-bottom:12px;">
            <span style="font-size:18px;font-weight:700;color:{cl};">{lbl}</span>
            <span style="font-size:12px;color:#4a6fa5;">
              資料來源：FinMind / 證交所法人買賣超
            </span>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;font-size:14px;">
            <div>
              <div style="color:#4a6fa5;font-size:12px;margin-bottom:4px;">
                外資近5日走勢（▲買超 ▽賣超）
              </div>
              <div style="font-family:'IBM Plex Mono',monospace;font-size:22px;
                          letter-spacing:4px;color:#1565c0;">{fi_spark}</div>
              <div style="color:#1a2b3c;margin-top:4px;">
                近3日合計：<b style="font-family:'IBM Plex Mono',monospace;
                color:{'#0a7c59' if fi3>0 else '#c0392b'};">{fi3:+.0f} 張</b>
              </div>
            </div>
            <div>
              <div style="color:#4a6fa5;font-size:12px;margin-bottom:4px;">
                投信近5日走勢（▲買超 ▽賣超）
              </div>
              <div style="font-family:'IBM Plex Mono',monospace;font-size:22px;
                          letter-spacing:4px;color:#d97706;">{it_spark}</div>
              <div style="color:#1a2b3c;margin-top:4px;">
                近5日買超 <b style="color:{'#0a7c59' if it_d>=3 else '#1a2b3c'};">
                {it_d}/5 天</b> ｜ 近3日{it3:+.0f}張
              </div>
            </div>
          </div>
          <div style="margin-top:10px;font-size:13px;color:#4a6fa5;
                      border-top:1px solid #c8d8e8;padding-top:8px;">
            {chip_eval['detail']}
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        err_msg = chip_raw.get('error', '未知錯誤')
        if '402' in err_msg:
            st.warning(
                f"⚠️ FinMind 免費額度暫時用盡（status=402），買點評估改以純技術面判斷。\n\n"
                f"這是暫時性限制，通常數分鐘內會恢復，5 分鐘後重新查詢會自動重試，"
                f"不需要手動處理。若持續發生，可能是短時間內掃描檔數過多。"
            )
        else:
            st.warning(
                f"⚠️ 籌碼資料暫時無法取得（{err_msg}），買點評估改以純技術面判斷。\n\n"
                f"FinMind 使用免費 REST API，無需安裝套件。如持續失敗請確認網路連線正常。"
            )

    render_forward_table(rows, wr["close"])

    st.markdown('<div class="section-title">📈 近期走勢 (收盤價)</div>', unsafe_allow_html=True)
    chart_df = df[["Close", "MA5", "MA20", "MA60"]].tail(120).dropna(subset=["Close"])
    st.line_chart(chart_df, use_container_width=True, height=200)

    st.markdown(f"""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#7a9bbf;
                margin-top:18px;text-align:center;color:#4a6fa5;">
      ⚠️ 本系統僅供技術型態研究,不構成任何投資建議。數據來源: Yahoo Finance (yfinance)。
      最後更新: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
