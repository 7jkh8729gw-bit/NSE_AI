"""
NSE AI BREAKOUT BOT V2
======================
Complete replacement for the previous Render/Flask-based screener.

What V2 does:
1. Scans the broad NSE equity universe.
2. Applies the user's core trend/liquidity/52-week-high filters.
3. Builds a morning candidate list from daily OHLCV data.
4. Detects bullish daily-chart structures:
   - Golden Cross / bullish EMA alignment
   - Double Bottom
   - Higher High / Higher Low
   - Bullish Engulfing
   - Hammer
   - Morning Star
   - 3 White Soldiers
   - MACD bullish crossover
   - RSI strength
   - OBV accumulation
   - Near-breakout / breakout
5. Searches Google News RSS for recent stock-specific headlines and
   estimates catalyst sentiment using a transparent keyword model.
6. Creates a composite AI-style score (rules + weighted evidence).
7. Before market: sends the ranked morning watchlist to Telegram.
8. During market: continuously checks ONLY the morning candidates for
   intraday volume spikes and breakout conditions and sends BUY alerts.
9. At the same time, the main universe scan continues periodically so
   NEW stocks that meet the criteria can join the live watchlist.
10. Stores state locally to avoid repeated Telegram alerts.
11. NO Flask / Render Web Service is used.

IMPORTANT:
- This is a rule-based quantitative assistant, not a guarantee of profit.
- yfinance is convenient but is not a guaranteed real-time NSE feed.
- For production-grade intraday execution, replace the market-data layer
  with a broker/data-provider API.
- Never blindly execute a signal; verify price, liquidity, spread and SL.
"""

import os
import re
import time
import json
import math
import pickle
import logging
import threading
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, time as dt_time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import telebot
from telebot import types
from datasets import load_dataset

from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator


# ============================================================
# CONFIGURATION
# ============================================================

IST_OFFSET_HOURS = 5
IST_OFFSET_MINUTES = 30

# Core universe filters
MIN_MARKET_CAP_CR = float(os.getenv("MIN_MARKET_CAP_CR", "1000"))
MIN_PRICE = float(os.getenv("MIN_PRICE", "100"))
MAX_DAY_CHANGE = float(os.getenv("MAX_DAY_CHANGE", "15"))
MIN_DAY_VOLUME = int(os.getenv("MIN_DAY_VOLUME", "200000"))
MAX_FROM_52W_HIGH = float(os.getenv("MAX_FROM_52W_HIGH", "10"))
MIN_DAILY_VOLUME_RATIO = float(os.getenv("MIN_DAILY_VOLUME_RATIO", "1.5"))
MIN_AVG_VOLUME = int(os.getenv("MIN_AVG_VOLUME", "500000"))

# Trend filters
REQUIRE_50_ABOVE_200 = True
REQUIRE_10_ABOVE_50 = True

# Morning scan
MORNING_SCAN_HOUR = int(os.getenv("MORNING_SCAN_HOUR", "8"))
MORNING_SCAN_MINUTE = int(os.getenv("MORNING_SCAN_MINUTE", "45"))

# Main universe re-scan during market
UNIVERSE_RESCAN_MINUTES = int(os.getenv("UNIVERSE_RESCAN_MINUTES", "15"))

# Intraday breakout scan
INTRADAY_SCAN_SECONDS = int(os.getenv("INTRADAY_SCAN_SECONDS", "60"))
INTRADAY_INTERVAL = os.getenv("INTRADAY_INTERVAL", "5m")

# Intraday signal thresholds
MIN_INTRADAY_VOLUME_RATIO = float(os.getenv("MIN_INTRADAY_VOLUME_RATIO", "1.8"))
MIN_BREAKOUT_SCORE = float(os.getenv("MIN_BREAKOUT_SCORE", "70"))
MIN_BUY_SCORE = float(os.getenv("MIN_BUY_SCORE", "75"))

# News
NEWS_LOOKBACK_HOURS = int(os.getenv("NEWS_LOOKBACK_HOURS", "30"))
MAX_NEWS_ITEMS = int(os.getenv("MAX_NEWS_ITEMS", "5"))

# Universe data workers
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))

# Persistence
STATE_FILE = "ai_bot_v2_state.pkl"
CACHE_FILE = "ai_bot_v2_daily_cache.pkl"
WATCHLIST_FILE = "ai_bot_v2_watchlist.pkl"

# Optional: limit Telegram morning list to this many stocks
MORNING_TOP_N = int(os.getenv("MORNING_TOP_N", "20"))

# ============================================================
# TELEGRAM
# ============================================================

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

bot = telebot.TeleBot(BOT_TOKEN) if BOT_TOKEN else None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

log = logging.getLogger("NSE-AI-V2")


# ============================================================
# GLOBAL STATE
# ============================================================

STATE = {
    "alerted": set(),
    "morning_sent_date": None,
    "last_universe_scan": None,
    "last_status_date": None,
}

WATCHLIST = {}
UNIVERSE = []
UNIVERSE_LOCK = threading.Lock()


# ============================================================
# TIME HELPERS
# ============================================================

def now_ist():
    # Avoid external timezone dependency.
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


def today_str():
    return now_ist().strftime("%Y-%m-%d")


def market_is_open():
    n = now_ist()
    if n.weekday() >= 5:
        return False

    start = dt_time(9, 15)
    end = dt_time(15, 30)
    return start <= n.time() <= end


def before_market():
    n = now_ist()
    return n.weekday() < 5 and n.time() < dt_time(9, 15)


def after_market():
    n = now_ist()
    return n.weekday() < 5 and n.time() > dt_time(15, 30)


# ============================================================
# PERSISTENCE
# ============================================================

def load_pickle(path, default):
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        log.warning("Could not load %s: %s", path, e)
    return default


def save_pickle(path, obj):
    try:
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    except Exception as e:
        log.warning("Could not save %s: %s", path, e)


def load_state():
    global STATE, WATCHLIST
    old_state = load_pickle(STATE_FILE, {})
    if isinstance(old_state, dict):
        STATE.update(old_state)

    if not isinstance(STATE.get("alerted"), set):
        STATE["alerted"] = set()

    WATCHLIST = load_pickle(WATCHLIST_FILE, {})
    if not isinstance(WATCHLIST, dict):
        WATCHLIST = {}


def save_state():
    save_pickle(STATE_FILE, STATE)
    save_pickle(WATCHLIST_FILE, WATCHLIST)


# ============================================================
# TELEGRAM HELPERS
# ============================================================

def tg_send(text, parse_mode="Markdown"):
    if not bot or not CHAT_ID:
        log.warning("Telegram is not configured.")
        return None

    try:
        return bot.send_message(
            CHAT_ID,
            text,
            parse_mode=parse_mode,
            disable_web_page_preview=True
        )
    except Exception as e:
        log.warning("Telegram send failed: %s", e)
        return None


def tg_long_send(text):
    # Telegram message limit is ~4096 chars.
    chunks = []
    while text:
        chunks.append(text[:3900])
        text = text[3900:]

    for chunk in chunks:
        tg_send(chunk)


# ============================================================
# NSE UNIVERSE
# ============================================================

def get_all_nse_stocks():
    """
    Uses the same Hugging Face security master approach as V1.
    Removes obvious non-equity / malformed symbols.
    """
    log.info("Loading NSE universe...")

    try:
        ds = load_dataset(
            "tickertruth/nse-india-security-master",
            data_files="data/nse_security_master.csv"
        )
        df = ds["train"].to_pandas()

        df = df[df["active_flag"] == True]

        symbols = (
            df["nse_symbol"]
            .astype(str)
            .str.strip()
            .str.upper()
            .tolist()
        )

        symbols = sorted(set(
            s for s in symbols
            if re.fullmatch(r"[A-Z0-9&._-]+", s)
        ))

        log.info("NSE universe loaded: %s symbols", len(symbols))
        return symbols

    except Exception as e:
        log.exception("Universe loading failed: %s", e)
        return [
            "RELIANCE",
            "TCS",
            "HDFCBANK",
            "INFY",
            "ICICIBANK",
        ]


# ============================================================
# YFINANCE DATA
# ============================================================

def yf_daily(symbol, period="1y"):
    try:
        df = yf.download(
            f"{symbol}.NS",
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False
        )

        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = ["Open", "High", "Low", "Close", "Volume"]
        if not all(c in df.columns for c in required):
            return None

        df = df[required].copy()
        df = df.dropna()
        return df

    except Exception as e:
        log.debug("Daily data error %s: %s", symbol, e)
        return None


def yf_intraday(symbol):
    try:
        df = yf.download(
            f"{symbol}.NS",
            period="2d",
            interval=INTRADAY_INTERVAL,
            auto_adjust=False,
            progress=False,
            threads=False
        )

        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = ["Open", "High", "Low", "Close", "Volume"]
        if not all(c in df.columns for c in required):
            return None

        df = df[required].dropna()
        return df

    except Exception as e:
        log.debug("Intraday data error %s: %s", symbol, e)
        return None


def get_info(symbol):
    """
    yfinance info is relatively expensive and can fail.
    It is used only when needed.
    """
    try:
        t = yf.Ticker(f"{symbol}.NS")
        info = t.info or {}

        return {
            "price": float(
                info.get("regularMarketPrice")
                or info.get("currentPrice")
                or 0
            ),
            "prev_close": float(
                info.get("regularMarketPreviousClose")
                or info.get("previousClose")
                or 0
            ),
            "volume": int(info.get("regularMarketVolume") or 0),
            "high_52w": float(info.get("fiftyTwoWeekHigh") or 0),
            "market_cap": float(info.get("marketCap") or 0) / 1e7,
        }

    except Exception:
        return {
            "price": 0,
            "prev_close": 0,
            "volume": 0,
            "high_52w": 0,
            "market_cap": 0,
        }


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def dema(series, period):
    ema1 = series.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    return (2 * ema1) - ema2


def add_indicators(df):
    x = df.copy()

    x["EMA10"] = EMAIndicator(x["Close"], window=10).ema_indicator()
    x["EMA20"] = EMAIndicator(x["Close"], window=20).ema_indicator()
    x["EMA50"] = EMAIndicator(x["Close"], window=50).ema_indicator()
    x["EMA200"] = EMAIndicator(x["Close"], window=200).ema_indicator()

    x["DEMA10"] = dema(x["Close"], 10)
    x["DEMA50"] = dema(x["Close"], 50)
    x["DEMA200"] = dema(x["Close"], 200)

    x["RSI"] = RSIIndicator(x["Close"], window=14).rsi()

    macd = MACD(
        x["Close"],
        window_slow=26,
        window_fast=12,
        window_sign=9
    )
    x["MACD"] = macd.macd()
    x["MACDSignal"] = macd.macd_signal()
    x["MACDHist"] = macd.macd_diff()

    x["ATR"] = AverageTrueRange(
        x["High"],
        x["Low"],
        x["Close"],
        window=14
    ).average_true_range()

    x["OBV"] = OnBalanceVolumeIndicator(
        x["Close"],
        x["Volume"]
    ).on_balance_volume()

    x["ADX"] = ADXIndicator(
        x["High"],
        x["Low"],
        x["Close"],
        window=14
    ).adx()

    x["AvgVol20"] = x["Volume"].rolling(20).mean()
    x["AvgVol50"] = x["Volume"].rolling(50).mean()

    return x


# ============================================================
# CANDLE PATTERNS
# ============================================================

def candle_body(row):
    return abs(row["Close"] - row["Open"])


def detect_candlestick_patterns(df):
    patterns = []

    if df is None or len(df) < 5:
        return patterns

    x = df.iloc[-5:].copy()

    last = x.iloc[-1]
    prev = x.iloc[-2]
    p2 = x.iloc[-3]

    body = candle_body(last)
    prev_body = candle_body(prev)

    # Bullish engulfing
    if (
        prev["Close"] < prev["Open"]
        and last["Close"] > last["Open"]
        and last["Open"] <= prev["Close"]
        and last["Close"] >= prev["Open"]
    ):
        patterns.append("Bullish Engulfing")

    # Hammer
    if body > 0:
        lower = min(last["Open"], last["Close"]) - last["Low"]
        upper = last["High"] - max(last["Open"], last["Close"])

        if lower >= 2 * body and upper <= body * 0.5:
            patterns.append("Hammer")

    # Morning star
    if (
        p2["Close"] < p2["Open"]
        and abs(prev["Close"] - prev["Open"]) <= abs(p2["Close"] - p2["Open"]) * 0.5
        and last["Close"] > last["Open"]
        and last["Close"] > (p2["Open"] + p2["Close"]) / 2
    ):
        patterns.append("Morning Star")

    # 3 white soldiers
    if len(x) >= 3:
        a, b, c = x.iloc[-3], x.iloc[-2], x.iloc[-1]
        if (
            a["Close"] > a["Open"]
            and b["Close"] > b["Open"]
            and c["Close"] > c["Open"]
            and b["Close"] > a["Close"]
            and c["Close"] > b["Close"]
        ):
            patterns.append("3 White Soldiers")

    return patterns


# ============================================================
# STRUCTURAL PATTERNS
# ============================================================

def detect_golden_cross(df):
    if len(df) < 210:
        return False

    ema50 = df["EMA50"]
    ema200 = df["EMA200"]

    recent = ema50.iloc[-10:] > ema200.iloc[-10:]

    # Current bullish alignment
    alignment = ema50.iloc[-1] > ema200.iloc[-1]

    # Actual recent crossover
    crossover = False
    for i in range(1, min(15, len(df))):
        a = ema50.iloc[-i-1] - ema200.iloc[-i-1]
        b = ema50.iloc[-i] - ema200.iloc[-i]
        if a <= 0 and b > 0:
            crossover = True
            break

    return bool(alignment and (crossover or recent.sum() >= 7))


def detect_double_bottom(df):
    """
    Practical approximate double-bottom detector.
    Looks for two swing lows within a tolerance and a neckline.
    """
    if len(df) < 60:
        return False

    close = df["Close"].values
    lows = df["Low"].values

    lookback = min(120, len(df))
    start = len(df) - lookback

    candidates = []

    for i in range(start + 3, len(df) - 3):
        if (
            lows[i] <= lows[i-1]
            and lows[i] <= lows[i-2]
            and lows[i] <= lows[i+1]
            and lows[i] <= lows[i+2]
        ):
            candidates.append(i)

    if len(candidates) < 2:
        return False

    for a_idx in candidates[:-1]:
        for b_idx in candidates:
            if b_idx <= a_idx:
                continue

            distance = b_idx - a_idx

            if distance < 10 or distance > 80:
                continue

            a = lows[a_idx]
            b = lows[b_idx]

            avg_low = (a + b) / 2
            if avg_low <= 0:
                continue

            # Bottoms should be reasonably similar.
            if abs(a - b) / avg_low > 0.04:
                continue

            neckline = max(high for high in df["High"].iloc[a_idx:b_idx+1])

            recent_close = close[-1]

            # Price should be near/above neckline for a valid setup.
            if recent_close >= neckline * 0.97:
                return True

    return False


def detect_higher_high_higher_low(df):
    if len(df) < 30:
        return False

    x = df.iloc[-30:]

    recent_high = x["High"].iloc[-1]
    previous_high = x["High"].iloc[-15:-3].max()

    recent_low = x["Low"].iloc[-1]
    previous_low = x["Low"].iloc[-15:-3].min()

    return (
        recent_high >= previous_high
        and recent_low >= previous_low * 0.97
    )


def detect_near_breakout(df):
    if len(df) < 30:
        return False

    resistance = df["High"].iloc[-21:-1].max()
    close = df["Close"].iloc[-1]

    return close >= resistance * 0.985


def detect_breakout(df):
    if len(df) < 25:
        return False

    resistance = df["High"].iloc[-21:-1].max()
    close = df["Close"].iloc[-1]
    volume = df["Volume"].iloc[-1]
    avg = df["Volume"].iloc[-21:-1].mean()

    return (
        close > resistance
        and avg > 0
        and volume >= avg * 1.5
    )


# ============================================================
# NEWS / CATALYST ENGINE
# ============================================================

BULLISH_WORDS = {
    "order": 3,
    "contract": 3,
    "wins": 2,
    "win": 2,
    "approval": 3,
    "approved": 3,
    "launch": 2,
    "expansion": 2,
    "acquisition": 2,
    "merger": 2,
    "earnings": 1,
    "profit": 3,
    "profits": 3,
    "revenue": 2,
    "growth": 2,
    "upgrade": 3,
    "buy": 2,
    "target": 1,
    "capacity": 2,
    "investment": 2,
    "partnership": 2,
    "export": 2,
    "record": 2,
    "strong": 1,
    "positive": 2,
    "surge": 2,
    "rises": 1,
}

BEARISH_WORDS = {
    "fraud": 6,
    "default": 5,
    "downgrade": 4,
    "loss": 3,
    "losses": 3,
    "decline": 2,
    "falls": 2,
    "fall": 2,
    "probe": 4,
    "investigation": 4,
    "resign": 3,
    "resignation": 3,
    "warning": 2,
    "debt": 2,
    "lawsuit": 3,
    "penalty": 3,
    "cut": 2,
    "weak": 2,
    "negative": 2,
}


def clean_text(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


def news_sentiment(text):
    text = text.lower()

    bull = 0
    bear = 0

    for word, weight in BULLISH_WORDS.items():
        if re.search(r"\b" + re.escape(word) + r"\b", text):
            bull += weight

    for word, weight in BEARISH_WORDS.items():
        if re.search(r"\b" + re.escape(word) + r"\b", text):
            bear += weight

    raw = bull - bear

    if raw >= 5:
        label = "Bullish"
    elif raw <= -4:
        label = "Bearish"
    else:
        label = "Neutral"

    return raw, label


def fetch_google_news(symbol):
    """
    Google News RSS. No API key required.
    """
    try:
        q = urllib.parse.quote(f"{symbol} NSE India stock")
        url = (
            "https://news.google.com/rss/search?"
            f"q={q}&hl=en-IN&gl=IN&ceid=IN:en"
        )

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read()

        root = ET.fromstring(xml_data)

        items = []

        for item in root.findall(".//item")[:MAX_NEWS_ITEMS]:
            title = clean_text(item.findtext("title"))
            link = clean_text(item.findtext("link"))
            pub = clean_text(item.findtext("pubDate"))

            if title:
                score, label = news_sentiment(title)

                items.append({
                    "title": title,
                    "link": link,
                    "published": pub,
                    "score": score,
                    "label": label,
                })

        return items

    except Exception as e:
        log.debug("News error %s: %s", symbol, e)
        return []


def compute_news_score(symbol):
    items = fetch_google_news(symbol)

    if not items:
        return {
            "score": 0,
            "label": "No recent news",
            "headlines": []
        }

    total = 0
    bullish = 0
    bearish = 0

    for item in items:
        total += item["score"]
        if item["label"] == "Bullish":
            bullish += 1
        elif item["label"] == "Bearish":
            bearish += 1

    # Convert roughly to 0-100.
    score = 50 + total * 5
    score += min(15, bullish * 5)
    score -= min(20, bearish * 7)

    score = max(0, min(100, score))

    if score >= 65:
        label = "Bullish"
    elif score <= 35:
        label = "Bearish"
    else:
        label = "Neutral"

    return {
        "score": round(score, 1),
        "label": label,
        "headlines": items,
    }


# ============================================================
# DAILY AI-STYLE ANALYSIS
# ============================================================

def analyze_daily(symbol, df):
    if df is None or len(df) < 210:
        return None

    x = add_indicators(df)

    last = x.iloc[-1]
    prev = x.iloc[-2]

    close = float(last["Close"])
    atr = float(last["ATR"]) if not pd.isna(last["ATR"]) else 0

    if close <= 0:
        return None

    patterns = detect_candlestick_patterns(x)
    golden_cross = detect_golden_cross(x)
    double_bottom = detect_double_bottom(x)
    hh_hl = detect_higher_high_higher_low(x)
    near_breakout = detect_near_breakout(x)
    breakout = detect_breakout(x)

    macd_cross = (
        last["MACD"] > last["MACDSignal"]
        and prev["MACD"] <= prev["MACDSignal"]
    )

    rsi = float(last["RSI"]) if not pd.isna(last["RSI"]) else 50
    adx = float(last["ADX"]) if not pd.isna(last["ADX"]) else 0

    obv_accumulation = (
        len(x) >= 10
        and x["OBV"].iloc[-1] > x["OBV"].iloc[-6]
    )

    ema_alignment = (
        last["EMA10"] > last["EMA20"]
        > last["EMA50"]
        > last["EMA200"]
    )

    dema_alignment = (
        last["DEMA10"] > last["DEMA50"] > last["DEMA200"]
    )

    volume_ratio = (
        float(last["Volume"]) / float(last["AvgVol20"])
        if last["AvgVol20"] and not pd.isna(last["AvgVol20"])
        else 0
    )

    score = 0
    reasons = []

    # Trend: 30 points
    if ema_alignment:
        score += 8
        reasons.append("EMA bullish alignment")

    if dema_alignment:
        score += 7
        reasons.append("DEMA bullish alignment")

    if golden_cross:
        score += 8
        reasons.append("Golden Cross")

    if hh_hl:
        score += 7
        reasons.append("Higher High / Higher Low")

    # Structure / patterns: 25 points
    if double_bottom:
        score += 10
        reasons.append("Double Bottom")

    if near_breakout:
        score += 6
        reasons.append("Near Breakout")

    if breakout:
        score += 9
        reasons.append("Confirmed Breakout")

    # Candles: up to 15
    if patterns:
        score += min(15, len(patterns) * 5)
        reasons.extend(patterns)

    # Momentum: 15
    if 50 <= rsi <= 68:
        score += 6
        reasons.append("Healthy RSI")

    if macd_cross:
        score += 6
        reasons.append("MACD Bullish Crossover")
    elif last["MACD"] > last["MACDSignal"]:
        score += 3
        reasons.append("MACD Bullish")

    if adx >= 20:
        score += 3
        reasons.append("ADX Trend Strength")

    # Volume / accumulation: 15
    if volume_ratio >= 1.5:
        score += 5
        reasons.append("Volume Expansion")

    if obv_accumulation:
        score += 5
        reasons.append("OBV Accumulation")

    if volume_ratio >= 2:
        score += 5
        reasons.append("Strong Volume")

    score = min(100, score)

    if breakout:
        setup = "BREAKOUT"
    elif double_bottom:
        setup = "DOUBLE BOTTOM"
    elif golden_cross:
        setup = "GOLDEN CROSS"
    elif near_breakout:
        setup = "NEAR BREAKOUT"
    elif hh_hl:
        setup = "UPTREND"
    else:
        setup = "BULLISH"

    stop_loss = close - (1.5 * atr) if atr > 0 else close * 0.97
    target1 = close + (2.0 * atr) if atr > 0 else close * 1.04
    target2 = close + (3.5 * atr) if atr > 0 else close * 1.07

    return {
        "symbol": symbol,
        "close": close,
        "rsi": round(rsi, 2),
        "adx": round(adx, 2),
        "atr": round(atr, 2),
        "volume_ratio": round(volume_ratio, 2),
        "ema_alignment": bool(ema_alignment),
        "dema_alignment": bool(dema_alignment),
        "golden_cross": bool(golden_cross),
        "double_bottom": bool(double_bottom),
        "hh_hl": bool(hh_hl),
        "near_breakout": bool(near_breakout),
        "breakout": bool(breakout),
        "macd_bullish": bool(last["MACD"] > last["MACDSignal"]),
        "macd_cross": bool(macd_cross),
        "obv_accumulation": bool(obv_accumulation),
        "patterns": patterns,
        "setup": setup,
        "score": round(score, 1),
        "reasons": reasons,
        "stop_loss": round(stop_loss, 2),
        "target1": round(target1, 2),
        "target2": round(target2, 2),
    }


# ============================================================
# CORE FILTER
# ============================================================

def apply_core_filters(symbol, df, info=None):
    if df is None or len(df) < 210:
        return None

    x = add_indicators(df)
    last = x.iloc[-1]

    price = float(last["Close"])
    volume = float(last["Volume"])

    if info is None:
        info = get_info(symbol)

    prev_close = info["prev_close"] or (
        float(x["Close"].iloc[-2]) if len(x) >= 2 else price
    )

    high_52w = info["high_52w"]
    if high_52w <= 0:
        high_52w = float(x["High"].tail(252).max())

    market_cap = info["market_cap"]

    avg_volume = float(
        x["Volume"].tail(21).mean()
    )

    day_change = (
        ((price - prev_close) / prev_close) * 100
        if prev_close > 0 else 0
    )

    volume_ratio = (
        volume / avg_volume
        if avg_volume > 0 else 0
    )

    pct_from_high = (
        ((high_52w - price) / high_52w) * 100
        if high_52w > 0 else 100
    )

    # Required user filters
    if market_cap < MIN_MARKET_CAP_CR:
        return None

    if price < MIN_PRICE:
        return None

    if day_change < 0:
        return None

    if day_change >= MAX_DAY_CHANGE:
        return None

    if volume < MIN_DAY_VOLUME:
        return None

    if avg_volume <= MIN_AVG_VOLUME:
        return None

    if pct_from_high > MAX_FROM_52W_HIGH:
        return None

    if volume_ratio < MIN_DAILY_VOLUME_RATIO:
        return None

    if REQUIRE_50_ABOVE_200:
        if float(last["DEMA50"]) <= float(last["DEMA200"]):
            return None

    if REQUIRE_10_ABOVE_50:
        if float(last["DEMA10"]) <= float(last["DEMA50"]):
            return None

    return {
        "symbol": symbol,
        "price": round(price, 2),
        "day_change": round(day_change, 2),
        "volume": int(volume),
        "avg_volume": int(avg_volume),
        "volume_ratio": round(volume_ratio, 2),
        "high_52w": round(high_52w, 2),
        "pct_from_high": round(pct_from_high, 2),
        "market_cap": round(market_cap, 2),
    }


# ============================================================
# CANDIDATE ANALYSIS
# ============================================================

def analyze_candidate(symbol):
    try:
        df = yf_daily(symbol, "1y")
        if df is None:
            return None

        base = apply_core_filters(symbol, df)

        if base is None:
            return None

        technical = analyze_daily(symbol, df)

        if technical is None:
            return None

        news = compute_news_score(symbol)

        # Final V2 score:
        # Technical/chart = 55%
        # News catalyst = 20%
        # Core market/volume = 25%
        core_score = 0

        if base["volume_ratio"] >= 3:
            core_score += 10
        elif base["volume_ratio"] >= 2:
            core_score += 7
        else:
            core_score += 4

        if base["pct_from_high"] <= 3:
            core_score += 8
        elif base["pct_from_high"] <= 7:
            core_score += 6
        else:
            core_score += 3

        if 0 <= base["day_change"] <= 5:
            core_score += 7
        else:
            core_score += 4

        combined = (
            technical["score"] * 0.55
            + news["score"] * 0.20
            + core_score * (25 / 25)
        )

        # Normalize because core_score is 0-25.
        combined = min(100, combined)

        return {
            "symbol": symbol,
            "base": base,
            "technical": technical,
            "news": news,
            "core_score": round(core_score, 1),
            "combined_score": round(combined, 1),
            "added_at": now_ist().isoformat(),
        }

    except Exception as e:
        log.debug("Candidate error %s: %s", symbol, e)
        return None


# ============================================================
# PARALLEL UNIVERSE SCAN
# ============================================================

def scan_universe(symbols=None):
    if symbols is None:
        symbols = UNIVERSE

    log.info("Starting main universe scan: %s stocks", len(symbols))

    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(analyze_candidate, s): s
            for s in symbols
        }

        completed = 0

        for future in as_completed(futures):
            completed += 1

            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception:
                pass

            if completed % 100 == 0:
                log.info(
                    "Universe progress: %s/%s | candidates=%s",
                    completed,
                    len(symbols),
                    len(results)
                )

    results.sort(
        key=lambda x: x["combined_score"],
        reverse=True
    )

    log.info(
        "Universe scan complete: %s candidates",
        len(results)
    )

    return results


# ============================================================
# WATCHLIST MANAGEMENT
# ============================================================

def merge_watchlist(results):
    added = []

    with UNIVERSE_LOCK:
        for item in results:
            symbol = item["symbol"]

            if symbol not in WATCHLIST:
                WATCHLIST[symbol] = item
                added.append(item)
            else:
                # Update latest analysis while preserving signal state.
                old = WATCHLIST[symbol]

                signal_state = old.get(
                    "signal_state",
                    "WATCHING"
                )

                item["signal_state"] = signal_state
                WATCHLIST[symbol] = item

    save_state()
    return added


# ============================================================
# MORNING REPORT
# ============================================================

def format_morning_report(results):
    date = today_str()

    top = results[:MORNING_TOP_N]

    msg = (
        f"ð *NSE AI BREAKOUT V2*\n"
        f"ð {date}\n"
        f"ð Candidates passing core filters: *{len(results)}*\n\n"
    )

    for i, item in enumerate(top, 1):
        t = item["technical"]
        n = item["news"]
        b = item["base"]

        patterns = ", ".join(t["patterns"]) if t["patterns"] else "None"

        msg += (
            f"*{i}. {item['symbol']}* â "
            f"AI Score *{item['combined_score']}/100*\n"
            f"ð° â¹{b['price']:.2f} | "
            f"ð {b['day_change']:.2f}% | "
            f"ð Vol {b['volume_ratio']:.2f}x\n"
            f"ð Setup: *{t['setup']}*\n"
            f"ð§  Patterns: {patterns}\n"
            f"ð RSI {t['rsi']} | ADX {t['adx']} | "
            f"MACD {'ð¢' if t['macd_bullish'] else 'ð´'}\n"
            f"ð° News: {n['label']} ({n['score']}/100)\n"
            f"ð¯ SL â¹{t['stop_loss']:.2f} | "
            f"T1 â¹{t['target1']:.2f} | "
            f"T2 â¹{t['target2']:.2f}\n"
            f"ââââââââââââââââ\n"
        )

    msg += (
        "\nâ¡ *Live mode:* I will now watch these candidates for "
        "intraday volume expansion + breakout confirmation.\n"
        "ð The main universe scanner will also continue searching "
        "for NEW qualifying stocks."
    )

    return msg


# ============================================================
# INTRADAY BREAKOUT ENGINE
# ============================================================

def analyze_intraday(symbol, morning_item):
    df = yf_intraday(symbol)

    if df is None or len(df) < 10:
        return None

    # Restrict to today's bars when possible.
    today = now_ist().date()

    try:
        dates = pd.to_datetime(df.index)
        today_mask = dates.date == today
        today_df = df.loc[today_mask]

        if len(today_df) >= 3:
            day = today_df.copy()
        else:
            day = df.tail(30).copy()
    except Exception:
        day = df.tail(30).copy()

    if len(day) < 3:
        return None

    last = day.iloc[-1]

    price = float(last["Close"])
    volume = float(last["Volume"])

    # Average of prior same-day bars.
    prior = day.iloc[:-1]

    avg_bar_volume = (
        float(prior["Volume"].tail(12).mean())
        if len(prior) else 0
    )

    volume_ratio = (
        volume / avg_bar_volume
        if avg_bar_volume > 0 else 0
    )

    # Breakout level from today's recent bars and morning range.
    lookback = min(12, len(day) - 1)
    resistance = float(
        day["High"].iloc[-lookback-1:-1].max()
    ) if lookback >= 2 else float(day["High"].iloc[:-1].max())

    # Also compare with previous daily close / morning candidate.
    morning_price = morning_item["base"]["price"]

    bullish_candle = last["Close"] > last["Open"]

    range_size = max(
        float(last["High"] - last["Low"]),
        0.01
    )

    body = abs(float(last["Close"] - last["Open"]))
    strong_body = body / range_size >= 0.55

    breakout = price > resistance

    # Score
    score = 0

    if breakout:
        score += 35

    if volume_ratio >= 3:
        score += 30
    elif volume_ratio >= 2.5:
        score += 25
    elif volume_ratio >= MIN_INTRADAY_VOLUME_RATIO:
        score += 18

    if bullish_candle:
        score += 10

    if strong_body:
        score += 10

    if price > morning_price:
        score += 5

    # Daily confirmation
    daily_score = morning_item["technical"]["score"]
    score += min(10, daily_score / 10)

    score = round(min(100, score), 1)

    # Signal only after BOTH breakout and volume confirmation.
    buy_signal = (
        breakout
        and volume_ratio >= MIN_INTRADAY_VOLUME_RATIO
        and bullish_candle
        and score >= MIN_BUY_SCORE
    )

    atr = morning_item["technical"]["atr"]

    sl = (
        price - 1.2 * atr
        if atr > 0 else price * 0.985
    )

    target1 = (
        price + 1.5 * atr
        if atr > 0 else price * 1.025
    )

    target2 = (
        price + 2.5 * atr
        if atr > 0 else price * 1.045
    )

    return {
        "symbol": symbol,
        "price": round(price, 2),
        "resistance": round(resistance, 2),
        "volume_ratio": round(volume_ratio, 2),
        "score": score,
        "breakout": breakout,
        "bullish_candle": bullish_candle,
        "strong_body": strong_body,
        "buy_signal": buy_signal,
        "sl": round(sl, 2),
        "target1": round(target1, 2),
        "target2": round(target2, 2),
        "time": now_ist().strftime("%H:%M:%S"),
    }


# ============================================================
# LIVE BUY ALERT
# ============================================================

def alert_key(symbol):
    return f"{today_str()}::{symbol}::BUY"


def send_buy_alert(signal, morning_item):
    symbol = signal["symbol"]
    key = alert_key(symbol)

    if key in STATE["alerted"]:
        return

    STATE["alerted"].add(key)

    t = morning_item["technical"]
    n = morning_item["news"]

    msg = (
        f"ð¨ *AI BUY SIGNAL â V2*\n"
        f"ââââââââââââââââââââ\n"
        f"ð *{symbol}*\n"
        f"ð° Price: *â¹{signal['price']:.2f}*\n"
        f"ð Breakout: *â¹{signal['resistance']:.2f}*\n"
        f"ð Intraday Volume: *{signal['volume_ratio']:.2f}x*\n"
        f"ð§  Breakout Score: *{signal['score']}/100*\n\n"
        f"ð Daily Setup: *{t['setup']}*\n"
        f"ð Daily AI Score: *{morning_item['combined_score']}/100*\n"
        f"ð RSI: {t['rsi']} | ADX: {t['adx']}\n"
        f"ð° Catalyst: *{n['label']}*\n\n"
        f"ð¯ *Entry:* â¹{signal['price']:.2f}\n"
        f"ð *SL:* â¹{signal['sl']:.2f}\n"
        f"ð *T1:* â¹{signal['target1']:.2f}\n"
        f"ð *T2:* â¹{signal['target2']:.2f}\n\n"
        f"â ï¸ Confirm spread/liquidity before entering.\n"
        f"â° {signal['time']}"
    )

    tg_send(msg)
    save_state()


# ============================================================
# CONTINUOUS INTRADAY WATCHER
# ============================================================

def intraday_watcher():
    log.info("Intraday watcher started.")

    while True:
        try:
            if market_is_open():
                with UNIVERSE_LOCK:
                    items = list(WATCHLIST.values())

                if items:
                    log.info(
                        "Intraday scan: watching %s stocks",
                        len(items)
                    )

                for item in items:
                    try:
                        signal = analyze_intraday(
                            item["symbol"],
                            item
                        )

                        if signal and signal["buy_signal"]:
                            send_buy_alert(signal, item)

                    except Exception as e:
                        log.debug(
                            "Intraday error %s: %s",
                            item.get("symbol"),
                            e
                        )

                    time.sleep(0.15)

            time.sleep(INTRADAY_SCAN_SECONDS)

        except Exception as e:
            log.exception("Intraday watcher failure: %s", e)
            time.sleep(10)


# ============================================================
# MAIN UNIVERSE CONTINUOUS SCANNER
# ============================================================

def universe_monitor():
    log.info("Continuous universe monitor started.")

    while True:
        try:
            if market_is_open():
                n = now_ist()

                last = STATE.get("last_universe_scan")

                due = False

                if last is None:
                    due = True
                else:
                    try:
                        previous = datetime.fromisoformat(last)
                        due = (
                            n - previous
                            >= timedelta(
                                minutes=UNIVERSE_RESCAN_MINUTES
                            )
                        )
                    except Exception:
                        due = True

                if due:
                    log.info("Running new-stock universe scan...")

                    results = scan_universe()

                    new_items = merge_watchlist(results)

                    STATE["last_universe_scan"] = n.isoformat()
                    save_state()

                    if new_items:
                        # Only alert Telegram when genuinely NEW candidates
                        # appear after the morning scan.
                        top_new = sorted(
                            new_items,
                            key=lambda x: x["combined_score"],
                            reverse=True
                        )[:10]

                        msg = (
                            f"ð *NEW AI CANDIDATES â V2*\n"
                            f"â° {n.strftime('%H:%M:%S')}\n"
                            f"ð {len(new_items)} new stocks passed filters.\n\n"
                        )

                        for i, item in enumerate(top_new, 1):
                            t = item["technical"]
                            msg += (
                                f"{i}. *{item['symbol']}* "
                                f"Score {item['combined_score']}\n"
                                f"   Setup: {t['setup']} | "
                                f"Vol {item['base']['volume_ratio']}x\n"
                            )

                        tg_send(msg)

            time.sleep(30)

        except Exception as e:
            log.exception("Universe monitor failure: %s", e)
            time.sleep(30)


# ============================================================
# MORNING SCAN
# ============================================================

def run_morning_scan():
    if now_ist().weekday() >= 5:
        return

    log.info("Starting V2 morning scan.")

    tg_send(
        "ð *NSE AI V2 Morning Scan Started*\n"
        "Scanning the NSE universe for trend + chart patterns + catalysts..."
    )

    results = scan_universe()

    with UNIVERSE_LOCK:
        WATCHLIST.clear()
        for item in results:
            WATCHLIST[item["symbol"]] = item

    STATE["morning_sent_date"] = today_str()
    STATE["last_universe_scan"] = now_ist().isoformat()

    save_state()

    if results:
        tg_long_send(format_morning_report(results))
    else:
        tg_send(
            "ð *NSE AI V2*\n"
            "No stock passed all core filters today."
        )


# ============================================================
# DAILY SCHEDULER
# ============================================================

def scheduler_loop():
    log.info(
        "Scheduler active. Morning scan %02d:%02d IST.",
        MORNING_SCAN_HOUR,
        MORNING_SCAN_MINUTE
    )

    morning_done = False
    current_date = None

    while True:
        try:
            n = now_ist()

            if n.date() != current_date:
                current_date = n.date()
                morning_done = False

                # Do not reuse yesterday's alerted symbols.
                STATE["alerted"] = {
                    x for x in STATE.get("alerted", set())
                    if x.startswith(today_str() + "::")
                }

            scheduled_time = dt_time(
                MORNING_SCAN_HOUR,
                MORNING_SCAN_MINUTE
            )

            if (
                n.weekday() < 5
                and n.time() >= scheduled_time
                and not morning_done
            ):
                # Morning scan can run before market, or if the process
                # starts late it runs immediately after startup.
                run_morning_scan()
                morning_done = True

            time.sleep(20)

        except Exception as e:
            log.exception("Scheduler failure: %s", e)
            time.sleep(30)


# ============================================================
# TELEGRAM COMMANDS
# ============================================================

def register_commands():
    if not bot:
        return

    @bot.message_handler(commands=["start"])
    def start_cmd(message):
        bot.reply_to(
            message,
            "ð¤ NSE AI Breakout Bot V2 is running.\n\n"
            "/scan - run a fresh universe scan\n"
            "/watchlist - show current watchlist\n"
            "/status - show bot status"
        )

    @bot.message_handler(commands=["scan"])
    def scan_cmd(message):
        bot.reply_to(
            message,
            "ð Manual V2 scan started..."
        )

        def worker():
            results = scan_universe()
            merge_watchlist(results)

            if results:
                tg_long_send(format_morning_report(results))
            else:
                tg_send("ð No qualifying stocks found.")

        threading.Thread(target=worker, daemon=True).start()

    @bot.message_handler(commands=["watchlist"])
    def watchlist_cmd(message):
        with UNIVERSE_LOCK:
            items = list(WATCHLIST.values())

        items.sort(
            key=lambda x: x.get("combined_score", 0),
            reverse=True
        )

        if not items:
            bot.reply_to(message, "Watchlist is empty.")
            return

        msg = "ð *CURRENT V2 WATCHLIST*\n\n"

        for i, item in enumerate(items[:30], 1):
            t = item["technical"]
            msg += (
                f"{i}. *{item['symbol']}* "
                f"{item['combined_score']}/100 â "
                f"{t['setup']}\n"
            )

        bot.reply_to(
            message,
            msg,
            parse_mode="Markdown"
        )

    @bot.message_handler(commands=["status"])
    def status_cmd(message):
        with UNIVERSE_LOCK:
            count = len(WATCHLIST)

        bot.reply_to(
            message,
            f"ð¤ *NSE AI V2 STATUS*\n\n"
            f"Universe: {len(UNIVERSE)}\n"
            f"Watchlist: {count}\n"
            f"Market open: {market_is_open()}\n"
            f"Last universe scan: "
            f"{STATE.get('last_universe_scan', 'Never')}\n"
            f"Morning scan: "
            f"{STATE.get('morning_sent_date', 'Not sent')}",
            parse_mode="Markdown"
        )


# ============================================================
# STARTUP
# ============================================================

def validate_environment():
    if not BOT_TOKEN:
        log.warning(
            "BOT_TOKEN is not set. Telegram alerts are disabled."
        )

    if not CHAT_ID:
        log.warning(
            "CHAT_ID is not set. Telegram alerts are disabled."
        )


def build_universe():
    global UNIVERSE

    UNIVERSE = get_all_nse_stocks()

    if not UNIVERSE:
        raise RuntimeError("NSE universe is empty.")


def main():
    print("=" * 72)
    print("ð¤ NSE AI BREAKOUT BOT V2")
    print("=" * 72)
    print("Render/Flask removed.")
    print("Morning chart + catalyst analysis enabled.")
    print("Continuous intraday breakout monitoring enabled.")
    print("Continuous new-stock universe scan enabled.")
    print("=" * 72)

    validate_environment()
    load_state()
    build_universe()
    register_commands()

    if bot:
        try:
            me = bot.get_me()
            log.info("Telegram connected: @%s", me.username)
        except Exception as e:
            log.warning("Telegram connection test failed: %s", e)

    # --------------------------------------------------------
    # Startup behaviour
    # --------------------------------------------------------
    n = now_ist()

    # If the process starts after morning-scan time but before
    # market close, run today's scan immediately.
    if (
        n.weekday() < 5
        and n.time() >= dt_time(
            MORNING_SCAN_HOUR,
            MORNING_SCAN_MINUTE
        )
        and STATE.get("morning_sent_date") != today_str()
    ):
        threading.Thread(
            target=run_morning_scan,
            daemon=True
        ).start()

    # Continuous universe scanner.
    threading.Thread(
        target=universe_monitor,
        daemon=True
    ).start()

    # Continuous intraday scanner.
    threading.Thread(
        target=intraday_watcher,
        daemon=True
    ).start()

    # Daily scheduler.
    threading.Thread(
        target=scheduler_loop,
        daemon=True
    ).start()

    tg_send(
        "â *NSE AI Breakout Bot V2 is ONLINE*\n"
        "â¢ Broad NSE scan\n"
        "â¢ Daily bullish pattern detection\n"
        "â¢ Golden Cross / Double Bottom\n"
        "â¢ News catalyst scoring\n"
        "â¢ Morning watchlist\n"
        "â¢ Intraday volume-spike breakout alerts\n"
        "â¢ Continuous new-stock discovery\n"
        "â¢ No Render/Flask dependency"
    )

    # Telegram polling must remain in the main thread.
    if bot:
        log.info("Telegram polling started.")
        bot.infinity_polling(
            timeout=30,
            long_polling_timeout=30
        )
    else:
        # Keep process alive even without Telegram.
        while True:
            time.sleep(60)


if __name__ == "__main__":
    main()
