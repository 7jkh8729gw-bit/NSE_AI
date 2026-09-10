"""
NSE Bearish Breakdown Radar — informational/defensive screener
================================================================
Mirror image of nse_ai_breakout_actions.py, but looking for WEAKNESS
instead of strength: death cross, double top, breakdown below support,
bearish candlestick patterns, near 52-week lows, volume expansion on a
down move.

This is INFORMATIONAL ONLY — no SL/target, no trade instructions. Purpose:
- Avoid taking a fresh bullish entry in a stock that's actually breaking down.
- Get an early warning to consider exiting an existing long position.

Two commands, mirroring the bullish bot's structure:
    python nse_bearish_breakdown_bot.py scan      -> pre-market: full universe
                                                      scan, builds weak-stocks
                                                      watchlist, sends report
    python nse_bearish_breakdown_bot.py recheck   -> intraday: checks watchlist
                                                      for confirmed breakdowns

Runs every day regardless of overall market mood (per your instruction) —
reuses the same NSE universe, data fetchers, and Telegram plumbing as the
bullish bot to avoid duplicating tested infrastructure.
"""

"""
NSE Bearish Breakdown Bot — intraday short-signal engine
==========================================================
Standalone bot, deliberately self-contained (no cross-repo imports) since
it lives in its own repository, separate from the bullish breakout bot.
Shared utility functions below are copied from that bot rather than
imported, to avoid a cross-repo dependency GitHub Actions can't resolve.

Same core methodology as the bullish bot, adapted for shorting rather than
mirrored blindly:
- Intraday ATR (not daily) sizes SL/T1/T2, since a cash-segment short must
  close same-day — a multi-day-sized target makes no sense compressed into
  one afternoon.
- No new entries after SHORT_ENTRY_CUTOFF — not enough runway left before
  the mandatory square-off.
- Extension guard, mirrored: don't short something that already crashed
  too far since it first qualified — same chasing risk, just downward.
- Mandatory square-off reminder — cash-segment shorts cannot be carried
  overnight; this is an exchange/settlement reality, not a design choice.

Commands:
    scan                -> pre-market: full universe scan, builds short
                            watchlist, sends report
    rescan              -> intraday: finds new weakening stocks during
                            the day, merges into watchlist
    recheck             -> intraday: checks watchlist for confirmed
                            breakdowns, fires short signals
    squareoff_reminder  -> once daily: mandatory reminder to close any
                            open short before the deadline
    eod_finalize        -> once daily: resolves today's pending signal
                            outcomes for the learning loop
    retrain             -> weekly: refits scoring weights from resolved
                            outcomes
"""

import os
import re
import sys
import math
import logging
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, time as dt_time
from concurrent.futures import ThreadPoolExecutor, as_completed

import json
import pandas as pd
import yfinance as yf
import requests
import telebot

from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

log = logging.getLogger("NSE-BEARISH-BOT")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
bot = telebot.TeleBot(BOT_TOKEN) if BOT_TOKEN else None

MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))
MIN_PRICE = float(os.getenv("MIN_PRICE", "100"))
MIN_AVG_VOLUME = int(os.getenv("MIN_AVG_VOLUME", "500000"))
MIN_MARKET_CAP_CR = float(os.getenv("MIN_MARKET_CAP_CR", "1000"))
MIN_DAY_VOLUME = int(os.getenv("MIN_DAY_VOLUME", "200000"))
INTRADAY_INTERVAL = os.getenv("INTRADAY_INTERVAL", "5m")
MAX_NEWS_ITEMS = int(os.getenv("MAX_NEWS_ITEMS", "5"))
MIN_SAMPLES_FOR_LEARNING = int(os.getenv("MIN_SAMPLES_FOR_LEARNING", "30"))
LEARNING_FULL_INFLUENCE_SAMPLES = int(os.getenv("LEARNING_FULL_INFLUENCE_SAMPLES", "150"))
RETRAIN_MIN_DATE = os.getenv("RETRAIN_MIN_DATE", "").strip()


# ============================================================
# TIME HELPERS
# ============================================================

def now_ist():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


def today_str():
    return now_ist().strftime("%Y-%m-%d")


def market_is_open():
    n = now_ist()
    if n.weekday() >= 5:
        return False
    return dt_time(9, 15) <= n.time() <= dt_time(15, 30)


def session_elapsed_fraction():
    if not market_is_open():
        return None
    n = now_ist().time()

    def _minutes(t):
        return t.hour * 60 + t.minute

    start_min, end_min = _minutes(dt_time(9, 15)), _minutes(dt_time(15, 30))
    fraction = (_minutes(n) - start_min) / (end_min - start_min)
    return max(0.05, min(1.0, fraction))


# ============================================================
# PERSISTENCE
# ============================================================

def load_json(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception as e:
        log.warning("Could not load %s: %s", path, e)
    return default


def save_json(path, obj):
    try:
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
        log.info("Saved %s", path)
    except Exception as e:
        log.warning("Could not save %s: %s", path, e)


# ============================================================
# TELEGRAM
# ============================================================

def sanitize_for_markdown(text):
    if not text:
        return text
    return (
        text.replace("*", "")
        .replace("_", " ")
        .replace("`", "'")
        .replace("[", "(")
        .replace("]", ")")
    )


def tg_send(text, parse_mode="Markdown"):
    if not bot or not CHAT_ID:
        log.warning("Telegram is not configured.")
        return None
    try:
        return bot.send_message(CHAT_ID, text, parse_mode=parse_mode, disable_web_page_preview=True)
    except Exception as e:
        log.warning("Telegram Markdown send failed (%s) — retrying as plain text.", e)
        try:
            return bot.send_message(CHAT_ID, text, parse_mode=None, disable_web_page_preview=True)
        except Exception as e2:
            log.warning("Telegram plain-text retry also failed: %s", e2)
            return None


def tg_long_send(text):
    max_len = 3500
    lines = text.split("\n")
    chunks, current = [], ""
    for line in lines:
        if len(current) + len(line) + 1 > max_len:
            chunks.append(current)
            current = line
        else:
            current = current + "\n" + line if current else line
    if current:
        chunks.append(current)
    for chunk in chunks:
        tg_send(chunk)


# ============================================================
# NSE UNIVERSE / DATA FETCHERS
# ============================================================

def get_all_nse_stocks():
    log.info("Loading NSE universe...")
    try:
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0"})
        session.get("https://www.nseindia.com", timeout=10)
        resp = session.get("https://archives.nseindia.com/content/equities/EQUITY_L.csv", timeout=15)
        if resp.status_code == 200 and "SYMBOL" in resp.text[:200]:
            from io import StringIO
            df = pd.read_csv(StringIO(resp.text))
            symbols = sorted(set(
                s for s in df["SYMBOL"].astype(str).str.strip().str.upper()
                if re.fullmatch(r"[A-Z0-9&._-]+", s)
            ))
            if len(symbols) > 500:
                log.info("NSE universe loaded from NSE archives: %s symbols", len(symbols))
                return symbols
    except Exception as e:
        log.warning("NSE archives list failed: %s", e)

    try:
        from datasets import load_dataset
        ds = load_dataset("tickertruth/nse-india-security-master", data_files="data/nse_security_master.csv")
        df = ds["train"].to_pandas()
        df = df[df["active_flag"] == True]
        symbols = sorted(set(
            s for s in df["nse_symbol"].astype(str).str.strip().str.upper()
            if re.fullmatch(r"[A-Z0-9&._-]+", s)
        ))
        log.info("NSE universe loaded from Hugging Face fallback: %s symbols", len(symbols))
        return symbols
    except Exception as e:
        log.exception("Universe loading failed: %s", e)
        return ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]


def yf_daily(symbol, period="1y"):
    try:
        df = yf.download(f"{symbol}.NS", period=period, interval="1d", auto_adjust=False, progress=False, threads=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        required = ["Open", "High", "Low", "Close", "Volume"]
        if not all(c in df.columns for c in required):
            return None
        return df[required].copy().dropna()
    except Exception as e:
        log.debug("Daily data error %s: %s", symbol, e)
        return None


def yf_intraday(symbol):
    try:
        df = yf.download(f"{symbol}.NS", period="2d", interval=INTRADAY_INTERVAL, auto_adjust=False, progress=False, threads=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        required = ["Open", "High", "Low", "Close", "Volume"]
        if not all(c in df.columns for c in required):
            return None
        return df[required].dropna()
    except Exception as e:
        log.debug("Intraday data error %s: %s", symbol, e)
        return None


def _fast_info_get(fi, *keys, default=0):
    for k in keys:
        try:
            v = fi[k]
            if v is not None:
                return v
        except Exception:
            pass
        try:
            v = getattr(fi, k)
            if v is not None:
                return v
        except Exception:
            pass
    return default


def get_info(symbol):
    try:
        fi = yf.Ticker(f"{symbol}.NS").fast_info
        price = float(_fast_info_get(fi, "last_price", "lastPrice") or 0)
        prev_close = float(_fast_info_get(fi, "previous_close", "regularMarketPreviousClose", "previousClose") or 0)
        volume = int(_fast_info_get(fi, "last_volume", "regularMarketVolume") or 0)
        market_cap = float(_fast_info_get(fi, "market_cap", "marketCap") or 0)
        if not market_cap:
            shares = _fast_info_get(fi, "shares", "shares_outstanding")
            if shares and price:
                market_cap = float(shares) * price
        return {"price": price, "prev_close": prev_close, "volume": volume, "high_52w": 0, "market_cap": market_cap / 1e7}
    except Exception as e:
        log.debug("fast_info failed for %s: %s", symbol, e)
        return {"price": 0, "prev_close": 0, "volume": 0, "high_52w": 0, "market_cap": 0}


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
    macd = MACD(x["Close"], window_slow=26, window_fast=12, window_sign=9)
    x["MACD"] = macd.macd()
    x["MACDSignal"] = macd.macd_signal()
    x["MACDHist"] = macd.macd_diff()
    x["ATR"] = AverageTrueRange(x["High"], x["Low"], x["Close"], window=14).average_true_range()
    x["OBV"] = OnBalanceVolumeIndicator(x["Close"], x["Volume"]).on_balance_volume()
    x["ADX"] = ADXIndicator(x["High"], x["Low"], x["Close"], window=14).adx()
    x["AvgVol20"] = x["Volume"].rolling(20).mean()
    x["AvgVol50"] = x["Volume"].rolling(50).mean()
    return x


# ============================================================
# NEWS / CATALYST ENGINE
# ============================================================

BULLISH_WORDS = {
    "order": 3, "contract": 3, "wins": 2, "win": 2, "approval": 3, "approved": 3,
    "launch": 2, "expansion": 2, "acquisition": 2, "merger": 2, "earnings": 1,
    "profit": 3, "profits": 3, "revenue": 2, "growth": 2, "upgrade": 3, "buy": 2,
    "target": 1, "capacity": 2, "investment": 2, "partnership": 2, "export": 2,
    "record": 2, "strong": 1, "positive": 2, "surge": 2, "rises": 1,
}

BEARISH_WORDS = {
    "fraud": 6, "default": 5, "downgrade": 4, "loss": 3, "losses": 3,
    "decline": 2, "falls": 2, "fall": 2, "probe": 4, "investigation": 4,
    "resign": 3, "resignation": 3, "warning": 2, "debt": 2, "lawsuit": 3,
    "penalty": 3, "cut": 2, "weak": 2, "negative": 2,
}


def clean_text(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


def news_sentiment(text):
    text = text.lower()
    bull = sum(w for word, w in BULLISH_WORDS.items() if re.search(r"\b" + re.escape(word) + r"\b", text))
    bear = sum(w for word, w in BEARISH_WORDS.items() if re.search(r"\b" + re.escape(word) + r"\b", text))
    raw = bull - bear
    label = "Bullish" if raw >= 5 else "Bearish" if raw <= -4 else "Neutral"
    return raw, label


def fetch_google_news(symbol):
    try:
        q = urllib.parse.quote(f"{symbol} NSE India stock")
        url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read()
        root = ET.fromstring(xml_data)
        items = []
        for item in root.findall(".//item")[:MAX_NEWS_ITEMS]:
            title = clean_text(item.findtext("title"))
            if title:
                score, label = news_sentiment(title)
                items.append({
                    "title": title, "link": clean_text(item.findtext("link")),
                    "published": clean_text(item.findtext("pubDate")),
                    "score": score, "label": label,
                })
        return items
    except Exception as e:
        log.debug("News error %s: %s", symbol, e)
        return []


def compute_news_score(symbol):
    items = fetch_google_news(symbol)
    if not items:
        return {"score": 0, "label": "No recent news", "headlines": []}

    total = sum(i["score"] for i in items)
    bullish = sum(1 for i in items if i["label"] == "Bullish")
    bearish = sum(1 for i in items if i["label"] == "Bearish")

    score = 50 + total * 5
    score += min(15, bullish * 5)
    score -= min(20, bearish * 7)
    score = max(0, min(100, score))
    label = "Bullish" if score >= 65 else "Bearish" if score <= 35 else "Neutral"

    return {"score": round(score, 1), "label": label, "headlines": items}


# ============================================================
# SELF-LEARNING SCORING (generic, weights supplied by our own retrain)
# ============================================================

def learned_adjustment(feature_dict, weights):
    if not weights:
        return None, 0.0
    try:
        n_samples = weights.get("n_samples", 0)
        if n_samples < MIN_SAMPLES_FOR_LEARNING:
            return None, 0.0
        names, mean, scale = weights["features"], weights["mean"], weights["scale"]
        coef, intercept = weights["coef"], weights["intercept"]
        z = intercept
        for i, name in enumerate(names):
            x = feature_dict.get(name, 0.0)
            denom = scale[i] if scale[i] else 1.0
            z += coef[i] * ((x - mean[i]) / denom)
        prob = 1.0 / (1.0 + math.exp(-z))
        blend_ratio = min(
            1.0,
            max(0.0, (n_samples - MIN_SAMPLES_FOR_LEARNING)) /
            max(1, (LEARNING_FULL_INFLUENCE_SAMPLES - MIN_SAMPLES_FOR_LEARNING))
        )
        return prob, blend_ratio
    except Exception as e:
        log.debug("learned_adjustment failed: %s", e)
        return None, 0.0


DATA_DIR = "data_bearish"
BEAR_WATCHLIST_FILE = os.path.join(DATA_DIR, "bearish_watchlist.json")
BEAR_ALERT_STATE_FILE = os.path.join(DATA_DIR, "bearish_alert_state.json")
BEAR_ALERT_HISTORY_FILE = os.path.join(DATA_DIR, "bearish_alert_history.json")
BEAR_MODEL_WEIGHTS_FILE = os.path.join(DATA_DIR, "bearish_model_weights.json")
os.makedirs(DATA_DIR, exist_ok=True)

# --- Config (mirrors the bullish bot's thresholds, inverted) ---
BEAR_MAX_DAY_DECLINE = float(os.getenv("BEAR_MAX_DAY_DECLINE", "10"))   # stock down 0% to -10%
BEAR_MIN_DAY_DECLINE = float(os.getenv("BEAR_MIN_DAY_DECLINE", "0.5"))  # must be down at least this much
BEAR_MAX_FROM_52W_LOW = float(os.getenv("BEAR_MAX_FROM_52W_LOW", "10")) # within 10% of 52w low
BEAR_MIN_VOLUME_RATIO = float(os.getenv("BEAR_MIN_VOLUME_RATIO", "1.5"))

# Excludes the most squeeze-prone candidates — deeply oversold + near a
# 52-week low is the classic short-squeeze setup. This is a floor, not a
# ceiling: RSI below this is excluded entirely, not just scored lower.
BEAR_MIN_RSI = float(os.getenv("BEAR_MIN_RSI", "25"))
BEAR_WATCHLIST_MAX_SIZE = int(os.getenv("BEAR_WATCHLIST_MAX_SIZE", "20"))
BEAR_MIN_WARN_SCORE = float(os.getenv("BEAR_MIN_WARN_SCORE", "70"))
BEAR_MIN_INTRADAY_VOLUME_RATIO = float(os.getenv("BEAR_MIN_INTRADAY_VOLUME_RATIO", "1.8"))

# --- Real short-trade signal config (cash-market, intraday-only) ---
# Daily ATR is the WRONG volatility measure for a position that must close
# same-day — it's sized for multi-day moves. These multipliers apply to
# INTRADAY (5-min bar) ATR instead, computed fresh at signal time.
SHORT_SL_ATR_MULT = float(os.getenv("SHORT_SL_ATR_MULT", "2.5"))
SHORT_T1_ATR_MULT = float(os.getenv("SHORT_T1_ATR_MULT", "3.5"))
SHORT_T2_ATR_MULT = float(os.getenv("SHORT_T2_ATR_MULT", "6.0"))
INTRADAY_ATR_PERIOD = int(os.getenv("INTRADAY_ATR_PERIOD", "14"))

# No new entries fired after this time — not enough runway left before the
# mandatory square-off to reasonably expect a target to develop.
SHORT_ENTRY_CUTOFF_HOUR = int(os.getenv("SHORT_ENTRY_CUTOFF_HOUR", "14"))
SHORT_ENTRY_CUTOFF_MINUTE = int(os.getenv("SHORT_ENTRY_CUTOFF_MINUTE", "30"))

# Exchange-enforced deadline — same concept as MIS square-off cutoffs most
# brokers apply, slightly before NSE's own 3:30 PM close for safety margin.
MANDATORY_SQUAREOFF_TIME_TEXT = os.getenv("MANDATORY_SQUAREOFF_TIME_TEXT", "3:20 PM")

# Same lesson as the long bot's extension guard, inverted: don't short a
# stock that has already crashed too far intraday — that risk is identical
# to chasing an extended breakout, just pointed the other direction.
SHORT_EXTENSION_CAP_PCT = float(os.getenv("SHORT_EXTENSION_CAP_PCT", "6.0"))



# ============================================================
# BEARISH CANDLESTICK PATTERNS (mirror of bullish set)
# ============================================================

def candle_body(row):
    return abs(row["Close"] - row["Open"])


def detect_bearish_candlestick_patterns(df):
    patterns = []
    if df is None or len(df) < 5:
        return patterns

    x = df.iloc[-5:].copy()
    last, prev, p2 = x.iloc[-1], x.iloc[-2], x.iloc[-3]
    body = candle_body(last)

    # Bearish engulfing
    if (
        prev["Close"] > prev["Open"]
        and last["Close"] < last["Open"]
        and last["Open"] >= prev["Close"]
        and last["Close"] <= prev["Open"]
    ):
        patterns.append("Bearish Engulfing")

    # Shooting Star / Hanging Man
    if body > 0:
        upper = last["High"] - max(last["Open"], last["Close"])
        lower = min(last["Open"], last["Close"]) - last["Low"]
        if upper >= 2 * body and lower <= body * 0.5:
            patterns.append("Shooting Star")

    # Evening Star
    if (
        p2["Close"] > p2["Open"]
        and abs(prev["Close"] - prev["Open"]) <= abs(p2["Close"] - p2["Open"]) * 0.5
        and last["Close"] < last["Open"]
        and last["Close"] < (p2["Open"] + p2["Close"]) / 2
    ):
        patterns.append("Evening Star")

    # 3 Black Crows
    if len(x) >= 3:
        a, b, c = x.iloc[-3], x.iloc[-2], x.iloc[-1]
        if (
            a["Close"] < a["Open"]
            and b["Close"] < b["Open"]
            and c["Close"] < c["Open"]
            and b["Close"] < a["Close"]
            and c["Close"] < b["Close"]
        ):
            patterns.append("3 Black Crows")

    return patterns


# ============================================================
# BEARISH STRUCTURAL SIGNALS (mirror of bullish set)
# ============================================================

def detect_death_cross(df):
    if len(df) < 210:
        return False
    ema50, ema200 = df["EMA50"], df["EMA200"]
    recent = ema50.iloc[-10:] < ema200.iloc[-10:]
    alignment = ema50.iloc[-1] < ema200.iloc[-1]
    crossover = False
    for i in range(1, min(15, len(df))):
        a = ema50.iloc[-i - 1] - ema200.iloc[-i - 1]
        b = ema50.iloc[-i] - ema200.iloc[-i]
        if a >= 0 and b < 0:
            crossover = True
            break
    return bool(alignment and (crossover or recent.sum() >= 7))


def detect_head_and_shoulders(df):
    """
    Bearish reversal: three peaks — left shoulder, head (the highest),
    right shoulder — with the two shoulders roughly level. Mirror of the
    bullish inverse-H&S detector.
    """
    if len(df) < 60:
        return False

    highs = df["High"].values
    lows = df["Low"].values
    close = df["Close"].values
    lookback = min(120, len(df))
    start = len(df) - lookback
    window = 5

    candidates = []
    for i in range(start + window, len(df) - window):
        left = highs[i - window:i]
        right = highs[i + 1:i + 1 + window]
        if highs[i] > left.max() and highs[i] > right.max():
            candidates.append(i)

    if len(candidates) < 3:
        return False

    for idx in range(len(candidates) - 2):
        l_idx, h_idx, r_idx = candidates[idx], candidates[idx + 1], candidates[idx + 2]
        if h_idx - l_idx < 8 or r_idx - h_idx < 8:
            continue
        if (r_idx - l_idx) > 90:
            continue
        if (len(df) - 1 - r_idx) > 40:
            continue

        L, H, R = highs[l_idx], highs[h_idx], highs[r_idx]
        if not (H > L and H > R):
            continue

        avg_shoulder = (L + R) / 2
        if avg_shoulder <= 0:
            continue
        if abs(L - R) / avg_shoulder > 0.07:
            continue
        if (H - avg_shoulder) / avg_shoulder < 0.03:
            continue  # head not meaningfully higher — not a real H&S

        neckline = max(lows[l_idx:h_idx + 1].min(), lows[h_idx:r_idx + 1].min())
        if close[-1] <= neckline * 1.02:
            return True

    return False


def detect_rounding_top(df):
    """
    Bearish continuation/reversal: inverted cup-and-handle — a rounded top
    followed by a shallow bounce ('handle') before breaking down. Fuzzier
    than a geometric pattern like H&S — treat with more skepticism until
    outcome data validates it.
    """
    if len(df) < 80:
        return False

    n = len(df)
    cup_window = min(90, n - 10)
    cup = df.iloc[-(cup_window + 10):-10] if n > cup_window + 10 else df.iloc[:-10]
    if len(cup) < 30:
        return False

    left_rim = float(cup["Low"].iloc[:10].min())
    right_rim = float(cup["Low"].iloc[-10:].min())
    avg_rim = (left_rim + right_rim) / 2
    if avg_rim <= 0:
        return False
    if abs(left_rim - right_rim) / avg_rim > 0.08:
        return False

    top_pos = int(cup["High"].values.argmax())
    rel_pos = top_pos / len(cup)
    if rel_pos < 0.25 or rel_pos > 0.75:
        return False

    cup_top = float(cup["High"].iloc[top_pos])
    height_pct = (cup_top - avg_rim) / avg_rim * 100
    if height_pct < 12 or height_pct > 50:
        return False

    handle = df.iloc[-10:]
    handle_high = float(handle["High"].max())
    handle_bounce_pct = (handle_high - right_rim) / right_rim * 100 if right_rim > 0 else 100
    if handle_bounce_pct > 15:
        return False

    recent_close = float(df["Close"].iloc[-1])
    return recent_close <= right_rim * 1.01


def detect_bear_flag(df):
    """
    Bearish continuation: a sharp decline (the flagpole) followed by a
    brief, tight consolidation (the flag), then a break below the flag's
    low. Mirror of the bullish bull-flag detector.
    """
    if len(df) < 25:
        return False

    pole = df.iloc[-20:-6]
    flag = df.iloc[-6:]
    if len(pole) < 8 or len(flag) < 4:
        return False

    pole_start = float(pole["Close"].iloc[0])
    pole_end = float(pole["Close"].iloc[-1])
    if pole_start <= 0:
        return False
    pole_loss_pct = (pole_start - pole_end) / pole_start * 100
    if pole_loss_pct < 12:
        return False

    flag_high = float(flag["High"].max())
    flag_low = float(flag["Low"].min())
    if flag_low <= 0:
        return False
    flag_range_pct = (flag_high - flag_low) / flag_low * 100
    if flag_range_pct > 10:
        return False

    pole_move = pole_start - pole_end
    if pole_move > 0:
        retrace_pct = (flag_high - pole_end) / pole_move * 100
        if retrace_pct > 50:
            return False

    recent_close = float(df["Close"].iloc[-1])
    return recent_close <= flag_low * 1.005


def detect_double_top(df):
    """
    Stricter double-top detector — mirrors the fix applied to the bullish
    bot's double-bottom detector. Requires genuine swing highs (5-bar
    window, strict greater-than) and a MEANINGFUL drop between the two
    tops (the actual "M" shape), not just two vaguely-similar highs
    anywhere in a 120-day window.
    """
    if len(df) < 60:
        return False

    close = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values

    lookback = min(120, len(df))
    start = len(df) - lookback
    window = 5
    min_drop_pct = 8.0

    candidates = []
    for i in range(start + window, len(df) - window):
        left = highs[i - window:i]
        right = highs[i + 1:i + 1 + window]
        if highs[i] > left.max() and highs[i] > right.max():
            candidates.append(i)

    if len(candidates) < 2:
        return False

    for a_idx in candidates[:-1]:
        for b_idx in candidates:
            if b_idx <= a_idx:
                continue
            distance = b_idx - a_idx
            if distance < 15 or distance > 60:
                continue
            if (len(df) - 1 - b_idx) > 40:
                continue

            a, b = highs[a_idx], highs[b_idx]
            avg_high = (a + b) / 2
            if avg_high <= 0:
                continue
            if abs(a - b) / avg_high > 0.04:
                continue

            middle_trough = lows[a_idx:b_idx + 1].min()
            drop_pct = ((avg_high - middle_trough) / avg_high) * 100
            if drop_pct < min_drop_pct:
                continue  # no real "M" shape — just noise near a similar level

            recent_close = close[-1]
            if recent_close <= middle_trough * 1.03:
                return True

    return False


def detect_lower_low_lower_high(df):
    if len(df) < 30:
        return False
    x = df.iloc[-30:]
    recent_low = x["Low"].iloc[-1]
    previous_low = x["Low"].iloc[-15:-3].min()
    recent_high = x["High"].iloc[-1]
    previous_high = x["High"].iloc[-15:-3].max()
    return recent_low <= previous_low and recent_high <= previous_high * 1.005


def detect_near_breakdown(df):
    if len(df) < 30:
        return False
    support = df["Low"].iloc[-21:-1].min()
    close = df["Close"].iloc[-1]
    return close <= support * 1.015


def detect_breakdown(df):
    if len(df) < 25:
        return False
    support = df["Low"].iloc[-21:-1].min()
    close = df["Close"].iloc[-1]
    volume = df["Volume"].iloc[-1]
    avg = df["Volume"].iloc[-21:-1].mean()
    return close < support and avg > 0 and volume >= avg * 1.5


# ============================================================
# DAILY BEARISH ANALYSIS
# ============================================================

def analyze_bearish_daily(symbol, df):
    if df is None or len(df) < 210:
        return None

    x = add_indicators(df)
    last, prev = x.iloc[-1], x.iloc[-2]
    close = float(last["Close"])
    atr = float(last["ATR"]) if not pd.isna(last["ATR"]) else 0
    if close <= 0:
        return None

    patterns = detect_bearish_candlestick_patterns(x)
    death_cross = detect_death_cross(x)
    double_top = detect_double_top(x)
    head_shoulders = detect_head_and_shoulders(x)
    rounding_top = detect_rounding_top(x)
    bear_flag = detect_bear_flag(x)
    ll_lh = detect_lower_low_lower_high(x)
    near_breakdown = detect_near_breakdown(x)
    breakdown = detect_breakdown(x)

    macd_cross_down = (
        last["MACD"] < last["MACDSignal"]
        and prev["MACD"] >= prev["MACDSignal"]
    )
    rsi = float(last["RSI"]) if not pd.isna(last["RSI"]) else 50
    adx = float(last["ADX"]) if not pd.isna(last["ADX"]) else 0

    obv_distribution = len(x) >= 10 and x["OBV"].iloc[-1] < x["OBV"].iloc[-6]

    ema_alignment_bearish = last["EMA10"] < last["EMA20"] < last["EMA50"] < last["EMA200"]
    dema_alignment_bearish = last["DEMA10"] < last["DEMA50"] < last["DEMA200"]

    volume_ratio = (
        float(last["Volume"]) / float(last["AvgVol20"])
        if last["AvgVol20"] and not pd.isna(last["AvgVol20"]) else 0
    )

    score = 0
    reasons = []

    if ema_alignment_bearish:
        score += 8; reasons.append("EMA bearish alignment")
    if dema_alignment_bearish:
        score += 7; reasons.append("DEMA bearish alignment")
    if death_cross:
        score += 8; reasons.append("Death Cross")
    if ll_lh:
        score += 7; reasons.append("Lower High / Lower Low")
    if double_top:
        score += 10; reasons.append("Double Top")
    if head_shoulders:
        score += 11; reasons.append("Head & Shoulders")
    if rounding_top:
        score += 8; reasons.append("Rounding Top")
    if bear_flag:
        score += 8; reasons.append("Bear Flag")
    if near_breakdown:
        score += 6; reasons.append("Near Breakdown")
    if breakdown:
        score += 9; reasons.append("Confirmed Breakdown")
    if patterns:
        score += min(15, len(patterns) * 5); reasons.extend(patterns)
    if 32 <= rsi <= 50:
        score += 6; reasons.append("Weak RSI")
    if macd_cross_down:
        score += 6; reasons.append("MACD Bearish Crossover")
    elif last["MACD"] < last["MACDSignal"]:
        score += 3; reasons.append("MACD Bearish")
    if adx >= 20:
        score += 3; reasons.append("ADX Trend Strength")
    if volume_ratio >= BEAR_MIN_VOLUME_RATIO:
        score += 5; reasons.append("Volume Expansion")
    if obv_distribution:
        score += 5; reasons.append("OBV Distribution")
    if volume_ratio >= 2:
        score += 5; reasons.append("Strong Volume")

    score = min(100, score)

    setup_candidates = [
        ("DEATH CROSS + BREAKDOWN", death_cross and breakdown),
        ("HEAD & SHOULDERS BREAKDOWN", head_shoulders and breakdown),
        ("DOUBLE TOP BREAKDOWN", double_top and breakdown),
        ("BREAKDOWN", breakdown),
        ("HEAD & SHOULDERS", head_shoulders),
        ("DOUBLE TOP", double_top),
        ("ROUNDING TOP", rounding_top),
        ("DEATH CROSS", death_cross),
        ("BEAR FLAG", bear_flag),
        ("CANDLESTICK REVERSAL", bool(patterns) and near_breakdown),
        ("NEAR BREAKDOWN", near_breakdown),
        ("DOWNTREND CONTINUATION", ll_lh),
        ("BEARISH", True),
    ]
    setup = next(label for label, matched in setup_candidates if matched)

    # A meaningful multi-week support level, used later by the intraday
    # engine so a "breakdown" means breaking real support, not just dipping
    # below the last hour's minor low.
    support_20d = float(x["Low"].tail(20).min())

    return {
        "symbol": symbol, "close": close, "rsi": round(rsi, 2), "adx": round(adx, 2),
        "atr": round(atr, 2), "volume_ratio": round(volume_ratio, 2),
        "ema_alignment_bearish": bool(ema_alignment_bearish),
        "dema_alignment_bearish": bool(dema_alignment_bearish),
        "death_cross": bool(death_cross), "double_top": bool(double_top),
        "head_shoulders": bool(head_shoulders), "rounding_top": bool(rounding_top),
        "bear_flag": bool(bear_flag),
        "ll_lh": bool(ll_lh), "near_breakdown": bool(near_breakdown),
        "breakdown": bool(breakdown), "macd_bearish": bool(last["MACD"] < last["MACDSignal"]),
        "macd_cross_down": bool(macd_cross_down), "obv_distribution": bool(obv_distribution),
        "patterns": patterns, "setup": setup, "score": score, "reasons": reasons,
        "support_20d": round(support_20d, 2),
    }


# ============================================================
# CORE BEARISH FILTERS (mirror of apply_core_filters, inverted)
# ============================================================

def apply_bearish_filters(symbol, df, info=None):
    if df is None or len(df) < 210:
        return None

    x = add_indicators(df)
    last = x.iloc[-1]
    price = float(last["Close"])
    volume = float(last["Volume"])
    avg_volume = float(x["Volume"].tail(21).mean())

    elapsed_fraction = session_elapsed_fraction()
    volume_denominator = (
        avg_volume * elapsed_fraction if elapsed_fraction is not None else avg_volume
    )

    prev_close_daily = float(x["Close"].iloc[-2]) if len(x) >= 2 else price
    day_change_daily = (
        ((price - prev_close_daily) / prev_close_daily) * 100
        if prev_close_daily > 0 else 0
    )
    volume_ratio_daily = volume / volume_denominator if volume_denominator > 0 else 0

    if price < MIN_PRICE:
        return None
    if avg_volume <= MIN_AVG_VOLUME:
        return None
    if volume < MIN_DAY_VOLUME:
        return None
    # Stock must be DOWN today, but not so far down it's already crashed —
    # this is a "weakness developing" screener, not a "circuit hit" alert.
    if not (-BEAR_MAX_DAY_DECLINE - 2 <= day_change_daily <= -BEAR_MIN_DAY_DECLINE + 1):
        return None
    if volume_ratio_daily < BEAR_MIN_VOLUME_RATIO * 0.9:
        return None
    if not (last["DEMA10"] < last["DEMA50"] < last["DEMA200"]):
        return None

    if info is None:
        info = get_info(symbol)

    prev_close = info["prev_close"] or prev_close_daily
    low_52w = float(x["Low"].tail(252).min())
    market_cap = info["market_cap"]

    day_change = ((price - prev_close) / prev_close) * 100 if prev_close > 0 else 0
    volume_ratio = volume / volume_denominator if volume_denominator > 0 else 0
    pct_from_low = ((price - low_52w) / price) * 100 if price > 0 else 100

    if market_cap < MIN_MARKET_CAP_CR:
        return None
    if not (-BEAR_MAX_DAY_DECLINE <= day_change <= -BEAR_MIN_DAY_DECLINE):
        return None
    if pct_from_low > BEAR_MAX_FROM_52W_LOW:
        return None
    if volume_ratio < BEAR_MIN_VOLUME_RATIO:
        return None

    return {
        "symbol": symbol, "price": price, "day_change": day_change,
        "volume_ratio": volume_ratio, "market_cap": market_cap,
        "pct_from_low": pct_from_low, "avg_volume": avg_volume, "volume": volume,
    }


def analyze_bearish_candidate(symbol):
    try:
        df = yf_daily(symbol)
        base = apply_bearish_filters(symbol, df)
        if not base:
            return None
        tech = analyze_bearish_daily(symbol, df)
        if not tech:
            return None

        # Deeply oversold + already near a 52-week low is the classic
        # short-squeeze setup (see: "momentum crashes" — the short leg of
        # momentum strategies, i.e. beaten-down stocks, is the one prone to
        # violent snap-back rallies, especially on any market-wide bounce).
        # Excluding the most extreme cases trims the riskiest tail rather
        # than removing the near-52w-low filter itself.
        if tech["rsi"] < BEAR_MIN_RSI:
            return None

        news_data = compute_news_score(symbol)
        news_score = news_data["score"]
        # For a bearish screener, a NEGATIVE news catalyst reinforces the
        # signal — so we invert: low news_score (bearish/neutral news) is
        # what we want to see confirming weakness, not a bullish headline
        # that might reverse it.
        bearish_news_score = round(100 - news_score, 2)
        chart_score = tech["score"]
        combined = tech["score"] * 0.6 + bearish_news_score * 0.2 + chart_score * 0.2
        return {
            "symbol": symbol, "base": base, "technical": tech,
            "news": {"score": news_score, "label": news_data.get("label", "Neutral"),
                     "headlines": news_data.get("headlines", [])},
            "combined_score": round(combined, 1),
            "added_at": now_ist().isoformat(),
        }
    except Exception as e:
        log.debug("Bearish analysis failed for %s: %s", symbol, e)
        return None


def scan_bearish_universe(symbols):
    log.info("Starting bearish universe scan: %s stocks", len(symbols))
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(analyze_bearish_candidate, s): s for s in symbols}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            try:
                r = future.result()
                if r:
                    results.append(r)
            except Exception:
                pass
            if completed % 100 == 0:
                log.info("Bearish progress: %s/%s | candidates=%s", completed, len(symbols), len(results))
    results.sort(key=lambda x: x["combined_score"], reverse=True)
    log.info("Bearish scan complete: %s candidates", len(results))
    return results


def merge_bearish_watchlist(existing_items, new_results):
    watchlist = {item["symbol"]: item for item in existing_items}
    added = []

    for item in new_results:
        symbol = item["symbol"]
        if symbol not in watchlist:
            item["first_seen_price"] = item["base"]["price"]
            item["first_seen_at"] = item["added_at"]
            watchlist[symbol] = item
            added.append(item)
        else:
            old = watchlist[symbol]
            item["first_seen_price"] = old.get("first_seen_price", old["base"]["price"])
            item["first_seen_at"] = old.get("first_seen_at", old["added_at"])
            watchlist[symbol] = item

    merged = list(watchlist.values())
    if len(merged) > BEAR_WATCHLIST_MAX_SIZE:
        merged = sorted(merged, key=lambda x: x["combined_score"], reverse=True)[:BEAR_WATCHLIST_MAX_SIZE]

    return merged, added


# ============================================================
# REPORTING
# ============================================================

def format_bearish_report(results):
    msg = f"🔻 *Short Watchlist (Intraday, Cash Segment)*\n📅 {today_str()}\n"
    msg += f"🔎 Weak candidates: {len(results)}\n"
    msg += (
        f"⚠️ Any short entered from this list MUST be squared off by "
        f"{MANDATORY_SQUAREOFF_TIME_TEXT} — no overnight carry in cash segment.\n"
    )
    msg += "━" * 20 + "\n"

    for i, item in enumerate(results[:BEAR_WATCHLIST_MAX_SIZE], 1):
        b, t, n = item["base"], item["technical"], item["news"]
        patterns = ", ".join(t["patterns"]) if t["patterns"] else "None"
        headline = sanitize_for_markdown(n["headlines"][0]["title"]) if n.get("headlines") else "No recent news"

        msg += (
            f"*{i}. {item['symbol']}* — Weakness Score *{item['combined_score']}/100*\n"
            f"💰 ₹{b['price']:.2f} | 📉 {b['day_change']:.2f}% | 📊 Vol {b['volume_ratio']:.2f}x\n"
            f"📐 Setup: *{t['setup']}*\n"
            f"🧠 Patterns: {patterns}\n"
            f"📊 RSI {t['rsi']} | ADX {t['adx']} | MACD {'🔴' if t['macd_bearish'] else '🟢'}\n"
            f"📰 News ({n['label']}, {n['score']}/100): {headline}\n"
            f"━━━━━━━━━━━━━━━━\n"
        )

    msg += (
        f"\n⚡ Watching for intraday breakdown confirmation. "
        f"No new entries fired after {SHORT_ENTRY_CUTOFF_HOUR}:{SHORT_ENTRY_CUTOFF_MINUTE:02d} — "
        f"not enough time left before square-off."
    )
    return msg


def build_short_signal_message(signal, watch_item):
    symbol = signal["symbol"]
    t = watch_item["technical"]
    n = watch_item["news"]
    headline = sanitize_for_markdown(n["headlines"][0]["title"]) if n.get("headlines") else "No recent news"

    return (
        f"🔻 *SHORT SIGNAL*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📌 *{symbol}*\n"
        f"💰 Entry: ₹{signal['price']:.2f} (broke below ₹{signal['support']:.2f})\n"
        f"📊 Intraday Volume: {signal['volume_ratio']:.2f}x\n"
        f"🧠 Signal Score: {signal['score']}/100\n\n"
        f"📐 Daily Setup: {t['setup']} | Weakness Score: {watch_item['combined_score']}/100\n"
        f"📰 Catalyst ({n['label']}): {headline}\n\n"
        f"🛑 *SL:* ₹{signal['sl']:.2f}\n"
        f"🎯 *T1:* ₹{signal['target1']:.2f}\n"
        f"🌟 *T2:* ₹{signal['target2']:.2f}\n\n"
        f"⚠️ Square off by {MANDATORY_SQUAREOFF_TIME_TEXT} — reminder incoming separately.\n"
        f"⏰ {signal['time']}"
    )


# ============================================================
# PERSISTENCE
# ============================================================

def load_bearish_watchlist():
    data = load_json(BEAR_WATCHLIST_FILE, {})
    if not isinstance(data, dict) or "items" not in data:
        data = {"date": today_str(), "items": []}
    return data


def save_bearish_watchlist(items):
    save_json(BEAR_WATCHLIST_FILE, {"date": today_str(), "items": items})


def load_bearish_alert_state():
    data = load_json(BEAR_ALERT_STATE_FILE, {})
    if not isinstance(data, dict) or "date" not in data or data["date"] != today_str():
        data = {"date": today_str(), "alerted": []}
    return data


def save_bearish_alert_state(state):
    save_json(BEAR_ALERT_STATE_FILE, state)


# ============================================================
# INTRADAY BREAKDOWN CONFIRMATION
# ============================================================

# ============================================================
# SELF-LEARNING FEEDBACK LOOP — features + outcome tracking
# ============================================================

BEAR_FEATURE_NAMES = [
    "rsi", "adx", "daily_volume_ratio", "daily_score", "news_score",
    "combined_score", "intraday_score", "intraday_volume_ratio",
    "ema_alignment_bearish", "dema_alignment_bearish", "death_cross",
    "double_top", "head_shoulders", "rounding_top", "bear_flag",
    "ll_lh", "near_breakdown", "breakdown",
    "macd_bearish", "macd_cross_down", "obv_distribution", "has_patterns",
    "move_since_first_seen_pct",
]


def build_bear_feature_dict(watch_item, intraday_score, intraday_volume_ratio, move_since_first_seen_pct=0.0):
    t = watch_item["technical"]
    n = watch_item["news"]
    return {
        "rsi": t["rsi"], "adx": t["adx"],
        "daily_volume_ratio": watch_item["base"]["volume_ratio"],
        "daily_score": t["score"], "news_score": n["score"],
        "combined_score": watch_item["combined_score"],
        "intraday_score": intraday_score,
        "intraday_volume_ratio": intraday_volume_ratio,
        "ema_alignment_bearish": 1.0 if t["ema_alignment_bearish"] else 0.0,
        "dema_alignment_bearish": 1.0 if t["dema_alignment_bearish"] else 0.0,
        "death_cross": 1.0 if t["death_cross"] else 0.0,
        "double_top": 1.0 if t["double_top"] else 0.0,
        "head_shoulders": 1.0 if t.get("head_shoulders") else 0.0,
        "rounding_top": 1.0 if t.get("rounding_top") else 0.0,
        "bear_flag": 1.0 if t.get("bear_flag") else 0.0,
        "ll_lh": 1.0 if t["ll_lh"] else 0.0,
        "near_breakdown": 1.0 if t["near_breakdown"] else 0.0,
        "breakdown": 1.0 if t["breakdown"] else 0.0,
        "macd_bearish": 1.0 if t["macd_bearish"] else 0.0,
        "macd_cross_down": 1.0 if t["macd_cross_down"] else 0.0,
        "obv_distribution": 1.0 if t["obv_distribution"] else 0.0,
        "has_patterns": 1.0 if t["patterns"] else 0.0,
        "move_since_first_seen_pct": move_since_first_seen_pct,
    }


def _tz_naive_index(df):
    idx = pd.to_datetime(df.index)
    try:
        idx = idx.tz_localize(None)
    except TypeError:
        pass
    return idx


def load_bear_alert_history():
    data = load_json(BEAR_ALERT_HISTORY_FILE, [])
    return data if isinstance(data, list) else []


def save_bear_alert_history(records):
    save_json(BEAR_ALERT_HISTORY_FILE, records)


def load_bear_model_weights():
    data = load_json(BEAR_MODEL_WEIGHTS_FILE, None)
    if not data or "coef" not in data:
        return None
    return data


def record_new_short_alert(signal, watch_item):
    history = load_bear_alert_history()
    history.append({
        "date": today_str(), "symbol": signal["symbol"], "alert_time": signal["time"],
        "bar_time": signal["bar_time"], "entry_price": signal["price"],
        "sl": signal["sl"], "target1": signal["target1"], "target2": signal["target2"],
        "features": signal["features"], "outcome": "PENDING", "exit_price": None,
        "exit_time": None, "max_favorable_pct": 0.0, "max_adverse_pct": 0.0,
    })
    save_bear_alert_history(history)


def update_pending_short_outcomes(finalize_eod=False):
    """
    Mirror of the bullish version, INVERTED for a short: profit when price
    falls. hit_sl = price rose to/above SL. hit target = price fell to/below
    target. max_favorable_pct = how far price fell (in your favor).
    max_adverse_pct = how far price rose (against you).
    """
    history = load_bear_alert_history()
    today = today_str()
    changed = False

    for rec in history:
        if rec.get("date") != today or rec.get("outcome") != "PENDING":
            continue

        symbol = rec["symbol"]
        df = yf_intraday(symbol)
        if df is None or df.empty:
            continue

        try:
            idx = _tz_naive_index(df)
            df = df.copy()
            df.index = idx

            anchor = pd.to_datetime(rec["bar_time"])
            try:
                anchor = anchor.tz_localize(None)
            except TypeError:
                pass

            bars_since = df.loc[df.index > anchor]
            if bars_since.empty:
                continue

            entry, sl, t1, t2 = rec["entry_price"], rec["sl"], rec["target1"], rec["target2"]
            resolved = False

            for ts, bar in bars_since.iterrows():
                high, low = float(bar["High"]), float(bar["Low"])

                fav_pct = ((entry - low) / entry) * 100
                adv_pct = ((high - entry) / entry) * 100
                rec["max_favorable_pct"] = round(max(rec["max_favorable_pct"], fav_pct), 2)
                rec["max_adverse_pct"] = round(max(rec["max_adverse_pct"], adv_pct), 2)

                hit_sl = high >= sl
                hit_t2 = low <= t2
                hit_t1 = low <= t1

                if hit_sl or hit_t1 or hit_t2:
                    if hit_sl:
                        rec["outcome"], rec["exit_price"] = "STOPLOSS_HIT", sl
                    elif hit_t2:
                        rec["outcome"], rec["exit_price"] = "TARGET2_HIT", t2
                    else:
                        rec["outcome"], rec["exit_price"] = "TARGET1_HIT", t1
                    rec["exit_time"] = ts.isoformat()
                    resolved = True
                    changed = True
                    break

            if not resolved:
                changed = True
                if finalize_eod:
                    last_close = float(bars_since["Close"].iloc[-1])
                    rec["outcome"] = "NO_TARGET_EOD"
                    rec["exit_price"] = round(last_close, 2)
                    rec["exit_time"] = bars_since.index[-1].isoformat()

        except Exception as e:
            log.debug("Bearish outcome tracking failed for %s: %s", symbol, e)

    if changed:
        save_bear_alert_history(history)
    return history


def compute_intraday_atr(day_df, period=INTRADAY_ATR_PERIOD):
    """
    True Range computed on 5-min bars, not daily bars — this is the
    session-appropriate volatility measure for a trade that must close
    within the same day.
    """
    if day_df is None or len(day_df) < 3:
        return 0.0
    high = day_df["High"]
    low = day_df["Low"]
    prev_close = day_df["Close"].shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    window = tr.tail(period)
    return float(window.mean()) if len(window) > 0 else 0.0


def analyze_bearish_intraday(symbol, watch_item, model_weights=None):
    df = yf_intraday(symbol)
    if df is None or df.empty:
        return None

    today = now_ist().date()
    try:
        day = df[pd.to_datetime(df.index).date == today]
    except Exception:
        day = df

    if day is None or len(day) < 5:
        return None

    last = day.iloc[-1]
    price = float(last["Close"])

    lookback = min(12, len(day) - 1)
    intraday_support = float(day["Low"].iloc[-lookback - 1:-1].min()) if lookback >= 2 else float(day["Low"].iloc[:-1].min())

    bearish_candle = last["Close"] < last["Open"]
    avg_bar_volume = float(day["Volume"].iloc[-lookback - 1:-1].mean()) if lookback >= 2 else 0
    volume_ratio = float(last["Volume"]) / avg_bar_volume if avg_bar_volume > 0 else 0

    # Require BOTH a fresh local low (immediate weakness) AND a break of
    # the actual 20-day support level from the daily scan — a break of
    # only the last hour's minor low isn't a meaningful support break,
    # it's just noise. This is the tightening we discussed: "recent good
    # support" should mean something real, not an arbitrary short window.
    support_20d = watch_item["technical"].get("support_20d", intraday_support)
    support = min(intraday_support, support_20d)
    breakdown = price < intraday_support and price < support_20d

    # Same lesson as the long bot's extension guard, mirrored: don't short
    # something that has already crashed too far since it first qualified
    # for the watchlist this morning — that's chasing, just downward.
    first_seen_price = watch_item.get("first_seen_price", watch_item["base"]["price"])
    move_since_first_seen = (
        ((first_seen_price - price) / first_seen_price) * 100
        if first_seen_price > 0 else 0
    )
    too_extended = move_since_first_seen > SHORT_EXTENSION_CAP_PCT

    # No new entries too close to the mandatory square-off — not enough
    # runway left to reasonably expect a target to develop.
    n = now_ist()
    past_cutoff = (n.hour, n.minute) >= (SHORT_ENTRY_CUTOFF_HOUR, SHORT_ENTRY_CUTOFF_MINUTE)

    score = 0
    if breakdown:
        score += 40
    if volume_ratio >= BEAR_MIN_INTRADAY_VOLUME_RATIO:
        score += 30
    elif volume_ratio >= 1.3:
        score += 15
    if bearish_candle:
        score += 15
    daily_score = watch_item["technical"]["score"]
    score += min(15, daily_score / 10)
    score = round(min(100, score), 1)

    # --- Learned adjustment (self-learning feedback loop) ---
    features = build_bear_feature_dict(watch_item, score, volume_ratio, move_since_first_seen)
    win_prob, blend_ratio = learned_adjustment(features, model_weights)
    if win_prob is not None and blend_ratio > 0:
        learned_score = win_prob * 100
        score = round(min(100, (1 - blend_ratio) * score + blend_ratio * learned_score), 1)

    short_signal = (
        breakdown
        and volume_ratio >= BEAR_MIN_INTRADAY_VOLUME_RATIO
        and bearish_candle
        and score >= BEAR_MIN_WARN_SCORE
        and not too_extended
        and not past_cutoff
    )

    if breakdown and volume_ratio >= BEAR_MIN_INTRADAY_VOLUME_RATIO and too_extended:
        log.info(
            "%s: breakdown+volume met but skipped, already -%.1f%% since first seen (cap %.1f%%)",
            symbol, move_since_first_seen, SHORT_EXTENSION_CAP_PCT
        )
    if breakdown and volume_ratio >= BEAR_MIN_INTRADAY_VOLUME_RATIO and past_cutoff and not too_extended:
        log.info(
            "%s: breakdown+volume met but skipped, past %02d:%02d entry cutoff",
            symbol, SHORT_ENTRY_CUTOFF_HOUR, SHORT_ENTRY_CUTOFF_MINUTE
        )

    intraday_atr = compute_intraday_atr(day)
    sl = price + SHORT_SL_ATR_MULT * intraday_atr if intraday_atr > 0 else price * 1.015
    target1 = price - SHORT_T1_ATR_MULT * intraday_atr if intraday_atr > 0 else price * 0.98
    target2 = price - SHORT_T2_ATR_MULT * intraday_atr if intraday_atr > 0 else price * 0.965

    return {
        "symbol": symbol, "price": round(price, 2), "support": round(support, 2),
        "volume_ratio": round(volume_ratio, 2), "score": score,
        "short_signal": short_signal,
        "sl": round(sl, 2), "target1": round(target1, 2), "target2": round(target2, 2),
        "time": now_ist().strftime("%H:%M:%S"),
        "bar_time": pd.to_datetime(day.index[-1]).isoformat(),
        "features": features,
    }


# ============================================================
# COMMANDS
# ============================================================

def cmd_scan():
    if now_ist().weekday() >= 5:
        log.info("Weekend — skipping bearish scan.")
        return

    log.info("Starting bearish breakdown scan.")
    universe = get_all_nse_stocks()
    if not universe:
        tg_send("⚠️ Bearish radar: could not load NSE universe.")
        return

    results = scan_bearish_universe(universe)
    watchlist_items = results[:BEAR_WATCHLIST_MAX_SIZE]
    for item in watchlist_items:
        item["first_seen_price"] = item["base"]["price"]
        item["first_seen_at"] = item["added_at"]
    save_bearish_watchlist(watchlist_items)
    save_bearish_alert_state({"date": today_str(), "alerted": []})

    if watchlist_items:
        tg_long_send(format_bearish_report(results))
    else:
        tg_send("🔻 *Short Watchlist*\nNo weak candidates found today.")

    log.info("Bearish scan complete. Watchlist size: %s", len(watchlist_items))


def cmd_rescan():
    if not market_is_open():
        log.info("Market closed — skipping bearish universe rescan.")
        return

    log.info("Starting periodic bearish universe rescan for new candidates.")
    watchlist_data = load_bearish_watchlist()
    existing_items = watchlist_data.get("items", [])

    universe = get_all_nse_stocks()
    if not universe:
        log.warning("Could not load universe for bearish rescan.")
        return

    results = scan_bearish_universe(universe)
    merged, added = merge_bearish_watchlist(existing_items, results)
    save_bearish_watchlist(merged)

    if added:
        top_new = sorted(added, key=lambda x: x["combined_score"], reverse=True)[:10]
        n = now_ist()
        msg = (
            f"🔻 *NEW WEAK CANDIDATES*\n"
            f"⏰ {n.strftime('%H:%M:%S')}\n"
            f"🆕 {len(added)} new stocks showing weakness.\n\n"
        )
        for i, item in enumerate(top_new, 1):
            t = item["technical"]
            msg += (
                f"{i}. *{item['symbol']}* Score {item['combined_score']}\n"
                f"   Setup: {t['setup']} | Vol {item['base']['volume_ratio']}x\n"
            )
        tg_send(msg)

    log.info("Bearish rescan complete. Watchlist size: %s (+%s new)", len(merged), len(added))


def cmd_recheck():
    if not market_is_open():
        log.info("Market closed — skipping bearish recheck.")
        return

    try:
        update_pending_short_outcomes()
    except Exception as e:
        log.warning("update_pending_short_outcomes failed: %s", e)

    model_weights = load_bear_model_weights()
    if model_weights:
        log.info("Using learned bearish model (n_samples=%s).", model_weights.get("n_samples"))

    watchlist_data = load_bearish_watchlist()
    if watchlist_data.get("date") != today_str():
        log.info("Bearish watchlist stale — skipping.")
        return

    items = watchlist_data.get("items", [])
    if not items:
        log.info("Bearish watchlist empty.")
        return

    alert_state = load_bearish_alert_state()
    alerted = set(alert_state.get("alerted", []))

    new_alerts = []
    for item in items:
        symbol = item["symbol"]
        if symbol in alerted:
            continue
        try:
            signal = analyze_bearish_intraday(symbol, item, model_weights=model_weights)
        except Exception as e:
            log.debug("Bearish intraday error %s: %s", symbol, e)
            continue

        if signal and signal["short_signal"]:
            tg_send(build_short_signal_message(signal, item))
            record_new_short_alert(signal, item)
            alerted.add(symbol)
            new_alerts.append(symbol)

    if new_alerts:
        alert_state["alerted"] = sorted(alerted)
        save_bearish_alert_state(alert_state)
        log.info("Sent short signals for: %s", ", ".join(new_alerts))
    else:
        log.info("No new short signals this recheck.")


def cmd_squareoff_reminder():
    """
    Mandatory reminder — cash-segment shorts CANNOT be carried overnight.
    Runs once daily shortly before the square-off deadline and lists any
    of today's active short signals.
    """
    if now_ist().weekday() >= 5:
        log.info("Weekend — skipping square-off reminder.")
        return

    alert_state = load_bearish_alert_state()
    if alert_state.get("date") != today_str():
        log.info("No short signals recorded today — nothing to remind.")
        return

    alerted = alert_state.get("alerted", [])
    if not alerted:
        log.info("No active short signals today — no reminder needed.")
        return

    msg = (
        f"⏰ *MANDATORY SQUARE-OFF REMINDER*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"You have {len(alerted)} intraday short signal(s) today that "
        f"MUST be closed by {MANDATORY_SQUAREOFF_TIME_TEXT}:\n\n"
        + "\n".join(f"• {s}" for s in alerted)
        + "\n\nCash-segment shorts cannot be carried overnight regardless "
          "of whether SL/target was hit."
    )
    tg_send(msg)
    log.info("Sent square-off reminder for: %s", ", ".join(alerted))


def cmd_eod_finalize():
    if now_ist().weekday() >= 5:
        log.info("Weekend — skipping bearish EOD finalize.")
        return

    log.info("Finalizing today's pending short-signal outcomes.")
    history = update_pending_short_outcomes(finalize_eod=True)

    today = today_str()
    today_records = [r for r in history if r.get("date") == today]
    if not today_records:
        log.info("No short signals recorded today.")
        return

    counts = {}
    for r in today_records:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1

    wins = counts.get("TARGET1_HIT", 0) + counts.get("TARGET2_HIT", 0)
    losses = counts.get("STOPLOSS_HIT", 0)
    no_target = counts.get("NO_TARGET_EOD", 0)
    total = len(today_records)
    avg_mfe = sum(r["max_favorable_pct"] for r in today_records) / total
    avg_mae = sum(r["max_adverse_pct"] for r in today_records) / total

    msg = (
        f"📊 *Short Signal EOD Report — {today}*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Total signals: {total}\n"
        f"🎯 Target hit: {wins}\n"
        f"🛑 Stop-loss hit: {losses}\n"
        f"➖ No target/SL hit: {no_target}\n\n"
        f"📉 Avg best move (in your favor): -{avg_mfe:.2f}%\n"
        f"📈 Avg worst move (against you): +{avg_mae:.2f}%\n\n"
        f"This data feeds the weekly bearish model retrain."
    )
    tg_send(msg)
    log.info("Bearish EOD finalize complete: %s", counts)


def cmd_retrain():
    log.info("Starting weekly retrain of bearish scoring weights.")
    history = load_bear_alert_history()

    resolved = [
        r for r in history
        if r.get("outcome") in ("TARGET1_HIT", "TARGET2_HIT", "STOPLOSS_HIT", "NO_TARGET_EOD")
        and r.get("features")
    ]

    if RETRAIN_MIN_DATE:
        before = len(resolved)
        resolved = [r for r in resolved if r.get("date", "") >= RETRAIN_MIN_DATE]
        if before - len(resolved):
            log.info("Excluded %s pre-%s short alerts from training.", before - len(resolved), RETRAIN_MIN_DATE)

    n = len(resolved)
    if n < MIN_SAMPLES_FOR_LEARNING:
        tg_send(
            f"🧠 *Bearish Self-Learning Retrain*\n"
            f"Only {n} resolved short signals so far (need {MIN_SAMPLES_FOR_LEARNING} minimum).\n"
            f"Still using the fixed heuristic scoring until enough data accumulates."
        )
        log.info("Not enough bearish samples yet: %s/%s", n, MIN_SAMPLES_FOR_LEARNING)
        return

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        tg_send("⚠️ Bearish retrain failed: scikit-learn is missing from requirements.txt.")
        return

    X, y = [], []
    for r in resolved:
        X.append([r["features"].get(name, 0.0) for name in BEAR_FEATURE_NAMES])
        y.append(1 if r["outcome"] in ("TARGET1_HIT", "TARGET2_HIT") else 0)

    X = pd.DataFrame(X, columns=BEAR_FEATURE_NAMES).fillna(0.0).values
    y = pd.Series(y).values

    if len(set(y)) < 2:
        tg_send(
            f"🧠 *Bearish Self-Learning Retrain*\n"
            f"All {n} resolved signals have the same outcome so far — "
            "can't fit a model until there's a mix of wins and losses."
        )
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_scaled, y)

    train_accuracy = model.score(X_scaled, y)
    win_rate = sum(y) / len(y) * 100

    weights = {
        "trained_at": now_ist().isoformat(), "n_samples": n,
        "features": BEAR_FEATURE_NAMES, "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(), "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]), "train_accuracy": round(train_accuracy, 3),
    }
    save_json(BEAR_MODEL_WEIGHTS_FILE, weights)

    ranked = sorted(zip(BEAR_FEATURE_NAMES, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True)[:5]
    top_features_text = "\n".join(f"  {'+' if c > 0 else '-'} {name}" for name, c in ranked)
    blend_pct = min(100, max(0, (n - MIN_SAMPLES_FOR_LEARNING) / max(1, (LEARNING_FULL_INFLUENCE_SAMPLES - MIN_SAMPLES_FOR_LEARNING)) * 100))

    tg_send(
        f"🧠 *Bearish Self-Learning Retrain Complete*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Samples used: {n}\n"
        f"Historical win rate: {win_rate:.1f}%\n"
        f"Model fit on training data: {train_accuracy*100:.1f}%\n"
        f"Blend influence: {blend_pct:.0f}% (reaches 100% at {LEARNING_FULL_INFLUENCE_SAMPLES} samples)\n\n"
        f"Top influential factors:\n{top_features_text}"
    )
    log.info("Bearish retrain complete. n=%s, train_accuracy=%.3f", n, train_accuracy)


def main():
    print("=" * 72)
    print("NSE BEARISH BREAKDOWN RADAR")
    print("=" * 72)

    command = sys.argv[1] if len(sys.argv) > 1 else "scan"
    if command == "scan":
        cmd_scan()
    elif command == "rescan":
        cmd_rescan()
    elif command == "recheck":
        cmd_recheck()
    elif command == "squareoff_reminder":
        cmd_squareoff_reminder()
    elif command == "eod_finalize":
        cmd_eod_finalize()
    elif command == "retrain":
        cmd_retrain()
    else:
        print(
            f"Unknown command '{command}'. Use 'scan', 'rescan', 'recheck', "
            f"'squareoff_reminder', 'eod_finalize', or 'retrain'."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
