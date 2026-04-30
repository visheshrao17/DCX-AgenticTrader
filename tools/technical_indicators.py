"""
DCX-AgenticTrader — Technical Indicators (Pure Pandas/NumPy)

All indicators implemented from scratch — no pandas-ta/numba dependency.
Works on Python 3.10-3.14+.
"""

from typing import Dict, Any, List

import pandas as pd
import numpy as np
from langchain_core.tools import tool

from config.constants import (
    TA_RSI_PERIOD, TA_MACD_FAST, TA_MACD_SLOW, TA_MACD_SIGNAL,
    TA_BOLLINGER_PERIOD, TA_BOLLINGER_STD, TA_EMA_PERIODS, TA_ATR_PERIOD,
    SIGNAL_STRONG_BUY_THRESHOLD, SIGNAL_BUY_THRESHOLD,
    SIGNAL_SELL_THRESHOLD, SIGNAL_STRONG_SELL_THRESHOLD,
)
from utils.logger import get_agent_logger

log = get_agent_logger("tech_indicators")


# =============================================================================
# Core Indicator Functions (pure pandas/numpy)
# =============================================================================

def compute_rsi(df: pd.DataFrame, period: int = TA_RSI_PERIOD) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Compute Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def compute_macd(
    df: pd.DataFrame,
    fast: int = TA_MACD_FAST,
    slow: int = TA_MACD_SLOW,
    signal: int = TA_MACD_SIGNAL,
) -> Dict[str, pd.Series]:
    """Compute MACD line, signal line, and histogram."""
    ema_fast = compute_ema(df["close"], fast)
    ema_slow = compute_ema(df["close"], slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return {"macd_line": macd_line, "macd_signal": signal_line, "macd_histogram": histogram}


def compute_bollinger_bands(
    df: pd.DataFrame,
    period: int = TA_BOLLINGER_PERIOD,
    std: float = TA_BOLLINGER_STD,
) -> Dict[str, pd.Series]:
    """Compute Bollinger Bands (upper, middle, lower)."""
    middle = df["close"].rolling(window=period).mean()
    rolling_std = df["close"].rolling(window=period).std()
    return {
        "bb_upper": middle + (rolling_std * std),
        "bb_middle": middle,
        "bb_lower": middle - (rolling_std * std),
    }


def compute_emas(df: pd.DataFrame, periods: List[int] = None) -> Dict[int, pd.Series]:
    """Compute EMAs for multiple periods."""
    if periods is None:
        periods = TA_EMA_PERIODS
    return {p: compute_ema(df["close"], p) for p in periods}


def compute_atr(df: pd.DataFrame, period: int = TA_ATR_PERIOD) -> pd.Series:
    """Compute Average True Range (volatility)."""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Compute Volume Weighted Average Price."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cumulative_tp_vol = (typical_price * df["volume"]).cumsum()
    cumulative_vol = df["volume"].cumsum()
    return cumulative_tp_vol / cumulative_vol.replace(0, np.nan)


def find_support_resistance(df: pd.DataFrame, window: int = 20) -> Dict[str, List[float]]:
    """Find key support and resistance levels using rolling min/max."""
    supports, resistances = [], []
    rolling_min = df["low"].rolling(window=window).min()
    rolling_max = df["high"].rolling(window=window).max()

    recent = df.tail(50)
    for i in range(len(recent)):
        idx = len(df) - 50 + i
        if idx < 0 or idx >= len(rolling_min):
            continue
        low = recent.iloc[i]["low"]
        high = recent.iloc[i]["high"]
        if not pd.isna(rolling_min.iloc[idx]) and low <= rolling_min.iloc[idx] * 1.001:
            supports.append(round(low, 2))
        if not pd.isna(rolling_max.iloc[idx]) and high >= rolling_max.iloc[idx] * 0.999:
            resistances.append(round(high, 2))

    supports = _deduplicate_levels(sorted(set(supports)))
    resistances = _deduplicate_levels(sorted(set(resistances), reverse=True))
    return {"support": supports[:3], "resistance": resistances[:3]}


def _deduplicate_levels(levels: List[float], threshold_pct: float = 0.5) -> List[float]:
    if not levels:
        return []
    deduped = [levels[0]]
    for level in levels[1:]:
        if deduped[-1] != 0 and abs(level - deduped[-1]) / abs(deduped[-1]) * 100 > threshold_pct:
            deduped.append(level)
    return deduped


def _safe_last(series: pd.Series) -> float:
    """Safely get last non-NaN value from a series."""
    if series is None or series.empty:
        return 0.0
    val = series.iloc[-1]
    return float(val) if not pd.isna(val) else 0.0


# =============================================================================
# Composite Signal Generator
# =============================================================================

def generate_signal_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Run all indicators and generate a composite trading signal."""
    if df is None or df.empty or len(df) < 30:
        return {
            "error": "Insufficient data (need at least 30 candles)",
            "composite_signal": "NEUTRAL",
            "signal_score": 0.0,
            "confidence": 0.0,
        }

    signals = []  # (signal_value, weight)
    result: Dict[str, Any] = {}

    # --- RSI ---
    rsi = compute_rsi(df)
    rsi_val = _safe_last(rsi) or 50.0
    result["rsi"] = round(rsi_val, 2)
    if rsi_val < 30:
        signals.append((1.0, 1.5))
    elif rsi_val < 40:
        signals.append((0.5, 1.0))
    elif rsi_val > 70:
        signals.append((-1.0, 1.5))
    elif rsi_val > 60:
        signals.append((-0.5, 1.0))
    else:
        signals.append((0.0, 0.5))

    # --- MACD ---
    macd = compute_macd(df)
    macd_line = _safe_last(macd["macd_line"])
    macd_sig = _safe_last(macd["macd_signal"])
    macd_hist = _safe_last(macd["macd_histogram"])
    result["macd_line"] = round(macd_line, 4)
    result["macd_signal"] = round(macd_sig, 4)
    result["macd_histogram"] = round(macd_hist, 4)

    if macd_line > macd_sig and macd_hist > 0:
        signals.append((1.0, 1.5))
    elif macd_line < macd_sig and macd_hist < 0:
        signals.append((-1.0, 1.5))
    else:
        signals.append((0.0, 0.5))

    # --- Bollinger Bands ---
    bb = compute_bollinger_bands(df)
    bb_lower = _safe_last(bb["bb_lower"])
    bb_mid = _safe_last(bb["bb_middle"])
    bb_upper = _safe_last(bb["bb_upper"])
    result["bb_upper"] = round(bb_upper, 2)
    result["bb_middle"] = round(bb_mid, 2)
    result["bb_lower"] = round(bb_lower, 2)

    current_price = float(df["close"].iloc[-1])
    if bb_lower > 0 and current_price <= bb_lower:
        signals.append((1.0, 1.0))
    elif bb_upper > 0 and current_price >= bb_upper:
        signals.append((-1.0, 1.0))
    else:
        signals.append((0.0, 0.3))

    # --- EMAs ---
    emas = compute_emas(df)
    for period, series in emas.items():
        val = _safe_last(series)
        result[f"ema_{period}"] = round(val, 2)

    if result.get("ema_9", 0) and result.get("ema_21", 0):
        if result["ema_9"] > result["ema_21"]:
            signals.append((0.5, 1.0))
        else:
            signals.append((-0.5, 1.0))

    if result.get("ema_50", 0) and result.get("ema_200", 0):
        if result["ema_50"] > result["ema_200"]:
            signals.append((0.3, 0.8))
        else:
            signals.append((-0.3, 0.8))

    # --- ATR ---
    atr = compute_atr(df)
    result["atr"] = round(_safe_last(atr), 2)

    # --- VWAP ---
    vwap = compute_vwap(df)
    result["vwap"] = round(_safe_last(vwap), 2)

    # --- Support/Resistance ---
    sr = find_support_resistance(df)
    result["support_levels"] = sr["support"]
    result["resistance_levels"] = sr["resistance"]

    # --- Composite Score ---
    total_weight = sum(w for _, w in signals)
    weighted_score = sum(s * w for s, w in signals) / total_weight if total_weight else 0
    result["signal_score"] = round(weighted_score, 3)

    if weighted_score >= SIGNAL_STRONG_BUY_THRESHOLD:
        result["composite_signal"] = "STRONG_BUY"
    elif weighted_score >= SIGNAL_BUY_THRESHOLD:
        result["composite_signal"] = "BUY"
    elif weighted_score <= SIGNAL_STRONG_SELL_THRESHOLD:
        result["composite_signal"] = "STRONG_SELL"
    elif weighted_score <= SIGNAL_SELL_THRESHOLD:
        result["composite_signal"] = "SELL"
    else:
        result["composite_signal"] = "NEUTRAL"

    result["confidence"] = round(min(abs(weighted_score) * 1.2, 1.0), 2)

    log.info(
        f"Signal: {result['composite_signal']} (score={result['signal_score']}, "
        f"conf={result['confidence']}, RSI={result['rsi']})"
    )
    return result


# =============================================================================
# LangGraph Tool
# =============================================================================

@tool
def run_technical_analysis(candles_json: str) -> dict:
    """
    Run full technical analysis on candle data.

    Args:
        candles_json: JSON string of candle data with open/high/low/close/volume fields.

    Returns:
        Dict with all indicator values and composite signal.
    """
    import json
    try:
        candles = json.loads(candles_json)
        df = pd.DataFrame(candles)
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return generate_signal_summary(df)
    except Exception as e:
        return {"error": str(e), "composite_signal": "NEUTRAL", "signal_score": 0.0}


TECHNICAL_TOOLS = [run_technical_analysis]
