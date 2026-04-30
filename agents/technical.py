"""
DCX-AgenticTrader — Technical Analyst Agent

Runs all technical indicators on market data and produces
a composite trading signal with confidence score.
"""

import json
from typing import Dict, Any

import pandas as pd

from tools.technical_indicators import generate_signal_summary
from agents.market_data import candles_to_dataframe
from graph.state import TradingState
from utils.logger import get_agent_logger

log = get_agent_logger("technical")


def technical_agent(state: TradingState) -> Dict[str, Any]:
    """
    LangGraph node: Technical Analyst Agent.

    Reads market_data from state, runs all TA indicators,
    and writes technical_signals back to state.
    """
    log.info("=== Technical Analyst Agent running ===")

    market_data = state.get("market_data")
    if not market_data or "error" in market_data:
        log.warning("No market data available — returning neutral signal")
        return {
            "technical_signals": {
                "composite_signal": "NEUTRAL",
                "signal_score": 0.0,
                "confidence": 0.0,
                "reasoning": "No market data available for analysis.",
                "error": "Missing market data",
            },
            "current_step": "technical",
        }

    pair = market_data.get("pair", "UNKNOWN")
    log.info(f"Analyzing {pair}")

    # Use 1h candles as primary timeframe for signal generation
    candles_1h = market_data.get("candles_1h", [])
    candles_15m = market_data.get("candles_15m", [])
    candles_4h = market_data.get("candles_4h", [])

    # Multi-timeframe analysis
    signals_by_tf = {}

    for tf_name, candles in [("15m", candles_15m), ("1h", candles_1h), ("4h", candles_4h)]:
        if not candles or len(candles) < 30:
            log.warning(f"Insufficient {tf_name} candle data ({len(candles) if candles else 0} candles)")
            continue

        df = candles_to_dataframe(candles)
        if df.empty or len(df) < 30:
            continue

        signal = generate_signal_summary(df)
        signals_by_tf[tf_name] = signal
        log.info(f"  {tf_name}: {signal.get('composite_signal', 'N/A')} (score={signal.get('signal_score', 0):.3f})")

    if not signals_by_tf:
        log.warning("No valid timeframe data — returning neutral")
        return {
            "technical_signals": {
                "pair": pair,
                "composite_signal": "NEUTRAL",
                "signal_score": 0.0,
                "confidence": 0.0,
                "reasoning": "Insufficient candle data across all timeframes.",
            },
            "current_step": "technical",
        }

    # Weight timeframes: 1h = primary, 4h = confirmation, 15m = timing
    weights = {"15m": 0.2, "1h": 0.5, "4h": 0.3}
    total_weight = 0.0
    weighted_score = 0.0

    for tf, signal in signals_by_tf.items():
        w = weights.get(tf, 0.3)
        weighted_score += signal.get("signal_score", 0) * w
        total_weight += w

    final_score = weighted_score / total_weight if total_weight else 0.0

    # Use the 1h signal as the base, override score with multi-TF weighted
    primary = signals_by_tf.get("1h", signals_by_tf.get("15m", {}))

    # Determine composite signal from weighted score
    if final_score >= 0.7:
        composite = "STRONG_BUY"
    elif final_score >= 0.3:
        composite = "BUY"
    elif final_score <= -0.7:
        composite = "STRONG_SELL"
    elif final_score <= -0.3:
        composite = "SELL"
    else:
        composite = "NEUTRAL"

    # Build reasoning
    tf_summary = ", ".join(
        f"{tf}={s.get('composite_signal', 'N/A')}({s.get('signal_score', 0):.2f})"
        for tf, s in signals_by_tf.items()
    )

    reasoning = (
        f"Multi-timeframe analysis for {pair}: {tf_summary}. "
        f"Weighted score: {final_score:.3f} → {composite}. "
        f"RSI(1h)={primary.get('rsi', 'N/A')}, "
        f"MACD hist={primary.get('macd_histogram', 'N/A')}, "
        f"Price vs BB: mid={primary.get('bb_middle', 'N/A')}."
    )

    result = {
        "pair": pair,
        **primary,
        "composite_signal": composite,
        "signal_score": round(final_score, 3),
        "confidence": round(min(abs(final_score) * 1.2, 1.0), 2),
        "reasoning": reasoning,
        "timeframe_signals": {
            tf: {"signal": s.get("composite_signal"), "score": s.get("signal_score")}
            for tf, s in signals_by_tf.items()
        },
    }

    log.info(f"Final signal: {composite} (score={final_score:.3f}, conf={result['confidence']})")

    return {
        "technical_signals": result,
        "current_step": "technical",
    }
