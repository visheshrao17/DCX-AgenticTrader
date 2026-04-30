"""
DCX-AgenticTrader — Backtest Performance Metrics

Calculates: total return, max drawdown, Sharpe ratio, win rate,
and Buy-&-Hold comparison from equity curve data.
"""

import math
from typing import Dict, Any, List


def calculate_metrics(
    equity_curve: List[Dict],
    initial_capital: float,
    buy_hold_final: float,
    risk_free_rate: float = 0.065,  # India 10Y bond ~6.5%
) -> Dict[str, Any]:
    """
    Calculate performance metrics from an equity curve.

    Args:
        equity_curve: List of {step, portfolio_value, ...} dicts.
        initial_capital: Starting capital in INR.
        buy_hold_final: Final value of Buy-&-Hold strategy.
        risk_free_rate: Annual risk-free rate for Sharpe calculation.

    Returns:
        Dict with all performance metrics.
    """
    if not equity_curve:
        return _empty_metrics()

    values = [e["portfolio_value"] for e in equity_curve]
    final_value = values[-1]

    # Total return
    total_return = (final_value - initial_capital) / initial_capital * 100
    bh_return = (buy_hold_final - initial_capital) / initial_capital * 100

    # Max drawdown
    peak = values[0]
    max_dd = 0.0
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # Returns series for Sharpe
    returns = []
    for i in range(1, len(values)):
        if values[i - 1] > 0:
            r = (values[i] - values[i - 1]) / values[i - 1]
            returns.append(r)

    # Sharpe Ratio (annualized)
    sharpe = 0.0
    if returns:
        avg_return = sum(returns) / len(returns)
        std_return = math.sqrt(sum((r - avg_return) ** 2 for r in returns) / len(returns)) if len(returns) > 1 else 1
        if std_return > 0:
            # Assume ~252 trading days equivalent
            periods_per_year = 252
            excess_return = avg_return - (risk_free_rate / periods_per_year)
            sharpe = (excess_return / std_return) * math.sqrt(periods_per_year)

    # Win rate from trades in equity curve
    positive_moves = sum(1 for r in returns if r > 0)
    win_rate = (positive_moves / len(returns) * 100) if returns else 0

    return {
        "total_return_pct": round(total_return, 2),
        "buy_hold_return_pct": round(bh_return, 2),
        "alpha_pct": round(total_return - bh_return, 2),
        "final_value": round(final_value, 2),
        "buy_hold_final": round(buy_hold_final, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 3),
        "win_rate_pct": round(win_rate, 1),
        "total_periods": len(values),
        "positive_periods": positive_moves,
        "negative_periods": len(returns) - positive_moves,
    }


def _empty_metrics() -> Dict[str, Any]:
    return {
        "total_return_pct": 0, "buy_hold_return_pct": 0, "alpha_pct": 0,
        "final_value": 0, "buy_hold_final": 0, "max_drawdown_pct": 0,
        "sharpe_ratio": 0, "win_rate_pct": 0, "total_periods": 0,
        "positive_periods": 0, "negative_periods": 0,
    }
