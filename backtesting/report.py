"""
DCX-AgenticTrader — Backtest Report Generator

Generates visual HTML/Plotly reports from backtest results.
"""

from typing import Dict, Any, List

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report(results: Dict[str, Any]) -> go.Figure:
    """
    Generate a comprehensive Plotly figure with backtest results.

    Args:
        results: Output from BacktestEngine.run()

    Returns:
        Plotly Figure with equity curve, drawdown, and trade markers.
    """
    equity = results.get("equity_curve", [])
    bh = results.get("buy_hold_curve", [])
    metrics = results.get("metrics", {})

    if not equity:
        fig = go.Figure()
        fig.add_annotation(text="No backtest data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("Equity Curve vs Buy & Hold", "Price", "Drawdown"),
    )

    steps = [e["step"] for e in equity]
    values = [e["portfolio_value"] for e in equity]
    prices = [e["price"] for e in equity]
    bh_values = [b["value"] for b in bh] if bh else []

    # Equity curve
    fig.add_trace(go.Scatter(
        x=steps, y=values, mode="lines",
        name="Strategy", line=dict(color="#00c853", width=2),
        fill="tozeroy", fillcolor="rgba(0, 200, 83, 0.05)",
    ), row=1, col=1)

    if bh_values:
        fig.add_trace(go.Scatter(
            x=steps, y=bh_values, mode="lines",
            name="Buy & Hold", line=dict(color="#ffa726", width=1.5, dash="dash"),
        ), row=1, col=1)

    # Price chart
    fig.add_trace(go.Scatter(
        x=steps, y=prices, mode="lines",
        name="Price", line=dict(color="#42a5f5", width=1.5),
    ), row=2, col=1)

    # Drawdown
    peak = values[0]
    drawdowns = []
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100
        drawdowns.append(-dd)

    fig.add_trace(go.Scatter(
        x=steps, y=drawdowns, mode="lines",
        name="Drawdown", line=dict(color="#ff1744", width=1.5),
        fill="tozeroy", fillcolor="rgba(255, 23, 68, 0.1)",
    ), row=3, col=1)

    # Layout
    title = (
        f"Backtest: {results.get('market', '')} | "
        f"Return: {metrics.get('total_return_pct', 0):+.1f}% vs "
        f"B&H: {metrics.get('buy_hold_return_pct', 0):+.1f}% | "
        f"Drawdown: {metrics.get('max_drawdown_pct', 0):.1f}% | "
        f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}"
    )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=800,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#ccc"),
        showlegend=True,
        legend=dict(x=0, y=1.12, orientation="h"),
    )

    fig.update_yaxes(title_text="Value (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Price (₹)", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)

    return fig


def print_report_summary(results: Dict[str, Any]) -> str:
    """Generate a text summary of backtest results."""
    m = results.get("metrics", {})
    return (
        f"\n{'='*60}\n"
        f"  BACKTEST REPORT — {results.get('market', '')}\n"
        f"{'='*60}\n"
        f"  Period:         {results.get('months', 0)} months ({results.get('total_candles', 0)} candles)\n"
        f"  Interval:       {results.get('interval', '')}\n"
        f"  Start Price:    ₹{results.get('start_price', 0):,.2f}\n"
        f"  End Price:      ₹{results.get('end_price', 0):,.2f}\n"
        f"{'─'*60}\n"
        f"  Strategy Return: {m.get('total_return_pct', 0):+.2f}%\n"
        f"  Buy & Hold:      {m.get('buy_hold_return_pct', 0):+.2f}%\n"
        f"  Alpha:           {m.get('alpha_pct', 0):+.2f}%\n"
        f"  Max Drawdown:    {m.get('max_drawdown_pct', 0):.2f}%\n"
        f"  Sharpe Ratio:    {m.get('sharpe_ratio', 0):.3f}\n"
        f"  Total Trades:    {results.get('total_trades', 0)}\n"
        f"  Win Rate:        {m.get('win_rate_pct', 0):.1f}%\n"
        f"{'='*60}\n"
    )
