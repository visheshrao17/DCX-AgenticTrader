"""
DCX-AgenticTrader — Streamlit Dashboard Components

Reusable UI widgets for the trading dashboard:
price cards, PnL charts, agent status indicators, trade cards.
"""

import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from typing import Dict, Any, List, Optional


def render_price_card(market: str, price: float, change_24h: float = 0, high: float = 0, low: float = 0, volume: float = 0):
    """Render a price overview card."""
    color = "#00c853" if change_24h >= 0 else "#ff1744"
    arrow = "▲" if change_24h >= 0 else "▼"

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px; padding: 20px; border: 1px solid #2a2a4a;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    ">
        <div style="font-size: 14px; color: #888; margin-bottom: 4px;">{market}</div>
        <div style="font-size: 32px; font-weight: 700; color: #fff;">₹{price:,.2f}</div>
        <div style="font-size: 16px; color: {color}; margin-top: 4px;">
            {arrow} {abs(change_24h):.2f}%
        </div>
        <div style="display: flex; gap: 20px; margin-top: 12px; font-size: 12px; color: #888;">
            <span>H: ₹{high:,.2f}</span>
            <span>L: ₹{low:,.2f}</span>
            <span>Vol: {volume:,.0f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_signal_badge(signal: str, score: float = 0, confidence: float = 0):
    """Render a trading signal badge."""
    colors = {
        "STRONG_BUY": ("#00c853", "🟢"),
        "BUY": ("#66bb6a", "🟢"),
        "NEUTRAL": ("#ffa726", "🟡"),
        "SELL": ("#ef5350", "🔴"),
        "STRONG_SELL": ("#d32f2f", "🔴"),
        "HOLD": ("#ffa726", "🟡"),
    }
    color, emoji = colors.get(signal, ("#888", "⚪"))

    st.markdown(f"""
    <div style="
        background: {color}22; border: 1px solid {color};
        border-radius: 12px; padding: 12px 20px; display: inline-block;
        text-align: center;
    ">
        <div style="font-size: 24px;">{emoji}</div>
        <div style="font-size: 18px; font-weight: 700; color: {color};">{signal}</div>
        <div style="font-size: 12px; color: #aaa;">
            Score: {score:+.2f} | Conf: {confidence:.0%}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_portfolio_summary(portfolio: Dict[str, Any]):
    """Render portfolio summary metrics."""
    total = portfolio.get("total_value_inr", 0)
    pnl = portfolio.get("realized_pnl", 0)
    unrealized = portfolio.get("unrealized_pnl", 0)
    drawdown = portfolio.get("max_drawdown_pct", 0)

    cols = st.columns(4)
    with cols[0]:
        st.metric("Portfolio Value", f"₹{total:,.2f}")
    with cols[1]:
        st.metric("Realized PnL", f"₹{pnl:,.2f}", delta=f"₹{pnl:,.2f}")
    with cols[2]:
        st.metric("Unrealized PnL", f"₹{unrealized:,.2f}")
    with cols[3]:
        st.metric("Max Drawdown", f"{drawdown:.1f}%", delta=f"-{drawdown:.1f}%", delta_color="inverse")


def render_candlestick_chart(candles: List[Dict], title: str = "Price Chart"):
    """Render an interactive Plotly candlestick chart."""
    if not candles:
        st.info("No candle data available")
        return

    import pandas as pd
    df = pd.DataFrame(candles)

    # Normalize column names
    col_map = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "T": "time"}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    fig = go.Figure(data=[go.Candlestick(
        x=df.index if "time" not in df.columns else df["time"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
    )])

    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        xaxis_rangeslider_visible=False,
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(color="#ccc"),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_pnl_chart(trade_history: List[Dict]):
    """Render cumulative PnL chart from trade history."""
    if not trade_history:
        st.info("No trade history yet")
        return

    import pandas as pd
    df = pd.DataFrame(trade_history)

    if "realized_pnl" not in df.columns:
        st.info("No PnL data available")
        return

    df = df.sort_values("timestamp")
    df["cumulative_pnl"] = df["realized_pnl"].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["cumulative_pnl"],
        mode="lines+markers",
        line=dict(color="#00c853", width=2),
        fill="tozeroy",
        fillcolor="rgba(0, 200, 83, 0.1)",
        name="Cumulative PnL",
    ))

    fig.update_layout(
        title="Cumulative PnL",
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis_title="PnL (₹)",
        font=dict(color="#ccc"),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_trade_table(trades: List[Dict]):
    """Render trade history as a styled table."""
    if not trades:
        st.info("No trades yet")
        return

    import pandas as pd
    df = pd.DataFrame(trades)

    display_cols = ["timestamp", "market", "side", "quantity", "price", "fee", "realized_pnl", "status"]
    available_cols = [c for c in display_cols if c in df.columns]

    styled = df[available_cols].style.applymap(
        lambda v: "color: #00c853" if isinstance(v, str) and v == "buy" else
                  "color: #ff1744" if isinstance(v, str) and v == "sell" else "",
        subset=["side"] if "side" in available_cols else [],
    )

    st.dataframe(styled, use_container_width=True, height=400)


def render_agent_status(agent_name: str, status: str, detail: str = ""):
    """Render agent status indicator."""
    status_map = {
        "running": ("🔄", "#ffa726"),
        "complete": ("✅", "#00c853"),
        "error": ("❌", "#ff1744"),
        "waiting": ("⏳", "#888"),
    }
    icon, color = status_map.get(status, ("⚪", "#888"))

    st.markdown(f"""
    <div style="
        display: flex; align-items: center; gap: 10px;
        padding: 8px 16px; border-radius: 8px;
        background: {color}11; border-left: 3px solid {color};
        margin-bottom: 4px;
    ">
        <span style="font-size: 18px;">{icon}</span>
        <span style="color: #fff; font-weight: 600;">{agent_name}</span>
        <span style="color: #888; font-size: 12px; margin-left: auto;">{detail}</span>
    </div>
    """, unsafe_allow_html=True)
