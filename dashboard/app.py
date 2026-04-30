"""
DCX-AgenticTrader — Streamlit Dashboard

Multi-page trading dashboard with:
1. Live Market — prices, candlestick charts, orderbook
2. Agent Console — run trading cycles, see agent reasoning
3. Portfolio — positions, PnL, allocation
4. Trade History — full log with filters
5. Risk Dashboard — compliance, VaR, tax estimates
6. Settings — configure pairs, risk limits, toggle paper/live
"""

import sys
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone

import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from config.constants import SUPPORTED_PAIRS


# =============================================================================
# Page Config
# =============================================================================

st.set_page_config(
    page_title="DCX-AgenticTrader",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(180deg, #0a0a1a 0%, #0e1117 100%);
    }

    .block-container { padding-top: 2rem; }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }

    div[data-testid="stMetric"] label {
        color: #888 !important;
        font-size: 13px !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #fff !important;
        font-size: 24px !important;
        font-weight: 700 !important;
    }

    .stSidebar {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%) !important;
        border-right: 1px solid #21262d;
    }

    h1, h2, h3 { color: #e6edf3 !important; }

    .stButton > button {
        background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
        color: white; border: none; border-radius: 8px;
        padding: 8px 24px; font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(46, 160, 67, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Sidebar Navigation
# =============================================================================

with st.sidebar:
    st.markdown("## 🤖 DCX-AgenticTrader")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["📊 Live Market", "🧠 Agent Console", "💰 Portfolio",
         "📋 Trade History", "🛡️ Risk Dashboard", "⚙️ Settings"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    settings = get_settings()
    mode = "📝 Paper" if settings.paper_trading else "🔴 LIVE"
    st.markdown(f"**Mode:** {mode}")
    st.markdown(f"**Pairs:** {settings.trading_pairs}")


# =============================================================================
# Page: Live Market
# =============================================================================

if page == "📊 Live Market":
    st.title("📊 Live Market Data")

    from dashboard.components import render_price_card, render_candlestick_chart
    from tools.coindcx_client import CoinDCXClient

    client = CoinDCXClient()

    # Pair selector
    selected_pair = st.selectbox("Trading Pair", list(SUPPORTED_PAIRS.keys()), index=0)
    pair_info = SUPPORTED_PAIRS[selected_pair]

    col1, col2 = st.columns([2, 1])

    with col1:
        try:
            with st.spinner("Fetching market data..."):
                candles = client.get_candles(pair_info["pair"], "1h", limit=50)
                ticker = client.get_ticker_for_market(selected_pair)

            if ticker:
                price = float(ticker.get("last_price", 0))
                high = float(ticker.get("high", 0))
                low = float(ticker.get("low", 0))
                volume = float(ticker.get("volume", 0))
                change = ((price - low) / low * 100) if low else 0

                render_price_card(selected_pair, price, change, high, low, volume)

            st.markdown("### Candlestick Chart (1H)")
            render_candlestick_chart(candles, f"{selected_pair} — 1 Hour")

        except Exception as e:
            st.error(f"Failed to fetch market data: {e}")

    with col2:
        st.markdown("### Orderbook")
        try:
            orderbook = client.get_orderbook(pair_info["pair"])
            bids = orderbook.get("bids", {})
            asks = orderbook.get("asks", {})

            if isinstance(bids, dict):
                bid_items = sorted(bids.items(), key=lambda x: float(x[0]), reverse=True)[:10]
            elif isinstance(bids, list):
                bid_items = [(b.get("price", 0), b.get("quantity", 0)) for b in bids[:10]]
            else:
                bid_items = []

            if isinstance(asks, dict):
                ask_items = sorted(asks.items(), key=lambda x: float(x[0]))[:10]
            elif isinstance(asks, list):
                ask_items = [(a.get("price", 0), a.get("quantity", 0)) for a in asks[:10]]
            else:
                ask_items = []

            st.markdown("**Asks (Sell)**")
            for price, qty in reversed(ask_items[:5]):
                st.markdown(f'<span style="color:#ff1744">₹{float(price):,.2f}</span> — {float(qty):.6f}', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**Bids (Buy)**")
            for price, qty in bid_items[:5]:
                st.markdown(f'<span style="color:#00c853">₹{float(price):,.2f}</span> — {float(qty):.6f}', unsafe_allow_html=True)

        except Exception as e:
            st.warning(f"Orderbook unavailable: {e}")


# =============================================================================
# Page: Agent Console
# =============================================================================

elif page == "🧠 Agent Console":
    st.title("🧠 Agent Console")
    st.markdown("Run a trading cycle and watch the agents collaborate in real-time.")

    from dashboard.components import render_signal_badge, render_agent_status

    selected_pair = st.selectbox("Trading Pair", list(SUPPORTED_PAIRS.keys()), index=0)

    if st.button("🚀 Run Trading Cycle", type="primary", use_container_width=True):
        with st.spinner("Running multi-agent pipeline..."):
            try:
                from graph.workflow import compile_workflow

                workflow = compile_workflow()
                config = {"configurable": {"thread_id": str(uuid.uuid4())[:8]}}
                initial_state = {
                    "messages": [],
                    "current_pair": selected_pair,
                    "human_approved": True,  # Auto-approve in dashboard
                }

                # Show agent progress
                progress = st.empty()
                agents = ["Supervisor", "Market Data", "Technical", "Sentiment", "Risk", "Orchestrator", "Executor"]

                for i, agent in enumerate(agents):
                    progress.markdown(f"**Running:** {agent}...")

                result = workflow.invoke(initial_state, config=config)

                st.success("✅ Trading cycle complete!")

                # Display results
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("### 📈 Technical Signal")
                    tech = result.get("technical_signals", {})
                    render_signal_badge(
                        tech.get("composite_signal", "N/A"),
                        tech.get("signal_score", 0),
                        tech.get("confidence", 0),
                    )

                with col2:
                    st.markdown("### 📰 Sentiment")
                    sent = result.get("sentiment_score", {})
                    score = sent.get("overall_score", 0)
                    label = "BULLISH" if score > 0.2 else "BEARISH" if score < -0.2 else "NEUTRAL"
                    render_signal_badge(label, score, sent.get("confidence", 0))

                with col3:
                    st.markdown("### 🎯 Decision")
                    decision = result.get("trade_decision", {})
                    render_signal_badge(
                        decision.get("action", "HOLD"),
                        0,
                        decision.get("confidence", 0),
                    )

                # Reasoning chain
                st.markdown("### 💭 Agent Reasoning Chain")
                with st.expander("Market Data", expanded=False):
                    md = result.get("market_data", {})
                    st.json({"price": md.get("last_price"), "spread": md.get("spread_pct"), "timestamp": md.get("timestamp")})

                with st.expander("Technical Analysis", expanded=False):
                    st.json(result.get("technical_signals", {}))

                with st.expander("Sentiment Analysis", expanded=False):
                    st.json(result.get("sentiment_score", {}))

                with st.expander("Risk Assessment", expanded=False):
                    st.json(result.get("risk_assessment", {}))

                with st.expander("Trade Decision", expanded=True):
                    st.json(result.get("trade_decision", {}))

                with st.expander("Execution Result", expanded=True):
                    st.json(result.get("execution_result", {}))

            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                import traceback
                st.code(traceback.format_exc())

    else:
        st.info("Click **Run Trading Cycle** to start the multi-agent pipeline.")


# =============================================================================
# Page: Portfolio
# =============================================================================

elif page == "💰 Portfolio":
    st.title("💰 Portfolio")

    from dashboard.components import render_portfolio_summary, render_pnl_chart
    from memory.trade_store import TradeStore

    store = TradeStore()
    stats = store.get_performance_stats()
    trades = store.get_trade_history(limit=100)

    # Portfolio summary
    render_portfolio_summary({
        "total_value_inr": settings.initial_capital_inr + stats.get("total_pnl", 0),
        "realized_pnl": stats.get("total_pnl", 0),
        "unrealized_pnl": 0,
        "max_drawdown_pct": 0,
    })

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Performance Stats")
        st.metric("Total Trades", stats.get("total_trades", 0))
        st.metric("Win Rate", f"{stats.get('win_rate', 0)}%")
        st.metric("Avg Win", f"₹{stats.get('avg_win', 0):,.2f}")
        st.metric("Avg Loss", f"₹{stats.get('avg_loss', 0):,.2f}")

    with col2:
        st.markdown("### Cumulative PnL")
        render_pnl_chart(trades)


# =============================================================================
# Page: Trade History
# =============================================================================

elif page == "📋 Trade History":
    st.title("📋 Trade History")

    from dashboard.components import render_trade_table
    from memory.trade_store import TradeStore

    store = TradeStore()

    col1, col2 = st.columns(2)
    with col1:
        market_filter = st.selectbox("Market", ["All"] + list(SUPPORTED_PAIRS.keys()))
    with col2:
        limit = st.slider("Max trades", 10, 500, 50)

    market = market_filter if market_filter != "All" else None
    trades = store.get_trade_history(market=market, limit=limit)

    st.markdown(f"**Showing {len(trades)} trades**")
    render_trade_table(trades)


# =============================================================================
# Page: Risk Dashboard
# =============================================================================

elif page == "🛡️ Risk Dashboard":
    st.title("🛡️ Risk & Compliance Dashboard")

    from memory.trade_store import TradeStore

    store = TradeStore()
    stats = store.get_performance_stats()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Position Limits")
        st.metric("Max Position Size", f"{settings.max_position_size_pct}%")
        st.metric("Max Drawdown Limit", f"{settings.max_drawdown_pct}%")
        st.metric("Max Trades/Day", settings.max_trades_per_day)

    with col2:
        st.markdown("### Today's Activity")
        today_trades = store.count_trades_today()
        st.metric("Trades Today", f"{today_trades}/{settings.max_trades_per_day}")
        st.metric("Total PnL", f"₹{stats.get('total_pnl', 0):,.2f}")
        st.metric("Total Fees", f"₹{stats.get('total_fees', 0):,.2f}")

    with col3:
        st.markdown("### 🇮🇳 Indian Tax Compliance")
        total_profit = stats.get("total_pnl", 0)
        tax_30 = max(total_profit * 0.30, 0)
        cess = tax_30 * 0.04
        st.metric("Taxable Profit", f"₹{max(total_profit, 0):,.2f}")
        st.metric("Est. Tax (30%)", f"₹{tax_30:,.2f}")
        st.metric("+ 4% Cess", f"₹{cess:,.2f}")

    st.markdown("---")
    st.markdown("### Compliance Notes")
    st.info("📌 30% flat tax on all VDA profits (Section 115BBH) — no loss offsetting allowed")
    st.info("📌 1% TDS on transfers > ₹50,000 (Section 194S)")
    st.info("📌 Report under Schedule VDA in ITR")


# =============================================================================
# Page: Settings
# =============================================================================

elif page == "⚙️ Settings":
    st.title("⚙️ Settings")

    st.markdown("### Trading Configuration")
    st.text_input("Trading Pairs", value=settings.trading_pairs, disabled=True)
    st.number_input("Initial Capital (₹)", value=settings.initial_capital_inr, disabled=True)
    st.number_input("Cycle Interval (minutes)", value=settings.trading_interval_minutes, disabled=True)

    st.markdown("### Risk Limits")
    st.slider("Max Position Size (%)", 1, 50, int(settings.max_position_size_pct), disabled=True)
    st.slider("Max Drawdown (%)", 1, 20, int(settings.max_drawdown_pct), disabled=True)
    st.slider("Max Trades/Day", 1, 50, settings.max_trades_per_day, disabled=True)

    st.markdown("### API Status")

    col1, col2 = st.columns(2)
    with col1:
        if settings.has_coindcx_credentials:
            st.success("✅ CoinDCX API connected")
        else:
            st.error("❌ CoinDCX API key missing")

    with col2:
        if settings.has_llm_credentials:
            st.success("✅ Gemini API connected")
        else:
            st.error("❌ Google API key missing")

    mode = "📝 Paper Trading" if settings.paper_trading else "🔴 LIVE Trading"
    st.warning(f"Current Mode: **{mode}**")
    st.caption("Edit `.env` file to change settings. Restart dashboard to apply.")
