"""
DCX-AgenticTrader — Main Entry Point

CLI interface for running the trading bot, backtesting, and launching the dashboard.

Usage:
    python main.py run              # Start trading loop (paper mode)
    python main.py run --once       # Single cycle
    python main.py run --live       # Live trading (requires approval)
    python main.py backtest         # Run backtesting
    python main.py dashboard        # Launch Streamlit dashboard
"""

import sys
import time
import uuid
from pathlib import Path

import click

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import get_settings
from utils.logger import setup_logger, get_agent_logger


@click.group()
def cli():
    """DCX-AgenticTrader — Multi-Agent Crypto Trading Desk"""
    pass


@cli.command()
@click.option("--once", is_flag=True, help="Run a single trading cycle")
@click.option("--live", is_flag=True, help="Enable live trading (default: paper)")
@click.option("--pair", default=None, help="Trading pair (e.g., BTCINR)")
@click.option("--interval", default=None, type=int, help="Minutes between cycles")
def run(once: bool, live: bool, pair: str, interval: int):
    """Start the trading loop."""
    setup_logger("INFO")
    log = get_agent_logger("main")

    settings = get_settings()

    if live:
        settings.paper_trading = False
        log.warning("🔴 LIVE TRADING MODE ENABLED — Real money at risk!")
        if not settings.has_coindcx_credentials:
            log.error("CoinDCX API credentials not configured. Set COINDCX_API_KEY and COINDCX_API_SECRET in .env")
            return
    else:
        log.info("📝 Paper trading mode (simulated)")

    trading_pair = pair or settings.trading_pairs_list[0]
    cycle_interval = interval or settings.trading_interval_minutes

    log.info(f"Trading pair: {trading_pair}")
    log.info(f"Cycle interval: {cycle_interval} minutes")
    log.info(f"Initial capital: ₹{settings.initial_capital_inr:,.2f}")

    # Build the LangGraph workflow
    from graph.workflow import compile_workflow
    workflow = compile_workflow()

    log.info("=" * 60)
    log.info("🚀 DCX-AgenticTrader Starting")
    log.info("=" * 60)

    cycle = 0
    while True:
        cycle += 1
        thread_id = str(uuid.uuid4())[:8]
        log.info(f"\n{'='*60}")
        log.info(f"📊 Trading Cycle {cycle} — {trading_pair}")
        log.info(f"{'='*60}")

        try:
            config = {"configurable": {"thread_id": thread_id}}
            initial_state = {
                "messages": [],
                "current_pair": trading_pair,
                "human_approved": settings.paper_trading,  # Auto-approve in paper mode
            }

            # Run the full pipeline
            result = workflow.invoke(initial_state, config=config)

            # Log results
            decision = result.get("trade_decision", {})
            execution = result.get("execution_result", {})
            portfolio = result.get("portfolio", {})

            log.info(f"Decision: {decision.get('action', 'N/A')} | "
                     f"Confidence: {decision.get('confidence', 0):.1%}")

            if execution and execution.get("status") == "filled":
                log.info(f"Executed: {execution.get('fill_quantity', 0)} @ "
                         f"₹{execution.get('fill_price', 0):,.2f} | "
                         f"PnL: ₹{execution.get('realized_pnl', 0):,.2f}")

            if portfolio:
                log.info(f"Portfolio: ₹{portfolio.get('total_value_inr', 0):,.2f} | "
                         f"PnL: ₹{portfolio.get('realized_pnl', 0):,.2f} | "
                         f"Drawdown: {portfolio.get('max_drawdown_pct', 0):.1f}%")

        except KeyboardInterrupt:
            log.info("\n🛑 Trading stopped by user")
            break
        except Exception as e:
            log.error(f"Cycle {cycle} failed: {e}")
            import traceback
            traceback.print_exc()

        if once:
            log.info("Single cycle complete — exiting")
            break

        # Wait for next cycle
        log.info(f"⏳ Next cycle in {cycle_interval} minutes...")
        try:
            time.sleep(cycle_interval * 60)
        except KeyboardInterrupt:
            log.info("\n🛑 Trading stopped by user")
            break

    log.info("DCX-AgenticTrader stopped")


@cli.command()
@click.option("--pair", default="BTCINR", help="Trading pair to backtest")
@click.option("--months", default=6, help="Months of historical data")
def backtest(pair: str, months: int):
    """Run backtesting simulation."""
    setup_logger("INFO")
    log = get_agent_logger("backtest")
    log.info(f"Starting backtesting for {pair} over {months} months...")
    
    from backtesting.engine import BacktestEngine
    from backtesting.report import generate_report, print_report_summary
    
    engine = BacktestEngine(market=pair, initial_capital=100000)
    results = engine.run(months=months)
    
    if "error" in results:
        log.error(f"Backtest failed: {results['error']}")
        return
        
    # Print summary to console
    summary = print_report_summary(results)
    print(summary)
    
    # Generate and save HTML report
    fig = generate_report(results)
    report_path = Path(__file__).parent / f"backtest_report_{pair}.html"
    fig.write_html(str(report_path))
    log.info(f"✅ Interactive backtest report saved to: {report_path}")


@cli.command()
def dashboard():
    """Launch the Streamlit dashboard."""
    setup_logger("INFO")
    log = get_agent_logger("main")

    import subprocess
    dashboard_path = Path(__file__).parent / "dashboard" / "app.py"

    if not dashboard_path.exists():
        log.error(f"Dashboard not found at {dashboard_path}")
        return

    settings = get_settings()
    log.info(f"Launching Streamlit dashboard on port {settings.streamlit_port}...")

    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.port", str(settings.streamlit_port),
        "--server.headless", "true",
    ])

@cli.command()
def visualize():
    """Generate a visualization of the LangGraph workflow."""
    setup_logger("INFO")
    log = get_agent_logger("main")
    
    from graph.workflow import compile_workflow
    log.info("Compiling workflow for visualization...")
    workflow = compile_workflow()
    
    try:
        # Get PNG bytes from LangGraph
        png_bytes = workflow.get_graph().draw_mermaid_png()
        
        output_path = Path(__file__).parent / "graph_visualization.png"
        with open(output_path, "wb") as f:
            f.write(png_bytes)
            
        log.info(f"✅ Workflow visualization saved to: {output_path}")
    except Exception as e:
        log.error(f"Failed to generate visualization. Ensure internet connection (uses mermaid.ink) or required libraries are installed. Error: {e}")
        
        # Fallback to saving mermaid text
        md_path = Path(__file__).parent / "graph_visualization.md"
        mermaid_syntax = workflow.get_graph().draw_mermaid()
        with open(md_path, "w") as f:
            f.write(f"```mermaid\n{mermaid_syntax}\n```")
        log.info(f"Saved Mermaid syntax fallback to: {md_path}")
if __name__ == "__main__":
    cli()
