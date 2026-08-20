import os
import json
from dotenv import load_dotenv

# Load env variables before importing anything else
load_dotenv()

from graph.state import TradingState
from agents.orchestrator import orchestrator_agent
from config.settings import get_settings

def run_test_case(name: str, state: TradingState, simulate_llm_failure: bool = False):
    print(f"\n{'='*50}\nTEST CASE: {name}\n{'='*50}")
    
    settings = get_settings()
    original_api_key = settings.google_api_key
    
    if simulate_llm_failure:
        print("Simulating LLM Failure (removing API key)...")
        settings.google_api_key = ""  # Force failure
        
    try:
        result = orchestrator_agent(state)
        decision = result["trade_decision"]
        print(f"ACTION: {decision['action']}")
        print(f"CONFIDENCE: {decision['confidence']}")
        print(f"REASONING:\n{decision['reasoning']}")
        print(f"LLM USED: {decision.get('llm_used', False)}")
        if decision['action'] != 'HOLD':
            print(f"QUANTITY: {decision.get('quantity')}")
            print(f"PRICE: {decision.get('price')}")
            print(f"SL: {decision.get('stop_loss')} TP: {decision.get('take_profit')}")
    except Exception as e:
        print(f"Error running orchestrator: {e}")
    finally:
        if simulate_llm_failure:
            settings.google_api_key = original_api_key

def main():
    # Base state template
    base_state = {
        "current_pair": "BTCINR",
        "market_data": {"last_price": 6000000.0},
        "portfolio": {
            "balances": {"INR": 100000.0},
            "positions": {"BTC": {"quantity": 0.05}}
        }
    }

    # 1. Clear BUY setup
    buy_state = base_state.copy()
    buy_state["technical_signals"] = {
        "composite_signal": "STRONG_BUY",
        "signal_score": 0.8,
        "confidence": 0.9,
        "atr": 100000.0
    }
    buy_state["sentiment_score"] = {
        "overall_score": 0.7,
        "fear_greed_index": 80,
        "confidence": 0.8
    }
    buy_state["risk_assessment"] = {
        "compliance_status": "PASS",
        "max_position_size": 10000.0,
        "current_drawdown_pct": 1.0,
        "value_at_risk": 500.0
    }

    # 2. Clear SELL setup
    sell_state = base_state.copy()
    sell_state["technical_signals"] = {
        "composite_signal": "STRONG_SELL",
        "signal_score": -0.9,
        "confidence": 0.85,
        "atr": 120000.0
    }
    sell_state["sentiment_score"] = {
        "overall_score": -0.6,
        "fear_greed_index": 20,
        "confidence": 0.7
    }
    sell_state["risk_assessment"] = {
        "compliance_status": "PASS",
        "max_position_size": 10000.0,
        "current_drawdown_pct": 2.0,
        "value_at_risk": 600.0
    }

    # 3. Conflicting setup (Mixed Signals)
    mixed_state = base_state.copy()
    mixed_state["technical_signals"] = {
        "composite_signal": "BUY",
        "signal_score": 0.4,
        "confidence": 0.5,
        "atr": 80000.0
    }
    mixed_state["sentiment_score"] = {
        "overall_score": -0.3,
        "fear_greed_index": 45,
        "confidence": 0.6
    }
    mixed_state["risk_assessment"] = {
        "compliance_status": "PASS",
        "max_position_size": 10000.0,
        "current_drawdown_pct": 1.5,
        "value_at_risk": 400.0
    }

    # 4. Compliance FAIL case
    fail_state = base_state.copy()
    fail_state["technical_signals"] = buy_state["technical_signals"]
    fail_state["sentiment_score"] = buy_state["sentiment_score"]
    fail_state["risk_assessment"] = {
        "compliance_status": "FAIL",
        "risk_warnings": ["Max daily trades exceeded", "High drawdown"]
    }

    # Run tests
    run_test_case("1. Clear BUY setup", buy_state)
    run_test_case("2. Clear SELL setup", sell_state)
    run_test_case("3. Conflicting setup", mixed_state)
    run_test_case("4. Compliance FAIL", fail_state)
    run_test_case("5. LLM Fallback (Simulated Error)", buy_state, simulate_llm_failure=True)

if __name__ == "__main__":
    main()
