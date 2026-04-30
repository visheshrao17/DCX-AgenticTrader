"""
DCX-AgenticTrader — LangGraph Workflow Compilation

Assembles the full StateGraph with all agent nodes,
conditional routing edges, and checkpointing.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from graph.state import TradingState
from graph.supervisor import (
    supervisor_node,
    human_review_node,
    route_after_supervisor,
    route_after_market_data,
    route_after_technical,
    route_after_sentiment,
    route_after_risk,
    route_after_orchestrator,
    route_after_human_review,
)
from agents.market_data import market_data_agent
from agents.technical import technical_agent
from agents.sentiment import sentiment_agent
from agents.risk_compliance import risk_compliance_agent
from agents.orchestrator import orchestrator_agent
from agents.executor import executor_agent
from utils.logger import get_agent_logger

log = get_agent_logger("workflow")


def build_trading_graph() -> StateGraph:
    """
    Build the complete LangGraph trading pipeline.

    Flow:
        supervisor → market_data → technical → sentiment → risk
        → orchestrator → [human_review] → executor → END

    Returns:
        Compiled StateGraph ready to invoke.
    """
    log.info("Building LangGraph trading pipeline...")

    graph = StateGraph(TradingState)

    # =========================================================================
    # Add Nodes
    # =========================================================================

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("market_data", market_data_agent)
    graph.add_node("technical", technical_agent)
    graph.add_node("sentiment", sentiment_agent)
    graph.add_node("risk", risk_compliance_agent)
    graph.add_node("orchestrator", orchestrator_agent)
    graph.add_node("human_review", human_review_node)
    graph.add_node("executor", executor_agent)

    # =========================================================================
    # Add Edges (routing logic)
    # =========================================================================

    # Entry point
    graph.set_entry_point("supervisor")

    # Supervisor → Market Data (always)
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"market_data": "market_data"},
    )

    # Market Data → Technical (or end on error)
    graph.add_conditional_edges(
        "market_data",
        route_after_market_data,
        {"technical": "technical", "end": END},
    )

    # Technical → Sentiment
    graph.add_conditional_edges(
        "technical",
        route_after_technical,
        {"sentiment": "sentiment"},
    )

    # Sentiment → Risk
    graph.add_conditional_edges(
        "sentiment",
        route_after_sentiment,
        {"risk": "risk"},
    )

    # Risk → Orchestrator
    graph.add_conditional_edges(
        "risk",
        route_after_risk,
        {"orchestrator": "orchestrator"},
    )

    # Orchestrator → Human Review / Executor / End
    graph.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "human_review": "human_review",
            "executor": "executor",
            "end": END,
        },
    )

    # Human Review → Executor / End
    graph.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {"executor": "executor", "end": END},
    )

    # Executor → End
    graph.add_edge("executor", END)

    log.info("Trading pipeline built successfully")
    return graph


def compile_workflow(checkpointer=None):
    """
    Compile the graph with optional checkpointing.

    Args:
        checkpointer: LangGraph checkpointer for state persistence.
                      Defaults to MemorySaver (in-memory).

    Returns:
        Compiled, runnable workflow.
    """
    graph = build_trading_graph()

    if checkpointer is None:
        checkpointer = MemorySaver()

    workflow = graph.compile(checkpointer=checkpointer)
    log.info("Workflow compiled with checkpointer")
    return workflow
