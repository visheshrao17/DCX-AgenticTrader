"""
DCX-AgenticTrader — Structured Logging

Uses loguru for structured, color-coded logging with file rotation.
Each agent gets its own log prefix for easy filtering.
"""

import sys
from pathlib import Path
from loguru import logger

from config.constants import LOG_DIR


def setup_logger(log_level: str = "INFO") -> None:
    """
    Configure loguru logger with console + file sinks.

    - Console: color-coded, human-readable format
    - File: JSON-structured, rotated at 10MB, kept for 7 days
    """
    # Remove default handler
    logger.remove()

    # Create log directory
    log_path = Path(LOG_DIR)
    log_path.mkdir(parents=True, exist_ok=True)

    # Console sink — human-readable with colors
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[agent]:>15}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
        filter=lambda record: record["extra"].setdefault("agent", "system"),
    )

    # File sink — structured, rotated
    logger.add(
        str(log_path / "trading_{time:YYYY-MM-DD}.log"),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[agent]:>15} | {message}",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        filter=lambda record: record["extra"].setdefault("agent", "system"),
    )

    # Trade-specific log (separate file for audit trail)
    logger.add(
        str(log_path / "trades_{time:YYYY-MM-DD}.log"),
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {message}",
        rotation="10 MB",
        retention="30 days",
        filter=lambda record: record["extra"].get("agent") == "executor",
    )

    logger.bind(agent="system").info("Logger initialized")


def get_agent_logger(agent_name: str):
    """
    Get a logger instance bound to a specific agent name.

    Usage:
        log = get_agent_logger("market_data")
        log.info("Fetching candles for BTCINR")
    """
    return logger.bind(agent=agent_name)
