"""
DCX-AgenticTrader — Settings

Pydantic Settings class that loads environment variables from .env file.
Validates all required config on startup so we fail fast if something's missing.
"""

from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # -------------------------------------------------------------------------
    # CoinDCX API
    # -------------------------------------------------------------------------
    coindcx_api_key: str = Field(default="", description="CoinDCX API key")
    coindcx_api_secret: str = Field(default="", description="CoinDCX API secret")

    # -------------------------------------------------------------------------
    # LLM Provider (Google Gemini)
    # -------------------------------------------------------------------------
    google_api_key: str = Field(default="", description="Google AI API key")
    gemini_model: str = Field(default="gemini-2.0-flash", description="Gemini model name")

    # -------------------------------------------------------------------------
    # News & Sentiment (Yahoo Finance — no API key needed)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Trading Configuration
    # -------------------------------------------------------------------------
    paper_trading: bool = Field(default=True, description="Enable paper trading mode")
    trading_pairs: str = Field(
        default="BTCINR,USDTINR",
        description="Comma-separated trading pairs",
    )
    initial_capital_inr: float = Field(
        default=100_000.0,
        description="Initial simulated capital in INR",
    )
    trading_interval_minutes: int = Field(
        default=15,
        description="Minutes between trading cycles",
    )

    # -------------------------------------------------------------------------
    # Risk Management
    # -------------------------------------------------------------------------
    max_position_size_pct: float = Field(
        default=10.0,
        description="Max position size as % of portfolio",
    )
    max_drawdown_pct: float = Field(
        default=8.0,
        description="Max allowed drawdown %",
    )
    max_trades_per_day: int = Field(
        default=10,
        description="Max number of trades per day",
    )

    # -------------------------------------------------------------------------
    # Dashboard
    # -------------------------------------------------------------------------
    streamlit_port: int = Field(default=8501, description="Streamlit server port")

    # -------------------------------------------------------------------------
    # Derived Properties
    # -------------------------------------------------------------------------
    @property
    def trading_pairs_list(self) -> List[str]:
        """Parse comma-separated trading pairs into a list."""
        return [p.strip() for p in self.trading_pairs.split(",") if p.strip()]

    @property
    def has_coindcx_credentials(self) -> bool:
        """Check if CoinDCX API credentials are configured."""
        return bool(self.coindcx_api_key and self.coindcx_api_secret)

    @property
    def has_llm_credentials(self) -> bool:
        """Check if Google Gemini credentials are configured."""
        return bool(self.google_api_key)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


def get_settings() -> Settings:
    """Get application settings singleton."""
    return Settings()
