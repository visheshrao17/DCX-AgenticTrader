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
    # LLM Provider
    # -------------------------------------------------------------------------
    llm_provider: str = Field(
        default="openrouter",
        description="LLM provider: 'openrouter', 'gemini', or 'none'",
    )
    openrouter_api_key: str = Field(default="", description="OpenRouter API key")
    llm_candidate_models: str = Field(
        default="meta-llama/llama-4-maverick:free,google/gemini-2.0-flash-exp:free,openrouter/auto",
        description="Comma-separated ordered list of candidate model IDs for OpenRouter",
    )

    # Legacy Gemini (kept for rollback)
    google_api_key: str = Field(default="", description="Google AI API key")
    gemini_model: str = Field(default="gemini-2.0-flash", description="Gemini model name")

    # Per-agent LLM toggles
    use_llm_technical: bool = Field(
        default=True,
        description="Enable LLM synthesis in Technical Analyst agent",
    )
    use_llm_sentiment: bool = Field(
        default=True,
        description="Enable LLM analysis in Sentiment Researcher agent",
    )
    use_llm_risk_explanation: bool = Field(
        default=True,
        description="Enable LLM-generated compliance rationale in Risk agent",
    )
    use_llm_orchestrator: bool = Field(
        default=True,
        description="Enable LLM decision in Strategy Orchestrator",
    )

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
        """Check if LLM credentials are configured for the active provider."""
        if self.llm_provider == "openrouter":
            return bool(self.openrouter_api_key)
        elif self.llm_provider == "gemini":
            return bool(self.google_api_key)
        return False

    @property
    def llm_candidate_models_list(self) -> List[str]:
        """Parse comma-separated candidate models into an ordered list."""
        return [m.strip() for m in self.llm_candidate_models.split(",") if m.strip()]

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


def get_settings() -> Settings:
    """Get application settings singleton."""
    return Settings()
