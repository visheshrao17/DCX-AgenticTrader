"""
DCX-AgenticTrader — Sentiment Researcher Agent

Analyzes India-focused crypto news and market sentiment.
Combines Yahoo Finance news, RSS feeds, Fear & Greed Index,
and rule-based headline analysis. All FREE — no API keys needed.
"""

import json
from typing import Dict, Any, List

from pydantic import BaseModel, Field

from tools.sentiment_tools import (
    fetch_yahoo_finance_news, fetch_rss_news,
    get_fear_greed_index, analyze_headlines_sentiment,
)
from graph.state import TradingState
from utils.logger import get_agent_logger
from config.settings import get_settings
from tools.llm_client import call_llm
from utils.error_handler import LLMError

log = get_agent_logger("sentiment")


class HeadlineSentiment(BaseModel):
    title: str = Field(description="The headline text")
    score: float = Field(description="Sentiment score from -1.0 (very negative) to 1.0 (very positive)")
    india_relevant: bool = Field(description="True if this headline specifically mentions or affects India (e.g. RBI, INR, Indian exchanges)")


class SentimentLLMAnalysis(BaseModel):
    """Structured output expected from the Sentiment LLM."""
    headlines: List[HeadlineSentiment] = Field(description="Individual analysis of each headline")
    overall_score: float = Field(description="Composite sentiment score from -1.0 to 1.0 considering both news and Fear&Greed index")
    india_score: float = Field(description="Sentiment score from -1.0 to 1.0 for India-specific news (0.0 if none found)")
    rationale: str = Field(description="Brief explanation of the overall sentiment picture")


def sentiment_agent(state: TradingState) -> Dict[str, Any]:
    """
    LangGraph node: Sentiment Researcher Agent.

    Fetches news from Yahoo Finance + RSS feeds, analyzes sentiment,
    and produces a composite score weighted toward India-specific news.
    """
    log.info("=== Sentiment Researcher Agent running ===")

    pair = state.get("current_pair", "BTCINR")

    # Map trading pair to Yahoo Finance ticker
    yf_symbol_map = {
        "BTCINR": "BTC-INR",
        "USDTINR": "USDT-INR",
        "ETHINR": "ETH-INR",
    }
    yf_symbol = yf_symbol_map.get(pair, "BTC-INR")

    # 1. Fetch Fear & Greed Index
    fng_raw = get_fear_greed_index.invoke({})
    try:
        fng = json.loads(fng_raw) if isinstance(fng_raw, str) else fng_raw
    except Exception:
        fng = {"value": 50, "classification": "Neutral"}

    fng_value = fng.get("value", 50)
    fng_label = fng.get("classification", "Neutral")
    log.info(f"Fear & Greed: {fng_value} ({fng_label})")

    # 2. Fetch Yahoo Finance news
    yf_articles = []
    try:
        yf_raw = fetch_yahoo_finance_news.invoke({"symbol": yf_symbol})
        yf_data = json.loads(yf_raw) if isinstance(yf_raw, str) else yf_raw
        yf_articles = yf_data.get("articles", [])
        log.info(f"Yahoo Finance: {len(yf_articles)} articles for {yf_symbol}")
    except Exception as e:
        log.warning(f"Yahoo Finance fetch failed (non-critical): {e}")

    # 3. Fetch RSS feed news
    rss_raw = fetch_rss_news.invoke({"max_per_feed": 5})
    try:
        rss_data = json.loads(rss_raw) if isinstance(rss_raw, str) else rss_raw
    except Exception:
        rss_data = {"articles": []}
    rss_articles = rss_data.get("articles", [])

    # 4. Combine all headlines for sentiment analysis
    all_headlines = []
    for article in yf_articles:
        title = article.get("title", "")
        if title:
            all_headlines.append(title)
    for article in rss_articles:
        title = article.get("title", "")
        if title:
            all_headlines.append(title)

    # 5 & 6. Analyze sentiment (Deterministic Fallback)
    sentiment_score = 0.0
    headlines_analyzed = 0
    india_score = 0.0
    india_headlines = []

    if all_headlines:
        sentiment_raw = analyze_headlines_sentiment.invoke({
            "headlines_json": json.dumps(all_headlines[:20])
        })
        try:
            sentiment_data = json.loads(sentiment_raw) if isinstance(sentiment_raw, str) else sentiment_raw
            sentiment_score = sentiment_data.get("overall_score", 0.0)
            headlines_analyzed = sentiment_data.get("headlines_analyzed", 0)
        except Exception:
            pass

        india_headlines = [h for h in all_headlines if any(
            w in h.lower() for w in ["india", "indian", "rbi", "sebi", "inr", "rupee", "fiu"]
        )]
        if india_headlines:
            india_raw = analyze_headlines_sentiment.invoke({
                "headlines_json": json.dumps(india_headlines)
            })
            try:
                india_data = json.loads(india_raw) if isinstance(india_raw, str) else india_raw
                india_score = india_data.get("overall_score", 0.0)
            except Exception:
                pass

    # 7. Composite: 40% headline, 30% FNG, 30% India-specific
    fng_normalized = (fng_value - 50) / 50.0
    composite = (
        sentiment_score * 0.4 +
        fng_normalized * 0.3 +
        india_score * 0.3
    )
    composite = max(-1.0, min(1.0, composite))

    # Build top news list
    top_news = []
    for article in (yf_articles + rss_articles)[:5]:
        top_news.append({
            "title": article.get("title", "")[:100],
            "source": article.get("publisher", article.get("source", "")),
            "published": article.get("published", article.get("published", "")),
        })

    # Confidence
    confidence = 0.3
    if headlines_analyzed > 10:
        confidence += 0.2
    if fng.get("error") is None:
        confidence += 0.2
    if india_headlines:
        confidence += 0.2
    confidence = min(confidence, 0.95)

    reasoning = (
        f"Sentiment analysis for {pair}: "
        f"Headlines score={sentiment_score:.2f} ({headlines_analyzed} articles), "
        f"Fear&Greed={fng_value} ({fng_label}), "
        f"India-specific={india_score:.2f} ({len(india_headlines)} articles). "
        f"Composite: {composite:.3f}."
    )

    result = {
        "overall_score": round(composite, 3),
        "fear_greed_index": fng_value,
        "fear_greed_label": fng_label,
        "india_sentiment": round(india_score, 3),
        "global_sentiment": round(sentiment_score, 3),
        "top_news": top_news,
        "sources_analyzed": headlines_analyzed,
        "confidence": round(confidence, 2),
        "reasoning": reasoning,
        "llm_used": False,
    }

    # =========================================================================
    # LLM Synthesis
    # =========================================================================
    settings = get_settings()
    if settings.use_llm_sentiment and all_headlines:
        try:
            system_prompt = "You are an expert crypto market Sentiment Analyst. Read the headlines and the Fear & Greed index, then output a structured sentiment analysis."
            user_prompt = f"Fear & Greed Index: {fng_value} ({fng_label})\n\nHeadlines:\n"
            for i, hl in enumerate(all_headlines[:30]):
                user_prompt += f"{i+1}. {hl}\n"

            llm_decision = call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=SentimentLLMAnalysis,
                agent_name="sentiment",
                temperature=0.2,
            )

            # Override with LLM results
            result["overall_score"] = round(llm_decision.overall_score, 3)
            result["global_sentiment"] = round(llm_decision.overall_score, 3) # Simplify by combining
            result["india_sentiment"] = round(llm_decision.india_score, 3)
            result["sources_analyzed"] = len(llm_decision.headlines)
            result["reasoning"] = f"[LLM] {llm_decision.rationale} (F&G: {fng_value})"
            result["llm_used"] = True
            log.info(f"LLM Sentiment Analysis generated successfully: {result['overall_score']}")

        except LLMError as e:
            log.warning(f"Sentiment LLM failed, using deterministic signals: {e}")
        except Exception as e:
            log.warning(f"Sentiment LLM failed unexpectedly: {e}")

    log.info(f"Sentiment: {result['overall_score']:.3f} (FNG={fng_value}, headlines={result['sources_analyzed']})")

    return {
        "sentiment_score": result,
        "current_step": "sentiment",
    }
