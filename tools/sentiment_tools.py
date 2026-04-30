"""
DCX-AgenticTrader — Sentiment Tools

Fetches crypto news from Yahoo Finance and RSS feeds (FREE, no API key).
Analyzes sentiment and provides the Fear & Greed index.
Focused on India crypto market.
"""

import json
from typing import Dict, Any, List
from datetime import datetime, timezone

import requests
import feedparser
from langchain_core.tools import tool

from utils.logger import get_agent_logger

log = get_agent_logger("sentiment")

# India-focused and global crypto RSS feeds
RSS_FEEDS = [
    {"name": "CoinDesk", "url": "https://www.coindesk.com/arc/outboundfeeds/rss/", "weight": 1.0},
    {"name": "CoinTelegraph", "url": "https://cointelegraph.com/rss", "weight": 1.0},
    {"name": "CryptoSlate", "url": "https://cryptoslate.com/feed/", "weight": 0.8},
    {"name": "LiveMint-Markets", "url": "https://www.livemint.com/rss/markets", "weight": 1.5},
]


@tool
def fetch_yahoo_finance_news(symbol: str = "BTC-INR") -> str:
    """
    Fetch latest crypto news from Yahoo Finance for a given symbol.
    Free — no API key required.

    Args:
        symbol: Yahoo Finance ticker symbol (e.g., 'BTC-INR', 'BTC-USD').

    Returns:
        JSON string of news articles with title, publisher, and link.
    """
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        news = ticker.news

        if not news:
            log.warning(f"No Yahoo Finance news for {symbol}")
            return json.dumps({"articles": [], "count": 0, "source": "yahoo_finance"})

        articles = []
        for item in news[:15]:
            articles.append({
                "title": item.get("title", ""),
                "publisher": item.get("publisher", ""),
                "link": item.get("link", ""),
                "published": datetime.fromtimestamp(
                    item.get("providerPublishTime", 0), tz=timezone.utc
                ).isoformat() if item.get("providerPublishTime") else "",
                "type": item.get("type", ""),
            })

        log.info(f"Fetched {len(articles)} Yahoo Finance news for {symbol}")
        return json.dumps({"articles": articles, "count": len(articles), "source": "yahoo_finance"})

    except Exception as e:
        log.error(f"Yahoo Finance news fetch failed: {e}")
        return json.dumps({"articles": [], "count": 0, "error": str(e)})


@tool
def fetch_rss_news(max_per_feed: int = 5) -> str:
    """
    Fetch latest crypto news from RSS feeds (no API key needed).

    Args:
        max_per_feed: Max articles per RSS feed.

    Returns:
        JSON string of articles from multiple crypto news sources.
    """
    all_articles = []

    for feed_info in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_info["url"])
            for entry in feed.entries[:max_per_feed]:
                all_articles.append({
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", "")[:200],
                    "source": feed_info["name"],
                    "published": entry.get("published", ""),
                    "link": entry.get("link", ""),
                    "weight": feed_info["weight"],
                })
        except Exception as e:
            log.warning(f"RSS fetch failed for {feed_info['name']}: {e}")

    log.info(f"Fetched {len(all_articles)} articles from {len(RSS_FEEDS)} RSS feeds")
    return json.dumps({"articles": all_articles, "count": len(all_articles)})


@tool
def get_fear_greed_index() -> str:
    """
    Get the current Crypto Fear & Greed Index from Alternative.me.
    Free — no API key required.

    Returns:
        JSON string with index value (0-100), classification, and timestamp.
    """
    try:
        response = requests.get(
            "https://api.alternative.me/fng/?limit=1",
            timeout=10,
        )
        data = response.json()

        if "data" in data and data["data"]:
            fng = data["data"][0]
            result = {
                "value": int(fng.get("value", 50)),
                "classification": fng.get("value_classification", "Neutral"),
                "timestamp": fng.get("timestamp", ""),
            }
            log.info(f"Fear & Greed Index: {result['value']} ({result['classification']})")
            return json.dumps(result)

        return json.dumps({"value": 50, "classification": "Neutral", "error": "No data"})

    except Exception as e:
        log.error(f"Fear & Greed fetch failed: {e}")
        return json.dumps({"value": 50, "classification": "Neutral", "error": str(e)})


@tool
def analyze_headlines_sentiment(headlines_json: str) -> str:
    """
    Simple rule-based sentiment scoring for crypto headlines.
    Returns a score from -1.0 (very bearish) to +1.0 (very bullish).

    Args:
        headlines_json: JSON string containing list of headline strings.

    Returns:
        JSON with overall score and per-headline scores.
    """
    try:
        headlines = json.loads(headlines_json)
        if isinstance(headlines, dict):
            headlines = headlines.get("headlines", [])
    except Exception:
        return json.dumps({"error": "Invalid JSON", "score": 0.0})

    bullish_words = [
        "surge", "rally", "bull", "breakout", "soar", "gain", "boost", "high",
        "record", "pump", "moon", "adoption", "approval", "institutional",
        "bullish", "recovery", "growth", "positive", "uptrend", "profit",
    ]
    bearish_words = [
        "crash", "plunge", "bear", "dump", "drop", "fall", "loss", "low",
        "ban", "regulate", "hack", "scam", "fraud", "selloff", "correction",
        "bearish", "decline", "negative", "downtrend", "fear", "warning",
    ]
    india_boost_words = ["india", "rbi", "sebi", "inr", "rupee", "indian", "fiu"]

    results = []
    total_score = 0.0

    for headline in headlines:
        if not isinstance(headline, str):
            headline = str(headline)
        lower = headline.lower()

        score = 0.0
        bull_hits = sum(1 for w in bullish_words if w in lower)
        bear_hits = sum(1 for w in bearish_words if w in lower)
        india_relevant = any(w in lower for w in india_boost_words)

        if bull_hits + bear_hits > 0:
            score = (bull_hits - bear_hits) / (bull_hits + bear_hits)
        weight = 1.5 if india_relevant else 1.0

        results.append({
            "headline": headline[:100],
            "score": round(score, 2),
            "india_relevant": india_relevant,
        })
        total_score += score * weight

    avg_score = total_score / len(headlines) if headlines else 0.0
    avg_score = max(-1.0, min(1.0, avg_score))

    return json.dumps({
        "overall_score": round(avg_score, 3),
        "headlines_analyzed": len(headlines),
        "details": results[:10],
    })


SENTIMENT_TOOLS = [fetch_yahoo_finance_news, fetch_rss_news, get_fear_greed_index, analyze_headlines_sentiment]
