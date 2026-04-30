"""
DCX-AgenticTrader — RAG Compliance Tool

RAG (Retrieval Augmented Generation) over Indian crypto regulatory docs.
Loads compliance documents into ChromaDB and answers regulatory questions.
"""

import json
from typing import List, Dict, Any
from pathlib import Path

from langchain_core.tools import tool

from config.constants import CHROMA_PERSIST_DIR
from utils.logger import get_agent_logger

log = get_agent_logger("rag_compliance")

# Embedded compliance knowledge (no external PDF needed)
COMPLIANCE_DOCUMENTS = [
    {
        "id": "sec115bbh",
        "title": "Section 115BBH — Tax on VDA Income",
        "content": (
            "Under Section 115BBH of the Income Tax Act, any income from the transfer of "
            "Virtual Digital Assets (VDA) is taxed at a flat rate of 30%, plus applicable "
            "surcharge and 4% health and education cess. No deduction is allowed except the "
            "cost of acquisition. Losses from VDA transfers cannot be set off against any "
            "other income, and such losses cannot be carried forward to subsequent years. "
            "This applies regardless of the investor's income tax slab or holding period."
        ),
    },
    {
        "id": "sec194s",
        "title": "Section 194S — TDS on VDA Transfers",
        "content": (
            "Under Section 194S, Tax Deducted at Source (TDS) at the rate of 1% is applicable "
            "on the transfer of Virtual Digital Assets where the consideration exceeds INR 50,000 "
            "in a financial year (INR 10,000 for specified persons). The buyer/exchange is "
            "responsible for deducting TDS before making the payment. This TDS is an advance tax "
            "and can be adjusted against the final tax liability while filing ITR."
        ),
    },
    {
        "id": "fiu_ind",
        "title": "FIU-IND Registration & Compliance",
        "content": (
            "All Virtual Digital Asset Service Providers (VDASPs) operating in India must register "
            "with the Financial Intelligence Unit - India (FIU-IND) under the Prevention of Money "
            "Laundering Act (PMLA). Registered entities must: 1) Perform KYC verification for all "
            "users, 2) Maintain records of all transactions for 5 years, 3) Report Suspicious "
            "Transaction Reports (STRs) to FIU-IND, 4) Implement AML/CFT compliance programs, "
            "5) Monitor transactions for unusual patterns. Non-compliance can result in penalties "
            "and blocking of services for Indian users."
        ),
    },
    {
        "id": "pmla_aml",
        "title": "PMLA & AML Requirements for Crypto",
        "content": (
            "The Prevention of Money Laundering Act (PMLA) 2002 applies to crypto transactions "
            "in India. Key requirements: 1) Customer Due Diligence (CDD) for all users, "
            "2) Enhanced Due Diligence (EDD) for high-risk customers, 3) Transaction monitoring "
            "to detect suspicious patterns, 4) Record keeping of all transactions for minimum "
            "5 years, 5) Reporting of cash transactions above INR 10 lakhs, 6) Cross-border "
            "wire transfer reporting. Violations can lead to imprisonment up to 7 years and fines."
        ),
    },
    {
        "id": "trading_rules",
        "title": "General Crypto Trading Compliance in India",
        "content": (
            "Legal status: Cryptocurrency is legal to buy, hold, sell, and trade in India but "
            "is NOT legal tender. GST: 18% GST applies on exchange service/platform fees. "
            "Reporting: All crypto income must be reported under Schedule VDA in ITR. "
            "No offsetting: Losses from one VDA cannot be offset against gains from another VDA. "
            "Infrastructure cess: No additional cess on crypto beyond the standard 4% health cess. "
            "Gift tax: VDAs received as gifts are taxable at 30% based on fair market value. "
            "Mining/staking: Income from mining or staking is taxable at applicable slab rates "
            "as business income, and subsequently at 30% on transfer."
        ),
    },
    {
        "id": "risk_limits",
        "title": "Risk Management Guidelines for Algorithmic Trading",
        "content": (
            "Best practices for algorithmic/automated crypto trading in India: "
            "1) Never risk more than 1-2% of portfolio on a single trade, "
            "2) Maintain maximum drawdown limit of 8-10% of portfolio, "
            "3) Implement kill switches that halt trading if drawdown exceeds limit, "
            "4) Keep detailed logs of all automated decisions for regulatory audit, "
            "5) Human oversight required for all trades above a configurable threshold, "
            "6) Paper trading validation before going live, "
            "7) Position sizing based on volatility (ATR-based), "
            "8) Maximum daily trade count limits to prevent overtrading."
        ),
    },
]


def get_compliance_store():
    """Initialize or load the compliance ChromaDB collection."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        collection = client.get_or_create_collection(
            name="compliance_docs",
            metadata={"description": "Indian crypto regulatory documents"},
        )

        # Check if docs are already loaded
        if collection.count() < len(COMPLIANCE_DOCUMENTS):
            log.info("Loading compliance documents into ChromaDB...")
            collection.upsert(
                ids=[d["id"] for d in COMPLIANCE_DOCUMENTS],
                documents=[d["content"] for d in COMPLIANCE_DOCUMENTS],
                metadatas=[{"title": d["title"]} for d in COMPLIANCE_DOCUMENTS],
            )
            log.info(f"Loaded {len(COMPLIANCE_DOCUMENTS)} compliance documents")

        return collection
    except Exception as e:
        log.error(f"ChromaDB init failed: {e}")
        return None


@tool
def query_compliance(question: str) -> str:
    """
    Query Indian crypto compliance regulations using RAG.

    Args:
        question: A regulatory or compliance question about Indian crypto trading.

    Returns:
        Relevant regulatory information from the compliance knowledge base.
    """
    collection = get_compliance_store()
    if collection is None:
        # Fallback: search through docs directly
        return _fallback_search(question)

    try:
        results = collection.query(query_texts=[question], n_results=3)

        if not results or not results.get("documents"):
            return _fallback_search(question)

        docs = results["documents"][0]
        metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)

        response_parts = []
        for doc, meta in zip(docs, metadatas):
            title = meta.get("title", "Regulation")
            response_parts.append(f"**{title}**:\n{doc}")

        response = "\n\n---\n\n".join(response_parts)
        log.info(f"Compliance query answered: '{question[:50]}...'")
        return response

    except Exception as e:
        log.error(f"Compliance query failed: {e}")
        return _fallback_search(question)


def _fallback_search(question: str) -> str:
    """Simple keyword search when ChromaDB is unavailable."""
    question_lower = question.lower()
    matches = []

    for doc in COMPLIANCE_DOCUMENTS:
        content_lower = doc["content"].lower()
        title_lower = doc["title"].lower()
        # Simple keyword match
        words = question_lower.split()
        score = sum(1 for w in words if w in content_lower or w in title_lower)
        if score > 0:
            matches.append((score, doc))

    matches.sort(key=lambda x: x[0], reverse=True)

    if matches:
        parts = [f"**{m[1]['title']}**:\n{m[1]['content']}" for _, m in matches[:3]]
        return "\n\n---\n\n".join(parts)

    return "No specific regulatory information found for this query. Consult a tax professional."


COMPLIANCE_TOOLS = [query_compliance]
