"""
DCX-AgenticTrader — RAG Compliance Tool

RAG (Retrieval Augmented Generation) over Indian crypto regulatory docs.
Loads compliance documents into ChromaDB and answers regulatory questions.
"""

import json
import os
import shutil
from typing import List, Dict, Any, Tuple
from pathlib import Path

from langchain_core.tools import tool
from rank_bm25 import BM25Okapi

from config.constants import CHROMA_PERSIST_DIR
from utils.logger import get_agent_logger

log = get_agent_logger("rag_compliance")

# Re-chunked compliance knowledge with rich metadata
COMPLIANCE_DOCUMENTS = [
    {
        "id": "sec115bbh_rate",
        "title": "Section 115BBH — Flat Tax Rate",
        "content": "Under Section 115BBH of the Income Tax Act, any income from the transfer of Virtual Digital Assets (VDA) is taxed at a flat rate of 30%, plus applicable surcharge and 4% health and education cess.",
        "metadata": {"regulation_type": "tax", "section": "115BBH", "keywords": "tax, 30%, flat rate, cess, income tax"}
    },
    {
        "id": "sec115bbh_deductions",
        "title": "Section 115BBH — Deductions and Losses",
        "content": "No deduction is allowed except the cost of acquisition. Losses from VDA transfers cannot be set off against any other income, and such losses cannot be carried forward to subsequent years. This applies regardless of the investor's income tax slab or holding period.",
        "metadata": {"regulation_type": "tax", "section": "115BBH", "keywords": "deductions, losses, carry forward, set off, acquisition"}
    },
    {
        "id": "sec194s_tds",
        "title": "Section 194S — TDS on VDA",
        "content": "Under Section 194S, Tax Deducted at Source (TDS) at the rate of 1% is applicable on the transfer of Virtual Digital Assets where the consideration exceeds INR 50,000 in a financial year (INR 10,000 for specified persons).",
        "metadata": {"regulation_type": "tax", "section": "194S", "keywords": "tds, 1%, 50000, 10000, threshold"}
    },
    {
        "id": "sec194s_process",
        "title": "Section 194S — TDS Process",
        "content": "The buyer/exchange is responsible for deducting TDS before making the payment. This TDS is an advance tax and can be adjusted against the final tax liability while filing ITR.",
        "metadata": {"regulation_type": "tax", "section": "194S", "keywords": "tds, advance tax, itr, exchange, buyer"}
    },
    {
        "id": "fiu_ind_registration",
        "title": "FIU-IND Registration",
        "content": "All Virtual Digital Asset Service Providers (VDASPs) operating in India must register with the Financial Intelligence Unit - India (FIU-IND) under the Prevention of Money Laundering Act (PMLA).",
        "metadata": {"regulation_type": "compliance", "agency": "FIU-IND", "keywords": "fiu-ind, registration, pmla, vdasp"}
    },
    {
        "id": "fiu_ind_duties",
        "title": "FIU-IND Duties",
        "content": "Registered entities must: 1) Perform KYC verification for all users, 2) Maintain records of all transactions for 5 years, 3) Report Suspicious Transaction Reports (STRs) to FIU-IND, 4) Implement AML/CFT compliance programs, 5) Monitor transactions for unusual patterns.",
        "metadata": {"regulation_type": "compliance", "agency": "FIU-IND", "keywords": "kyc, records, 5 years, str, aml, cft"}
    },
    {
        "id": "pmla_aml",
        "title": "PMLA & AML Requirements",
        "content": "The Prevention of Money Laundering Act (PMLA) 2002 applies to crypto transactions in India. Key requirements include Customer Due Diligence (CDD), Enhanced Due Diligence (EDD) for high-risk customers, and reporting of cash transactions above INR 10 lakhs.",
        "metadata": {"regulation_type": "compliance", "act": "PMLA", "keywords": "pmla, aml, cdd, edd, 10 lakhs, cash"}
    },
    {
        "id": "trading_rules_general",
        "title": "General Trading Rules",
        "content": "Legal status: Cryptocurrency is legal to buy, hold, sell, and trade in India but is NOT legal tender. GST: 18% GST applies on exchange service/platform fees. Reporting: All crypto income must be reported under Schedule VDA in ITR.",
        "metadata": {"regulation_type": "general", "keywords": "legal, tender, gst, 18%, itr, schedule vda"}
    },
    {
        "id": "trading_rules_gifts_mining",
        "title": "Gifts and Mining",
        "content": "Gift tax: VDAs received as gifts are taxable at 30% based on fair market value. Mining/staking: Income from mining or staking is taxable at applicable slab rates as business income, and subsequently at 30% on transfer.",
        "metadata": {"regulation_type": "tax", "keywords": "gift, mining, staking, business income, 30%"}
    },
    {
        "id": "risk_limits_algo",
        "title": "Risk Management for Algo Trading",
        "content": "Best practices for automated crypto trading in India: 1) Never risk more than 1-2% of portfolio on a single trade, 2) Maintain maximum drawdown limit of 8-10%, 3) Implement kill switches, 4) Keep detailed logs for regulatory audit.",
        "metadata": {"regulation_type": "risk", "keywords": "algo, automated, drawdown, kill switch, audit, logs"}
    }
]

# Global cache for BM25 to avoid re-tokenization
_BM25_CORPUS = None
_BM25_DOC_IDS = None


def get_bm25_index() -> Tuple[BM25Okapi, List[str]]:
    """Initialize in-memory BM25 index."""
    global _BM25_CORPUS, _BM25_DOC_IDS
    if _BM25_CORPUS is None:
        tokenized_corpus = [doc["content"].lower().split() for doc in COMPLIANCE_DOCUMENTS]
        _BM25_CORPUS = BM25Okapi(tokenized_corpus)
        _BM25_DOC_IDS = [doc["id"] for doc in COMPLIANCE_DOCUMENTS]
    return _BM25_CORPUS, _BM25_DOC_IDS


def get_compliance_store():
    """Initialize or load the compliance ChromaDB collection with bge-small-en-v1.5."""
    try:
        import chromadb
        import chromadb.utils.embedding_functions as embedding_functions
        
        # BAAI/bge-small-en-v1.5 is excellent for regulatory text
        bge_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-small-en-v1.5"
        )
        
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        
        collection_name = "compliance_docs_v2" # Version bump forces re-indexing
        
        try:
            collection = client.get_collection(name=collection_name, embedding_function=bge_ef)
        except Exception:
            collection = client.create_collection(
                name=collection_name, 
                embedding_function=bge_ef,
                metadata={"description": "Indian crypto regulatory documents (v2)"}
            )

        # Check if docs are already loaded
        if collection.count() < len(COMPLIANCE_DOCUMENTS):
            log.info("Loading compliance documents into ChromaDB (bge-small-en-v1.5)...")
            collection.upsert(
                ids=[d["id"] for d in COMPLIANCE_DOCUMENTS],
                documents=[d["content"] for d in COMPLIANCE_DOCUMENTS],
                metadatas=[{
                    "title": d["title"],
                    **d.get("metadata", {})
                } for d in COMPLIANCE_DOCUMENTS],
            )
            log.info(f"Loaded {len(COMPLIANCE_DOCUMENTS)} compliance documents")

        return collection
    except Exception as e:
        log.error(f"ChromaDB init failed: {e}")
        return None


def rrf_score(vector_ranks: Dict[str, int], bm25_ranks: Dict[str, int], k=60) -> List[Tuple[str, float]]:
    """Reciprocal Rank Fusion to combine vector and BM25 search results."""
    scores = {}
    all_docs = set(vector_ranks.keys()) | set(bm25_ranks.keys())
    
    for doc_id in all_docs:
        v_rank = vector_ranks.get(doc_id, 1000)
        b_rank = bm25_ranks.get(doc_id, 1000)
        # RRF formula: 1 / (k + rank)
        score = (1.0 / (k + v_rank)) + (1.0 / (k + b_rank))
        scores[doc_id] = score
        
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


@tool
def query_compliance(question: str) -> str:
    """
    Query Indian crypto compliance regulations using Hybrid RAG (BM25 + Vector).

    Args:
        question: A regulatory or compliance question about Indian crypto trading.

    Returns:
        Relevant regulatory information from the compliance knowledge base.
    """
    collection = get_compliance_store()
    if collection is None:
        return _fallback_search(question)

    try:
        # 1. Vector Search
        vector_results = collection.query(query_texts=[question], n_results=5)
        vector_ranks = {}
        if vector_results and vector_results.get("ids") and len(vector_results["ids"]) > 0:
            for rank, doc_id in enumerate(vector_results["ids"][0]):
                vector_ranks[doc_id] = rank + 1
                
        # 2. BM25 Search
        bm25, doc_ids = get_bm25_index()
        tokenized_query = question.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        # Sort by score descending
        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:5]
        bm25_ranks = {doc_ids[idx]: rank + 1 for rank, idx in enumerate(top_bm25_indices) if bm25_scores[idx] > 0}
        
        # 3. Hybrid Fusion (RRF)
        fused = rrf_score(vector_ranks, bm25_ranks)
        top_ids = [doc_id for doc_id, score in fused[:3]]
        
        if not top_ids:
            return _fallback_search(question)
            
        # Reconstruct response
        doc_map = {d["id"]: d for d in COMPLIANCE_DOCUMENTS}
        response_parts = []
        for doc_id in top_ids:
            doc = doc_map.get(doc_id)
            if doc:
                response_parts.append(f"**{doc['title']}**:\n{doc['content']}")

        response = "\n\n---\n\n".join(response_parts)
        log.info(f"Hybrid RAG query answered: '{question[:50]}...'")
        return response

    except Exception as e:
        log.error(f"Hybrid RAG query failed: {e}")
        return _fallback_search(question)


def _fallback_search(question: str) -> str:
    """Simple keyword search when ChromaDB is unavailable."""
    question_lower = question.lower()
    matches = []

    for doc in COMPLIANCE_DOCUMENTS:
        content_lower = doc["content"].lower()
        title_lower = doc["title"].lower()
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
