import json
from typing import Dict, Any, List, Optional
from pathlib import Path

from config.constants import CHROMA_PERSIST_DIR
from utils.logger import get_agent_logger

log = get_agent_logger("vector_store")


class VectorStore:
    """
    ChromaDB wrapper for semantic memory across the trading system.

    Usage:
        store = VectorStore()
        store.store_trade_memory(trade_id, context, decision, outcome)
        similar = store.recall_similar_trades_text("BTC dropping after fed meeting", k=3)
    """

    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR):
        self.persist_dir = persist_dir
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = None
        self._collections = {}

    @property
    def client(self):
        if self._client is None:
            import chromadb
            # Use default embeddings for trade memory (lightweight)
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            log.info(f"ChromaDB initialized at {self.persist_dir}")
        return self._client

    def _get_collection(self, name: str):
        if name not in self._collections:
            self._collections[name] = self.client.get_or_create_collection(name=name)
        return self._collections[name]

    # =========================================================================
    # Trade Memory
    # =========================================================================

    def store_trade_memory(
        self,
        trade_id: str,
        context: str,
        decision: str,
        outcome: str,
        rationale: str = "",
        risk_status: str = "",
        sentiment: str = "",
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Store a trade decision + outcome for future reflection.

        Args:
            trade_id: Unique trade identifier.
            context: Market context at the time of decision (price, tech indicators).
            decision: What action was taken (e.g. BUY, SELL, HOLD).
            outcome: Result of the trade (PnL, fill, etc.).
            rationale: The LLM or deterministic reasoning.
            risk_status: Risk compliance status (PASS/FAIL).
            sentiment: Overall sentiment composite score/label.
            metadata: Additional metadata dict.
        """
        collection = self._get_collection("trade_memory_v2")
        
        # Build a richer document for better semantic search and LLM context
        document = (
            f"Trade ID: {trade_id}\n"
            f"Context: {context}\n"
            f"Sentiment: {sentiment}\n"
            f"Risk: {risk_status}\n"
            f"Decision: {decision}\n"
            f"Rationale: {rationale}\n"
            f"Outcome: {outcome}"
        )

        meta = metadata or {}
        meta["trade_id"] = trade_id
        meta["decision"] = decision
        meta["risk"] = risk_status

        collection.upsert(ids=[trade_id], documents=[document], metadatas=[meta])
        log.debug(f"Stored enhanced trade memory: {trade_id}")

    def recall_similar_trades(self, current_context: str, k: int = 5) -> List[Dict]:
        """
        Retrieve past trades with similar market context.

        Args:
            current_context: Current market state description.
            k: Number of similar trades to retrieve.

        Returns:
            List of past trade memory dicts.
        """
        collection = self._get_collection("trade_memory_v2")
        if collection.count() == 0:
            return []

        results = collection.query(query_texts=[current_context], n_results=min(k, collection.count()))

        memories = []
        if results and results.get("documents"):
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                memories.append({"document": doc, "metadata": meta})

        log.debug(f"Recalled {len(memories)} similar trades")
        return memories
        
    def recall_similar_trades_text(self, current_context: str, k: int = 3) -> str:
        """
        Retrieve past trades and format them as a string for LLM injection.
        """
        memories = self.recall_similar_trades(current_context, k)
        if not memories:
            return "No similar historical trades found."
            
        formatted = []
        for i, m in enumerate(memories):
            formatted.append(f"--- Past Trade {i+1} ---\n{m['document']}")
            
        return "\n\n".join(formatted)

    # =========================================================================
    # Market Context
    # =========================================================================

    def store_market_event(self, event_id: str, description: str, metadata: Optional[Dict] = None) -> None:
        """Store a significant market event for future reference."""
        collection = self._get_collection("market_context")
        collection.upsert(ids=[event_id], documents=[description], metadatas=[metadata or {}])

    def search_market_context(self, query: str, k: int = 3) -> List[Dict]:
        """Search past market events similar to current context."""
        collection = self._get_collection("market_context")
        if collection.count() == 0:
            return []

        results = collection.query(query_texts=[query], n_results=min(k, collection.count()))
        contexts = []
        if results and results.get("documents"):
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                contexts.append({"document": doc, "metadata": meta})
        return contexts
