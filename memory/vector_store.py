"""
DCX-AgenticTrader — Vector Store (ChromaDB)

Manages ChromaDB collections for:
- trade_memory: Past trade decisions + outcomes for reflection
- compliance_docs: Indian regulatory documents for RAG
- market_context: Key market events / news summaries
"""

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
        store.store_trade_memory(trade_id, reasoning, outcome)
        similar = store.recall_similar_trades("BTC dropping after fed meeting", k=3)
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
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Store a trade decision + outcome for future reflection.

        Args:
            trade_id: Unique trade identifier.
            context: Market context at the time of decision.
            decision: What action was taken and why.
            outcome: Result of the trade (PnL, fill, etc.).
            metadata: Additional metadata dict.
        """
        collection = self._get_collection("trade_memory")
        document = f"Context: {context}\nDecision: {decision}\nOutcome: {outcome}"

        meta = metadata or {}
        meta["trade_id"] = trade_id

        collection.upsert(ids=[trade_id], documents=[document], metadatas=[meta])
        log.debug(f"Stored trade memory: {trade_id}")

    def recall_similar_trades(self, current_context: str, k: int = 5) -> List[Dict]:
        """
        Retrieve past trades with similar market context.

        Args:
            current_context: Current market state description.
            k: Number of similar trades to retrieve.

        Returns:
            List of past trade memory dicts.
        """
        collection = self._get_collection("trade_memory")
        if collection.count() == 0:
            return []

        results = collection.query(query_texts=[current_context], n_results=min(k, collection.count()))

        memories = []
        if results and results.get("documents"):
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                memories.append({"document": doc, "metadata": meta})

        log.debug(f"Recalled {len(memories)} similar trades")
        return memories

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
