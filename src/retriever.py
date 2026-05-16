"""
retriever.py
------------
Implements Strategy A (raw vector search) and Strategy B (AI-enhanced
retrieval with query expansion).

Both strategies share the same VectorStore and EmbeddingEngine; the only
difference is the query pre-processing step.
"""

from __future__ import annotations

from typing import List

from src.embedding import EmbeddingEngine
from src.query_expander import QueryExpander
from src.vector_store import SearchResult, VectorStore


class Retriever:
    """
    Provides two retrieval strategies over a populated VectorStore.

    Parameters
    ----------
    vector_store : VectorStore
        A populated vector store (call pipeline.ingest() first).
    embedding_engine : EmbeddingEngine
        The same engine used during ingestion.
    query_expander : QueryExpander | None
        Required for Strategy B.  If None, Strategy B raises an error.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_engine: EmbeddingEngine,
        query_expander: QueryExpander | None = None,
    ) -> None:
        self._store = vector_store
        self._embedder = embedding_engine
        self._expander = query_expander

    # ------------------------------------------------------------------
    # Strategy A — Raw Vector Search
    # ------------------------------------------------------------------

    def retrieve_strategy_a(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """
        Embed the raw query and search the vector store.

        Parameters
        ----------
        query : str
            The original user query, unchanged.
        top_k : int
            Number of results to return.

        Returns
        -------
        List[SearchResult]
            Ranked by descending cosine similarity.
        """
        vector = self._embedder.embed_single(query)
        return self._store.search(vector, top_k=top_k)

    # ------------------------------------------------------------------
    # Strategy B — AI-Enhanced Retrieval
    # ------------------------------------------------------------------

    def retrieve_strategy_b(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """
        Expand the query via the generative model, then embed and search.

        Parameters
        ----------
        query : str
            The original user query.
        top_k : int
            Number of results to return.

        Returns
        -------
        List[SearchResult]
            Ranked by descending cosine similarity of the *expanded* query.
        """
        if self._expander is None:
            raise RuntimeError(
                "QueryExpander is required for Strategy B.  "
                "Initialise Retriever with a QueryExpander instance."
            )
        expanded_query = self._expander.expand(query)
        vector = self._embedder.embed_single(expanded_query)
        return self._store.search(vector, top_k=top_k)

    def get_expanded_query(self, query: str) -> str:
        """Utility: return the expanded form of *query* without searching."""
        if self._expander is None:
            raise RuntimeError("QueryExpander not configured.")
        return self._expander.expand(query)
