"""
vector_store.py
---------------
A lightweight FAISS-backed vector store that stores chunk text, IDs, and
metadata alongside the embedding index.

Architecture:
  - VectorStore wraps a FAISS IndexFlatIP (inner product on L2-normalised
    vectors == cosine similarity).
  - Metadata (text, id, source, topic) is stored in a parallel Python list.
  - The store is intentionally in-memory for local/test use; serialisation
    helpers (save/load) are provided for persistence.

Why IndexFlatIP?
  Because all vectors are L2-normalised in EmbeddingEngine, inner product
  and cosine similarity are mathematically equivalent.  IndexFlatIP performs
  exact search, which is ideal for corpora of up to ~1M vectors and gives
  perfectly reproducible results for benchmarking.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import faiss  # type: ignore
import numpy as np


@dataclass
class Chunk:
    """A single stored unit of text with its metadata."""

    chunk_id: str
    text: str
    source: str = ""
    topic: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Chunk":
        extra = {k: v for k, v in d.items() if k not in ("chunk_id", "text", "source", "topic")}
        return cls(
            chunk_id=d["chunk_id"],
            text=d["text"],
            source=d.get("source", ""),
            topic=d.get("topic", ""),
            extra=extra,
        )


@dataclass
class SearchResult:
    """A single retrieval result returned by VectorStore.search()."""

    rank: int
    chunk: Chunk
    score: float  # cosine similarity in [−1, 1]; higher is better

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "chunk_id": self.chunk.chunk_id,
            "score": round(float(self.score), 6),
            "text": self.chunk.text,
            "source": self.chunk.source,
            "topic": self.chunk.topic,
        }


class VectorStore:
    """
    In-memory FAISS vector store with full metadata retention.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of the embedding vectors that will be added.
    """

    def __init__(self, embedding_dim: int) -> None:
        self.embedding_dim = embedding_dim
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(embedding_dim)
        self._chunks: List[Chunk] = []

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add(self, vectors: np.ndarray, chunks: List[Chunk]) -> None:
        """
        Add vectors and their corresponding Chunk objects to the store.

        Parameters
        ----------
        vectors : np.ndarray
            Shape (N, D) float32, L2-normalised.
        chunks : List[Chunk]
            Must have len == N.
        """
        if len(vectors) != len(chunks):
            raise ValueError(
                f"vectors and chunks must have the same length "
                f"(got {len(vectors)} vs {len(chunks)})."
            )
        if vectors.ndim != 2 or vectors.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected vectors of shape (N, {self.embedding_dim}), "
                f"got {vectors.shape}."
            )
        vectors_f32 = np.ascontiguousarray(vectors, dtype=np.float32)
        self._index.add(vectors_f32)
        self._chunks.extend(chunks)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[SearchResult]:
        """
        Return the top-k most similar chunks for a query vector.

        Parameters
        ----------
        query_vector : np.ndarray
            Shape (1, D) or (D,) float32, L2-normalised.
        top_k : int
            Number of results to return.

        Returns
        -------
        List[SearchResult]
            Ordered by descending similarity score.
        """
        if self._index.ntotal == 0:
            raise RuntimeError("VectorStore is empty.  Call add() first.")

        q = np.ascontiguousarray(query_vector, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)

        effective_k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q, effective_k)

        results: List[SearchResult] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0:  # FAISS returns −1 for unfilled slots
                continue
            results.append(
                SearchResult(rank=rank, chunk=self._chunks[idx], score=float(score))
            )
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """Persist the FAISS index and metadata to disk."""
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self._index, os.path.join(directory, "index.faiss"))
        meta = [c.to_dict() for c in self._chunks]
        with open(os.path.join(directory, "chunks.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

    @classmethod
    def load(cls, directory: str, embedding_dim: int) -> "VectorStore":
        """Load a previously saved VectorStore from disk."""
        index_path = os.path.join(directory, "index.faiss")
        meta_path = os.path.join(directory, "chunks.json")
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"No saved store found in {directory!r}.")

        store = cls(embedding_dim=embedding_dim)
        store._index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        store._chunks = [Chunk.from_dict(d) for d in meta]
        return store

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of vectors currently stored."""
        return self._index.ntotal

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"VectorStore(dim={self.embedding_dim}, "
            f"stored={self.size})"
        )
