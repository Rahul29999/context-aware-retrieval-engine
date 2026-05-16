"""
pipeline.py
-----------
Central orchestrator that wires together ingestion, embedding, indexing,
retrieval, and benchmarking into a single cohesive API.

Usage (quick-start):
    from src.pipeline import RAGPipeline

    pipeline = RAGPipeline()
    pipeline.ingest("data/sample_texts.json")
    results = pipeline.run_benchmark()
    pipeline.save_benchmark_json("benchmark_results.json")
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.embedding import EmbeddingEngine
from src.query_expander import QueryExpander
from src.retriever import Retriever
from src.utils import (
    load_json_dataset,
    load_text_dataset,
    print_benchmark_table,
    results_to_dict,
    truncate,
)
from src.vector_store import Chunk, VectorStore


# ---------------------------------------------------------------------------
# Default benchmark queries
# ---------------------------------------------------------------------------

_DEFAULT_QUERIES: List[Dict[str, str]] = [
    {
        "query": "How does the system handle peak load?",
        "note_template": (
            "Strategy {better} surfaced more relevant chunks for peak-load "
            "scalability.  Strategy B typically adds Kubernetes/HPA terms "
            "that improve recall of infrastructure chunks."
        ),
    },
    {
        "query": "What are the best strategies for semantic vector search?",
        "note_template": (
            "Strategy {better} retrieved more on-topic vector-search chunks.  "
            "Expansion adds FAISS/cosine/ANN terminology that increases recall "
            "of the FAISS and embeddings chunks."
        ),
    },
    {
        "query": "Explain how the RAG pipeline ingests and retrieves documents",
        "note_template": (
            "Strategy {better} returned more RAG-specific chunks.  "
            "Expansion keywords such as 'chunking', 'grounding', and "
            "'language model' help surface the RAG architecture chunk."
        ),
    },
]


class RAGPipeline:
    """
    End-to-end RAG pipeline: ingest → embed → index → retrieve → benchmark.

    Parameters
    ----------
    embedding_model : str
        Model identifier forwarded to EmbeddingEngine / mock.
    generative_model : str
        Model identifier forwarded to QueryExpander / mock.
    top_k : int
        Default number of results per query in benchmarking.
    """

    def __init__(
        self,
        embedding_model: str = "textembedding-gecko@003",
        generative_model: str = "gemini-1.5-pro",
        top_k: int = 3,
    ) -> None:
        self.top_k = top_k
        self._embedder = EmbeddingEngine(model_name=embedding_model)
        self._store: Optional[VectorStore] = None
        self._retriever: Optional[Retriever] = None
        self._expander = QueryExpander(model_name=generative_model)
        self._benchmark_results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, dataset_path: str | Path) -> None:
        """
        Load a dataset, embed all chunks, and add them to the vector store.

        Supported formats:
          - .json  → list of objects with keys: id, text, source, topic
          - .txt   → paragraphs separated by blank lines (auto-ID assigned)

        Parameters
        ----------
        dataset_path : str | Path
            Path to the data file.
        """
        path = Path(dataset_path)
        if path.suffix == ".json":
            records = load_json_dataset(path)
        elif path.suffix == ".txt":
            records = load_text_dataset(path)
        else:
            raise ValueError(
                f"Unsupported dataset format: {path.suffix!r}.  "
                "Use .json or .txt."
            )

        if not records:
            raise ValueError(f"Dataset at {path} is empty.")

        # Build Chunk objects
        chunks = [
            Chunk(
                chunk_id=rec.get("id", f"chunk_{i:04d}"),
                text=rec["text"],
                source=rec.get("source", ""),
                topic=rec.get("topic", ""),
            )
            for i, rec in enumerate(records)
        ]

        # Embed all texts in a single batch call
        texts = [c.text for c in chunks]
        vectors = self._embedder.embed(texts)

        # Initialise the vector store on first ingest
        if self._store is None:
            self._store = VectorStore(embedding_dim=vectors.shape[1])

        self._store.add(vectors, chunks)

        # Wire up the retriever
        self._retriever = Retriever(
            vector_store=self._store,
            embedding_engine=self._embedder,
            query_expander=self._expander,
        )

    # ------------------------------------------------------------------
    # Single-query retrieval
    # ------------------------------------------------------------------

    def search(self, query: str, strategy: str = "A", top_k: Optional[int] = None) -> list:
        """
        Run a single search using the specified strategy.

        Parameters
        ----------
        query : str
        strategy : str
            "A" for raw vector search, "B" for AI-enhanced.
        top_k : int | None
            Overrides the pipeline default if provided.

        Returns
        -------
        List[SearchResult]
        """
        self._require_ingestion()
        k = top_k if top_k is not None else self.top_k
        if strategy.upper() == "A":
            return self._retriever.retrieve_strategy_a(query, top_k=k)
        elif strategy.upper() == "B":
            return self._retriever.retrieve_strategy_b(query, top_k=k)
        else:
            raise ValueError(f"Unknown strategy {strategy!r}.  Use 'A' or 'B'.")

    # ------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------

    def run_benchmark(
        self,
        queries: Optional[List[Dict[str, str]]] = None,
        top_k: Optional[int] = None,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Run Strategy A vs Strategy B for each query and collect results.

        Parameters
        ----------
        queries : list of dicts with keys "query" and optionally "note_template"
        top_k : int | None
        verbose : bool
            If True, print a formatted table to stdout.

        Returns
        -------
        List[Dict[str, Any]]
            Structured benchmark results.
        """
        self._require_ingestion()
        if queries is None:
            queries = _DEFAULT_QUERIES
        k = top_k if top_k is not None else self.top_k

        self._benchmark_results = []
        for item in queries:
            query = item["query"]
            note_tmpl = item.get("note_template", "")

            results_a = self._retriever.retrieve_strategy_a(query, top_k=k)
            results_b = self._retriever.retrieve_strategy_b(query, top_k=k)
            expanded_q = self._retriever.get_expanded_query(query)

            # Determine which strategy scored higher on average (for note)
            avg_a = sum(r.score for r in results_a) / len(results_a) if results_a else 0
            avg_b = sum(r.score for r in results_b) / len(results_b) if results_b else 0
            better = "B" if avg_b >= avg_a else "A"

            note = note_tmpl.format(better=better) if note_tmpl else ""

            entry = results_to_dict(
                query=query,
                expanded_query=expanded_q,
                results_a=results_a,
                results_b=results_b,
                comparison_note=note,
            )
            self._benchmark_results.append(entry)

            if verbose:
                print_benchmark_table(entry)

        return self._benchmark_results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_benchmark_json(self, path: str | Path = "benchmark_results.json") -> None:
        """Write the benchmark results to a JSON file."""
        if not self._benchmark_results:
            raise RuntimeError("No benchmark results yet.  Call run_benchmark() first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self._benchmark_results, fh, indent=2)
        print(f"Benchmark JSON saved to {path}")

    def save_store(self, directory: str | Path) -> None:
        """Persist the vector store to disk."""
        self._require_ingestion()
        self._store.save(str(directory))
        print(f"Vector store saved to {directory}")

    def load_store(self, directory: str | Path) -> None:
        """Load a previously saved vector store from disk."""
        dim = self._embedder.embedding_dim
        self._store = VectorStore.load(str(directory), embedding_dim=dim)
        self._retriever = Retriever(
            vector_store=self._store,
            embedding_engine=self._embedder,
            query_expander=self._expander,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _require_ingestion(self) -> None:
        if self._store is None or self._retriever is None:
            raise RuntimeError(
                "No data ingested yet.  Call pipeline.ingest(path) first."
            )

    @property
    def store_size(self) -> int:
        """Number of vectors currently in the store."""
        if self._store is None:
            return 0
        return self._store.size
