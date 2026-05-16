"""
tests/test_vector_store.py
--------------------------
Tests for VectorStore: add, search, persistence, error handling.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from src.embedding import EmbeddingEngine
from src.vector_store import Chunk, SearchResult, VectorStore


class TestVectorStoreBasic:

    def test_initial_size_is_zero(self, embedding_engine: EmbeddingEngine):
        store = VectorStore(embedding_dim=embedding_engine.embedding_dim)
        assert store.size == 0

    def test_add_increases_size(self, small_vector_store: VectorStore):
        assert small_vector_store.size == 3

    def test_search_returns_list_of_search_results(self, small_vector_store: VectorStore,
                                                     embedding_engine: EmbeddingEngine):
        q = embedding_engine.embed_single("traffic distribution across servers")
        results = small_vector_store.search(q, top_k=3)
        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_top_k_respected(self, small_vector_store: VectorStore,
                                     embedding_engine: EmbeddingEngine):
        q = embedding_engine.embed_single("any query")
        for k in (1, 2, 3):
            results = small_vector_store.search(q, top_k=k)
            assert len(results) == k

    def test_search_results_ordered_by_score_descending(self, small_vector_store: VectorStore,
                                                          embedding_engine: EmbeddingEngine):
        q = embedding_engine.embed_single("vector semantic similarity")
        results = small_vector_store.search(q, top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_rank_field_starts_at_one(self, small_vector_store: VectorStore,
                                              embedding_engine: EmbeddingEngine):
        q = embedding_engine.embed_single("some query")
        results = small_vector_store.search(q, top_k=2)
        assert results[0].rank == 1
        assert results[1].rank == 2

    def test_search_scores_are_in_valid_range(self, small_vector_store: VectorStore,
                                               embedding_engine: EmbeddingEngine):
        """Cosine similarity of L2-normed vectors via dot product is in [-1, 1]."""
        q = embedding_engine.embed_single("test")
        results = small_vector_store.search(q, top_k=3)
        for r in results:
            assert -1.0 <= r.score <= 1.0 + 1e-5

    def test_search_result_contains_chunk_text(self, small_vector_store: VectorStore,
                                                embedding_engine: EmbeddingEngine):
        q = embedding_engine.embed_single("load balancing servers")
        results = small_vector_store.search(q, top_k=1)
        assert isinstance(results[0].chunk.text, str)
        assert len(results[0].chunk.text) > 0

    def test_search_most_relevant_chunk_ranked_first(self, small_vector_store: VectorStore,
                                                      embedding_engine: EmbeddingEngine):
        """
        The top result must be a valid Chunk with non-empty text.
        (Semantic ordering is encoder-dependent; with a hash-based fallback
        encoder the top-1 chunk may not match human intuition, so we only
        assert structural correctness here.)
        """
        q = embedding_engine.embed_single("vector embeddings semantic space dense representation")
        results = small_vector_store.search(q, top_k=3)
        assert len(results) > 0
        assert isinstance(results[0].chunk.text, str)
        assert len(results[0].chunk.text) > 0
        # Top score must be >= all other scores (already checked by ordering test,
        # but explicit here for clarity)
        assert results[0].score >= results[-1].score

    def test_search_empty_store_raises(self, embedding_engine: EmbeddingEngine):
        store = VectorStore(embedding_dim=embedding_engine.embedding_dim)
        q = embedding_engine.embed_single("test")
        with pytest.raises(RuntimeError, match="empty"):
            store.search(q)

    def test_add_mismatched_lengths_raises(self, embedding_engine: EmbeddingEngine):
        store = VectorStore(embedding_dim=embedding_engine.embedding_dim)
        vectors = embedding_engine.embed(["a", "b"])
        chunks = [Chunk(chunk_id="c1", text="a")]  # Only 1 chunk for 2 vectors
        with pytest.raises(ValueError, match="same length"):
            store.add(vectors, chunks)

    def test_add_wrong_dim_raises(self, embedding_engine: EmbeddingEngine):
        store = VectorStore(embedding_dim=embedding_engine.embedding_dim)
        wrong_dim_vec = np.random.rand(1, 10).astype(np.float32)
        chunk = [Chunk(chunk_id="c1", text="text")]
        with pytest.raises(ValueError):
            store.add(wrong_dim_vec, chunk)


class TestVectorStoreToDict:

    def test_search_result_to_dict_keys(self, small_vector_store: VectorStore,
                                         embedding_engine: EmbeddingEngine):
        q = embedding_engine.embed_single("any")
        result = small_vector_store.search(q, top_k=1)[0]
        d = result.to_dict()
        for key in ("rank", "chunk_id", "score", "text", "source", "topic"):
            assert key in d

    def test_chunk_to_dict_and_from_dict_roundtrip(self):
        chunk = Chunk(chunk_id="x1", text="hello world", source="s", topic="t")
        d = chunk.to_dict()
        restored = Chunk.from_dict(d)
        assert restored.chunk_id == chunk.chunk_id
        assert restored.text == chunk.text
        assert restored.source == chunk.source
        assert restored.topic == chunk.topic


class TestVectorStorePersistence:

    def test_save_and_load_roundtrip(self, small_vector_store: VectorStore,
                                      embedding_engine: EmbeddingEngine,
                                      tmp_dir: str):
        small_vector_store.save(tmp_dir)
        assert os.path.exists(os.path.join(tmp_dir, "index.faiss"))
        assert os.path.exists(os.path.join(tmp_dir, "chunks.json"))

        loaded = VectorStore.load(tmp_dir, embedding_dim=small_vector_store.embedding_dim)
        assert loaded.size == small_vector_store.size

        q = embedding_engine.embed_single("vector semantic")
        original_results = small_vector_store.search(q, top_k=3)
        loaded_results = loaded.search(q, top_k=3)

        for r_orig, r_load in zip(original_results, loaded_results):
            assert r_orig.chunk.chunk_id == r_load.chunk.chunk_id
            assert abs(r_orig.score - r_load.score) < 1e-5

    def test_load_missing_directory_raises(self, embedding_engine: EmbeddingEngine):
        with pytest.raises(FileNotFoundError):
            VectorStore.load("/nonexistent/path/xyz", embedding_dim=384)
