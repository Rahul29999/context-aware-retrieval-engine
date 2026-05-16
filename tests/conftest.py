"""
conftest.py
-----------
Shared pytest fixtures used across all test modules.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

from src.embedding import EmbeddingEngine
from src.pipeline import RAGPipeline
from src.query_expander import QueryExpander
from src.vector_store import Chunk, VectorStore


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
SAMPLE_JSON = DATA_DIR / "sample_texts.json"


# ---------------------------------------------------------------------------
# Shared component fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def embedding_engine() -> EmbeddingEngine:
    """A single EmbeddingEngine reused across the session to avoid re-loading."""
    return EmbeddingEngine()


@pytest.fixture(scope="session")
def query_expander() -> QueryExpander:
    return QueryExpander()


@pytest.fixture(scope="session")
def populated_pipeline() -> RAGPipeline:
    """A fully ingested pipeline reused across the session."""
    pipeline = RAGPipeline()
    pipeline.ingest(SAMPLE_JSON)
    return pipeline


@pytest.fixture()
def small_vector_store(embedding_engine: EmbeddingEngine) -> VectorStore:
    """A VectorStore pre-populated with 3 chunks for fast unit tests."""
    texts = [
        "Load balancing distributes traffic across multiple servers.",
        "Vector embeddings capture semantic meaning in dense space.",
        "RAG combines retrieval with language model generation.",
    ]
    vectors = embedding_engine.embed(texts)
    dim = vectors.shape[1]
    store = VectorStore(embedding_dim=dim)
    chunks = [
        Chunk(chunk_id=f"c{i}", text=t, source="test", topic=f"topic_{i}")
        for i, t in enumerate(texts)
    ]
    store.add(vectors, chunks)
    return store


@pytest.fixture()
def tmp_dir() -> Generator[str, None, None]:
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture()
def tiny_json_dataset(tmp_dir: str) -> str:
    """Write a minimal JSON dataset to a temp file and return its path."""
    records = [
        {"id": "t1", "text": "Horizontal scaling adds more machines.", "source": "test", "topic": "scaling"},
        {"id": "t2", "text": "Cosine similarity measures vector angle.", "source": "test", "topic": "math"},
        {"id": "t3", "text": "FAISS provides efficient approximate search.", "source": "test", "topic": "faiss"},
    ]
    path = os.path.join(tmp_dir, "tiny.json")
    with open(path, "w") as fh:
        json.dump(records, fh)
    return path
