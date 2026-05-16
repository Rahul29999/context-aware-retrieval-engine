"""
embedding.py
------------
Wraps the mock Vertex AI TextEmbeddingModel to provide a clean embedding
interface for the rest of the pipeline.

Design:
  - EmbeddingEngine is a thin adapter around the mock (or real) model.
  - All vectors are L2-normalised so that dot-product == cosine similarity,
    enabling FAISS IndexFlatIP to act as a cosine similarity index.
  - The encoder is loaded once and reused across calls (singleton pattern
    via __init__ caching).

Cosine vs Euclidean — see README.md and retrieval_benchmark.md for the
full rationale.  Short version: cosine similarity is scale-invariant and
works better for embedding spaces where magnitude carries no information.
"""

from __future__ import annotations

import numpy as np
from typing import List

# Import the mock.  In production swap this import for the real SDK:
#   from vertexai.language_models import TextEmbeddingModel
from src.mock_vertexai import TextEmbeddingModel


class EmbeddingEngine:
    """
    Provides text → embedding conversion for both documents and queries.

    Parameters
    ----------
    model_name : str
        Vertex AI model name (used by the mock and would be used in
        production with the real SDK).
    """

    def __init__(self, model_name: str = "textembedding-gecko@003") -> None:
        self.model_name = model_name
        self._model = TextEmbeddingModel.from_pretrained(model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts.

        Returns
        -------
        np.ndarray
            Shape (N, D) float32 array of L2-normalised vectors.
        """
        if not texts:
            raise ValueError("embed() received an empty list of texts.")
        results = self._model.get_embeddings(texts)
        matrix = np.array([r.values for r in results], dtype=np.float32)
        return self._normalise(matrix)

    def embed_single(self, text: str) -> np.ndarray:
        """
        Embed a single string.

        Returns
        -------
        np.ndarray
            Shape (1, D) float32 array.
        """
        return self.embed([text])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(matrix: np.ndarray) -> np.ndarray:
        """L2-normalise each row so dot-product == cosine similarity."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        # Avoid division by zero for zero vectors (edge case)
        norms = np.where(norms == 0, 1.0, norms)
        return matrix / norms

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of this model's output vectors."""
        sample = self.embed(["dimension probe"])
        return sample.shape[1]
