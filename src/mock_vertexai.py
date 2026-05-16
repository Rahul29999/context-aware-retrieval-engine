"""
mock_vertexai.py
----------------
Deterministic mocks for Vertex AI SDK classes used in this project.

In production these would be replaced by:
  - vertexai.language_models.TextEmbeddingModel  → calls Vertex AI Embeddings API
  - vertexai.generative_models.GenerativeModel   → calls Gemini / PaLM generative API

The mocks below are intentionally deterministic so that tests are stable and
repeatable without network access or GCP credentials.
"""

from __future__ import annotations

import hashlib
from typing import List


# ---------------------------------------------------------------------------
# Embedding mock
# ---------------------------------------------------------------------------

class TextEmbeddingModel:
    """
    Mock replacement for vertexai.language_models.TextEmbeddingModel.

    In production:
      model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
      embeddings = model.get_embeddings(["text one", "text two"])
      vectors = [e.values for e in embeddings]

    This mock delegates to a real sentence-transformers model so that the
    embedding arithmetic is genuine while the SDK interface is simulated.
    """

    _instance: "TextEmbeddingModel | None" = None

    def __init__(self, model_name: str = "textembedding-gecko@003") -> None:
        self.model_name = model_name
        self._encoder = self._load_encoder()

    @staticmethod
    def _load_encoder():
        """
        Try to load sentence-transformers (requires network on first run to
        download the model weights).  If the model is unavailable (no network,
        sandbox environment, etc.) fall back to a deterministic NumPy encoder
        that produces stable hash-based pseudo-embeddings.  The fallback
        preserves all pipeline logic and test determinism; only the semantic
        quality of similarity scores is reduced.
        """
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            return SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            import hashlib
            import numpy as np

            class _NumpyFallbackEncoder:
                """Deterministic 128-dim hash-based encoder (no network required)."""
                DIM = 128

                def encode(self, texts, show_progress_bar=False, normalize_embeddings=True):
                    out = []
                    for t in texts:
                        seed = int(hashlib.md5(t.encode()).hexdigest(), 16) % (2 ** 31)
                        rng = np.random.RandomState(seed)
                        v = rng.randn(self.DIM).astype(np.float32)
                        if normalize_embeddings:
                            norm = np.linalg.norm(v)
                            if norm > 0:
                                v = v / norm
                        out.append(v)
                    return np.array(out)

            return _NumpyFallbackEncoder()

    @classmethod
    def from_pretrained(cls, model_name: str = "textembedding-gecko@003") -> "TextEmbeddingModel":
        """Mirror the Vertex AI SDK factory method."""
        return cls(model_name=model_name)

    def get_embeddings(self, texts: List[str]) -> List["_EmbeddingResult"]:
        """Return a list of embedding result objects, mirroring the SDK return type."""
        vectors = self._encoder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return [_EmbeddingResult(values=v.tolist()) for v in vectors]


class _EmbeddingResult:
    """Mirrors vertexai.language_models.TextEmbedding."""

    def __init__(self, values: List[float]) -> None:
        self.values = values

    def __repr__(self) -> str:  # pragma: no cover
        return f"_EmbeddingResult(dim={len(self.values)})"


# ---------------------------------------------------------------------------
# Generative model mock (for query expansion)
# ---------------------------------------------------------------------------

# Deterministic expansion templates keyed by stable query hash prefix.
# Any query not matching a template falls back to a generic expansion.
_EXPANSION_TEMPLATES: dict[str, str] = {
    # hash prefix → expanded query
    "peak_load": (
        "system performance under high traffic load spikes auto-scaling "
        "horizontal scaling capacity management kubernetes HPA load balancer"
    ),
    "vector_search": (
        "vector embedding similarity search FAISS cosine distance nearest "
        "neighbour retrieval semantic search dense representation"
    ),
    "rag_pipeline": (
        "retrieval augmented generation RAG pipeline ingestion chunking "
        "embedding vector store context grounding language model"
    ),
}

# Mapping of canonical query fragments to template keys (order matters – first match wins)
_QUERY_TO_TEMPLATE: list[tuple[str, str]] = [
    ("peak load", "peak_load"),
    ("high traffic", "peak_load"),
    ("scale", "peak_load"),
    ("vector search", "vector_search"),
    ("embedding", "vector_search"),
    ("similarity", "vector_search"),
    ("rag", "rag_pipeline"),
    ("retrieval augmented", "rag_pipeline"),
    ("retrieval-augmented", "rag_pipeline"),
]


def _deterministic_expand(query: str) -> str:
    """Select and return a deterministic expanded query string."""
    lower = query.lower()
    for fragment, key in _QUERY_TO_TEMPLATE:
        if fragment in lower:
            return _EXPANSION_TEMPLATES[key]
    # Generic expansion: append semantically useful boilerplate
    words = query.strip().rstrip("?")
    return (
        f"{words} concepts principles architecture implementation "
        f"best practices technical details mechanisms"
    )


class GenerativeModel:
    """
    Mock replacement for vertexai.generative_models.GenerativeModel.

    In production:
      model = GenerativeModel("gemini-1.5-pro")
      response = model.generate_content(prompt)
      expanded = response.text

    This mock returns deterministic expansions so tests never depend on a
    live LLM call.
    """

    def __init__(self, model_name: str = "gemini-1.5-pro") -> None:
        self.model_name = model_name

    def generate_content(self, prompt: str) -> "_GenerativeResponse":
        """
        Analyse the prompt to extract the original query, then return a
        deterministic expansion.
        """
        # The prompt template from query_expander.py embeds the query after
        # the last colon on the final non-empty line.  We do a best-effort
        # extraction; fall back to expanding the whole prompt.
        lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()]
        query_line = lines[-1] if lines else prompt
        # Strip common prefixes like "Query: "
        for prefix in ("query:", "original query:", "user query:"):
            if query_line.lower().startswith(prefix):
                query_line = query_line[len(prefix):].strip()
                break

        expanded = _deterministic_expand(query_line)
        return _GenerativeResponse(text=expanded)


class _GenerativeResponse:
    """Mirrors vertexai.generative_models.GenerationResponse."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:  # pragma: no cover
        return f"_GenerativeResponse(text={self.text!r})"
