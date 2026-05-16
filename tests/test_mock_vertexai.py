"""
tests/test_mock_vertexai.py
---------------------------
Tests for the mock Vertex AI SDK classes:
  - TextEmbeddingModel  (embedding mock)
  - GenerativeModel     (query expansion mock)

Verifies that the mocks faithfully simulate the SDK interface and that
query expansion is deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.mock_vertexai import (
    GenerativeModel,
    TextEmbeddingModel,
    _deterministic_expand,
)


class TestMockTextEmbeddingModel:

    def test_from_pretrained_returns_instance(self):
        model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        assert isinstance(model, TextEmbeddingModel)

    def test_get_embeddings_returns_list(self):
        model = TextEmbeddingModel.from_pretrained()
        results = model.get_embeddings(["hello", "world"])
        assert isinstance(results, list)
        assert len(results) == 2

    def test_embedding_result_has_values_attribute(self):
        model = TextEmbeddingModel.from_pretrained()
        results = model.get_embeddings(["test sentence"])
        assert hasattr(results[0], "values")
        assert isinstance(results[0].values, list)

    def test_embedding_values_are_floats(self):
        model = TextEmbeddingModel.from_pretrained()
        results = model.get_embeddings(["embedding type check"])
        for val in results[0].values:
            assert isinstance(val, float)

    def test_embedding_dimension_consistent(self):
        model = TextEmbeddingModel.from_pretrained()
        r1 = model.get_embeddings(["first sentence"])
        r2 = model.get_embeddings(["second sentence", "third sentence"])
        assert len(r1[0].values) == len(r2[0].values) == len(r2[1].values)

    def test_embedding_is_deterministic(self):
        model = TextEmbeddingModel.from_pretrained()
        text = ["deterministic embedding test"]
        v1 = model.get_embeddings(text)[0].values
        v2 = model.get_embeddings(text)[0].values
        assert v1 == v2

    def test_different_texts_give_different_embeddings(self):
        model = TextEmbeddingModel.from_pretrained()
        v1 = model.get_embeddings(["auto scaling kubernetes"])[0].values
        v2 = model.get_embeddings(["transformer attention mechanism"])[0].values
        assert v1 != v2

    def test_batch_length_matches_input(self):
        model = TextEmbeddingModel.from_pretrained()
        texts = ["a", "b", "c", "d", "e"]
        results = model.get_embeddings(texts)
        assert len(results) == 5

    def test_model_name_stored(self):
        model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
        assert model.model_name == "textembedding-gecko@001"


class TestMockGenerativeModel:

    def test_generate_content_returns_response_with_text(self):
        model = GenerativeModel("gemini-1.5-pro")
        response = model.generate_content("Query: How does the system handle peak load?")
        assert hasattr(response, "text")
        assert isinstance(response.text, str)
        assert len(response.text) > 0

    def test_model_name_stored(self):
        model = GenerativeModel("gemini-2.0-flash")
        assert model.model_name == "gemini-2.0-flash"

    def test_expansion_contains_relevant_terms_for_peak_load(self):
        model = GenerativeModel()
        response = model.generate_content("Query: How does the system handle peak load?")
        text = response.text.lower()
        # Should contain at least one scalability-related term
        relevant = {"scal", "load", "traffic", "kubernetes", "horizontal", "auto"}
        assert any(term in text for term in relevant)

    def test_expansion_contains_relevant_terms_for_vector_search(self):
        model = GenerativeModel()
        response = model.generate_content("Query: What are the best strategies for semantic vector search?")
        text = response.text.lower()
        relevant = {"vector", "embed", "similarity", "faiss", "cosine", "search"}
        assert any(term in text for term in relevant)

    def test_expansion_contains_relevant_terms_for_rag(self):
        model = GenerativeModel()
        response = model.generate_content("Query: Explain how the RAG pipeline ingests and retrieves documents")
        text = response.text.lower()
        relevant = {"rag", "retrieval", "chunk", "embed", "grounding", "language"}
        assert any(term in text for term in relevant)

    def test_expansion_is_deterministic(self):
        model = GenerativeModel()
        prompt = "Query: How does the system handle peak load?"
        r1 = model.generate_content(prompt).text
        r2 = model.generate_content(prompt).text
        assert r1 == r2

    def test_unknown_query_falls_back_gracefully(self):
        model = GenerativeModel()
        response = model.generate_content("Query: What is the boiling point of water?")
        assert len(response.text) > 0  # fallback produces something

    def test_expansion_differs_from_original(self):
        model = GenerativeModel()
        original = "How does the system handle peak load?"
        response = model.generate_content(f"Query: {original}")
        # Expansion should be richer than the original query
        assert len(response.text) > len(original)


class TestDeterministicExpand:

    def test_peak_load_variant(self):
        result = _deterministic_expand("How does the system handle peak load?")
        assert "scal" in result.lower() or "load" in result.lower()

    def test_vector_search_variant(self):
        result = _deterministic_expand("semantic vector search strategies")
        assert "vector" in result.lower() or "embed" in result.lower()

    def test_rag_variant(self):
        result = _deterministic_expand("explain the RAG pipeline")
        assert "retrieval" in result.lower() or "rag" in result.lower()

    def test_unknown_query_returns_non_empty(self):
        result = _deterministic_expand("random unrelated query xyz")
        assert len(result) > 0

    def test_determinism_across_calls(self):
        q = "peak load handling"
        assert _deterministic_expand(q) == _deterministic_expand(q)
