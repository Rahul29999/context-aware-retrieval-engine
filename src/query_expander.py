"""
query_expander.py
-----------------
Uses the mock GenerativeModel to rewrite a user query into a richer,
more retrieval-friendly form before embedding.

In production this module would:
  1. Initialise vertexai with a GCP project and region.
  2. Instantiate the real GenerativeModel("gemini-1.5-pro").
  3. Send the expansion prompt and parse the response.

The expansion prompt is crafted to instruct the model to:
  - Identify the core intent of the query.
  - Add synonyms, related technical terms, and contextual keywords.
  - Return only the expanded query string (no preamble).
"""

from __future__ import annotations

# Swap this import for the real SDK in production:
#   from vertexai.generative_models import GenerativeModel
from src.mock_vertexai import GenerativeModel

_EXPANSION_PROMPT_TEMPLATE = """\
You are a search query optimisation assistant.
Your task is to rewrite the user query into an expanded form that is more \
suitable for semantic vector search.

Rules:
- Add synonyms, related technical terms, and relevant context.
- Do NOT change the core intent.
- Return ONLY the expanded query text; no explanation, no bullet points.

Query: {query}"""


class QueryExpander:
    """
    Expands a raw user query into a richer retrieval query.

    Parameters
    ----------
    model_name : str
        Generative model identifier (used by mock; would be used by real SDK).
    """

    def __init__(self, model_name: str = "gemini-1.5-pro") -> None:
        self.model_name = model_name
        self._model = GenerativeModel(model_name)

    def expand(self, query: str) -> str:
        """
        Rewrite *query* into an expanded form.

        Parameters
        ----------
        query : str
            The original user query.

        Returns
        -------
        str
            Expanded query string.  Falls back to the original if the model
            returns an empty response.
        """
        if not query.strip():
            raise ValueError("Query must not be empty.")

        prompt = _EXPANSION_PROMPT_TEMPLATE.format(query=query.strip())
        response = self._model.generate_content(prompt)
        expanded = response.text.strip()

        # Defensive fallback
        return expanded if expanded else query

    def expand_batch(self, queries: list[str]) -> list[str]:
        """Expand multiple queries.  Returns results in the same order."""
        return [self.expand(q) for q in queries]
