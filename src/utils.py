"""
utils.py
--------
Shared helper utilities for data loading, text processing, and output
formatting used across the pipeline.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_json_dataset(path: str | Path) -> List[Dict[str, Any]]:
    """
    Load a JSON dataset file and return a list of record dicts.

    Expected format:
      [
        {"id": "...", "text": "...", "source": "...", "topic": "..."},
        ...
      ]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}.")
    return data


def load_text_dataset(path: str | Path) -> List[Dict[str, Any]]:
    """
    Load a plain-text dataset where paragraphs are separated by blank lines.
    Returns synthetic record dicts with auto-generated IDs.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    raw = path.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
    return [
        {"id": f"chunk_{i+1:03d}", "text": para, "source": str(path.name), "topic": ""}
        for i, para in enumerate(paragraphs)
    ]


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def truncate(text: str, max_chars: int = 120) -> str:
    """Truncate *text* to *max_chars*, appending '…' if truncated."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def wrap(text: str, width: int = 80, indent: str = "  ") -> str:
    """Wrap *text* to *width* characters with a leading *indent*."""
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def results_to_dict(
    query: str,
    expanded_query: str,
    results_a: list,
    results_b: list,
    comparison_note: str = "",
) -> Dict[str, Any]:
    """
    Serialise a single benchmark entry to a plain dict.

    Parameters
    ----------
    query : str
        Original query.
    expanded_query : str
        Query after expansion (Strategy B).
    results_a : list[SearchResult]
        Strategy A results.
    results_b : list[SearchResult]
        Strategy B results.
    comparison_note : str
        Human-readable comparison note.
    """
    return {
        "original_query": query,
        "expanded_query": expanded_query,
        "strategy_a": [r.to_dict() for r in results_a],
        "strategy_b": [r.to_dict() for r in results_b],
        "comparison_note": comparison_note,
    }


def print_benchmark_table(entry: Dict[str, Any]) -> None:  # pragma: no cover
    """Pretty-print a single benchmark entry to stdout."""
    sep = "─" * 72
    print(f"\n{sep}")
    print(f"  Query   : {entry['original_query']}")
    print(f"  Expanded: {truncate(entry['expanded_query'], 120)}")
    print(sep)
    print(f"  {'Rank':<5} {'Strategy':<12} {'Score':>7}  {'Topic':<20}  Text")
    print(f"  {'----':<5} {'--------':<12} {'-----':>7}  {'-----':<20}  ----")
    for r in entry["strategy_a"]:
        print(
            f"  {r['rank']:<5} {'A (raw)':<12} {r['score']:>7.4f}"
            f"  {r['topic']:<20}  {truncate(r['text'], 60)}"
        )
    for r in entry["strategy_b"]:
        print(
            f"  {r['rank']:<5} {'B (expanded)':<12} {r['score']:>7.4f}"
            f"  {r['topic']:<20}  {truncate(r['text'], 60)}"
        )
    if entry.get("comparison_note"):
        print(f"\n  Note: {entry['comparison_note']}")
    print(sep)
