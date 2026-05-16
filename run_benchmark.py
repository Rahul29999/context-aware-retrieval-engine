#!/usr/bin/env python3
"""
run_benchmark.py
----------------
Entry-point script.  Run this to execute the full pipeline demo:
  1. Ingest the sample dataset.
  2. Run the Strategy A vs Strategy B benchmark.
  3. Save results to benchmark_results.json.

Usage:
    python run_benchmark.py
    python run_benchmark.py --dataset data/sample_texts.json --top-k 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG Pipeline Benchmark Runner")
    parser.add_argument(
        "--dataset",
        default="data/sample_texts.json",
        help="Path to the dataset file (.json or .txt). Default: data/sample_texts.json",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of results per query. Default: 3",
    )
    parser.add_argument(
        "--output",
        default="benchmark_results.json",
        help="Output path for the JSON benchmark results.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the formatted table output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    print("=" * 72)
    print("  Context-Aware Retrieval Engine — Semantic RAG & Vector Search")
    print("=" * 72)
    print(f"  Dataset : {dataset_path}")
    print(f"  Top-K   : {args.top_k}")
    print(f"  Output  : {args.output}")
    print("=" * 72)

    # Import here so CLI startup is fast even if imports are slow
    from src.pipeline import RAGPipeline

    pipeline = RAGPipeline(top_k=args.top_k)

    print(f"\n[1/3] Ingesting dataset: {dataset_path} ...")
    pipeline.ingest(dataset_path)
    print(f"      ✓ {pipeline.store_size} chunks embedded and indexed.\n")

    print("[2/3] Running benchmark (Strategy A vs Strategy B) ...\n")
    pipeline.run_benchmark(verbose=not args.quiet)

    print(f"\n[3/3] Saving results to {args.output} ...")
    pipeline.save_benchmark_json(args.output)
    print("      ✓ Done.\n")

    print("=" * 72)
    print("  Benchmark complete.  See retrieval_benchmark.md for analysis.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
