"""
tests/test_pipeline.py
----------------------
End-to-end tests for RAGPipeline:
  - Ingestion from JSON and TXT datasets
  - Single-query search (both strategies)
  - Benchmark execution and output format
  - Persistence (save/load store)
  - Error handling for pre-ingestion calls
"""

from __future__ import annotations

import json
import os
import textwrap

import pytest

from src.pipeline import RAGPipeline


class TestIngestion:

    def test_ingest_json_increases_store_size(self, tiny_json_dataset: str):
        pipeline = RAGPipeline()
        pipeline.ingest(tiny_json_dataset)
        assert pipeline.store_size == 3

    def test_ingest_full_sample_dataset(self, populated_pipeline: RAGPipeline):
        assert populated_pipeline.store_size == 10

    def test_ingest_txt_dataset(self, tmp_dir: str):
        txt_path = os.path.join(tmp_dir, "sample.txt")
        content = textwrap.dedent("""\
            Load balancing spreads traffic across servers.

            Embeddings encode semantic meaning as dense vectors.

            RAG combines retrieval with language models.
        """)
        with open(txt_path, "w") as fh:
            fh.write(content)
        pipeline = RAGPipeline()
        pipeline.ingest(txt_path)
        assert pipeline.store_size == 3

    def test_ingest_empty_dataset_raises(self, tmp_dir: str):
        empty_path = os.path.join(tmp_dir, "empty.json")
        with open(empty_path, "w") as fh:
            json.dump([], fh)
        pipeline = RAGPipeline()
        with pytest.raises(ValueError, match="empty"):
            pipeline.ingest(empty_path)

    def test_ingest_unsupported_format_raises(self, tmp_dir: str):
        bad_path = os.path.join(tmp_dir, "data.csv")
        with open(bad_path, "w") as fh:
            fh.write("a,b,c\n")
        pipeline = RAGPipeline()
        with pytest.raises(ValueError, match="Unsupported"):
            pipeline.ingest(bad_path)

    def test_search_before_ingest_raises(self):
        pipeline = RAGPipeline()
        with pytest.raises(RuntimeError, match="No data ingested"):
            pipeline.search("test query")


class TestSearch:

    def test_strategy_a_returns_results(self, populated_pipeline: RAGPipeline):
        results = populated_pipeline.search("peak load auto scaling", strategy="A", top_k=3)
        assert len(results) == 3

    def test_strategy_b_returns_results(self, populated_pipeline: RAGPipeline):
        results = populated_pipeline.search("peak load auto scaling", strategy="B", top_k=3)
        assert len(results) == 3

    def test_invalid_strategy_raises(self, populated_pipeline: RAGPipeline):
        with pytest.raises(ValueError, match="Unknown strategy"):
            populated_pipeline.search("test", strategy="C")

    def test_top_k_override(self, populated_pipeline: RAGPipeline):
        results = populated_pipeline.search("any query", strategy="A", top_k=2)
        assert len(results) == 2

    def test_search_result_has_text(self, populated_pipeline: RAGPipeline):
        results = populated_pipeline.search("vector embeddings", strategy="A", top_k=1)
        assert isinstance(results[0].chunk.text, str)
        assert len(results[0].chunk.text) > 0

    def test_search_result_has_score(self, populated_pipeline: RAGPipeline):
        results = populated_pipeline.search("faiss index", strategy="A", top_k=1)
        assert isinstance(results[0].score, float)

    def test_strategy_a_case_insensitive(self, populated_pipeline: RAGPipeline):
        r1 = populated_pipeline.search("peak load", strategy="A", top_k=3)
        r2 = populated_pipeline.search("peak load", strategy="a", top_k=3)
        ids1 = [r.chunk.chunk_id for r in r1]
        ids2 = [r.chunk.chunk_id for r in r2]
        assert ids1 == ids2


class TestBenchmark:

    def test_benchmark_returns_list(self, populated_pipeline: RAGPipeline):
        results = populated_pipeline.run_benchmark(verbose=False)
        assert isinstance(results, list)

    def test_benchmark_default_has_three_entries(self, populated_pipeline: RAGPipeline):
        results = populated_pipeline.run_benchmark(verbose=False)
        assert len(results) == 3

    def test_benchmark_entry_keys(self, populated_pipeline: RAGPipeline):
        results = populated_pipeline.run_benchmark(verbose=False)
        required_keys = {
            "original_query", "expanded_query",
            "strategy_a", "strategy_b", "comparison_note"
        }
        for entry in results:
            assert required_keys.issubset(entry.keys())

    def test_benchmark_strategy_results_have_three_chunks(self, populated_pipeline: RAGPipeline):
        results = populated_pipeline.run_benchmark(verbose=False)
        for entry in results:
            assert len(entry["strategy_a"]) == 3
            assert len(entry["strategy_b"]) == 3

    def test_benchmark_chunk_entry_keys(self, populated_pipeline: RAGPipeline):
        results = populated_pipeline.run_benchmark(verbose=False)
        chunk_keys = {"rank", "chunk_id", "score", "text", "source", "topic"}
        for entry in results:
            for chunk in entry["strategy_a"] + entry["strategy_b"]:
                assert chunk_keys.issubset(chunk.keys())

    def test_benchmark_expanded_query_differs_from_original(self, populated_pipeline: RAGPipeline):
        results = populated_pipeline.run_benchmark(verbose=False)
        for entry in results:
            assert entry["expanded_query"] != entry["original_query"]

    def test_benchmark_scores_are_floats(self, populated_pipeline: RAGPipeline):
        results = populated_pipeline.run_benchmark(verbose=False)
        for entry in results:
            for chunk in entry["strategy_a"] + entry["strategy_b"]:
                assert isinstance(chunk["score"], float)

    def test_benchmark_custom_queries(self, populated_pipeline: RAGPipeline):
        custom = [
            {"query": "How does FAISS indexing work?", "note_template": ""},
            {"query": "What is the transformer attention mechanism?", "note_template": ""},
        ]
        results = populated_pipeline.run_benchmark(queries=custom, verbose=False)
        assert len(results) == 2

    def test_benchmark_before_ingest_raises(self):
        pipeline = RAGPipeline()
        with pytest.raises(RuntimeError, match="No data ingested"):
            pipeline.run_benchmark(verbose=False)


class TestPersistence:

    def test_save_and_load_store(self, tiny_json_dataset: str, tmp_dir: str):
        pipeline = RAGPipeline()
        pipeline.ingest(tiny_json_dataset)
        store_dir = os.path.join(tmp_dir, "store")
        pipeline.save_store(store_dir)

        assert os.path.exists(os.path.join(store_dir, "index.faiss"))
        assert os.path.exists(os.path.join(store_dir, "chunks.json"))

        pipeline2 = RAGPipeline()
        pipeline2.load_store(store_dir)
        results = pipeline2.search("faiss approximate search", strategy="A", top_k=2)
        assert len(results) == 2

    def test_save_benchmark_json(self, populated_pipeline: RAGPipeline, tmp_dir: str):
        populated_pipeline.run_benchmark(verbose=False)
        out_path = os.path.join(tmp_dir, "bench.json")
        populated_pipeline.save_benchmark_json(out_path)
        assert os.path.exists(out_path)
        with open(out_path) as fh:
            data = json.load(fh)
        assert isinstance(data, list)
        assert len(data) == 3

    def test_save_benchmark_without_running_raises(self, tmp_dir: str):
        pipeline = RAGPipeline()
        with pytest.raises(RuntimeError, match="No benchmark results"):
            pipeline.save_benchmark_json(os.path.join(tmp_dir, "x.json"))
