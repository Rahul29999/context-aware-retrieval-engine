# Context-Aware Retrieval Engine — Semantic RAG & Vector Search

A fully local, production-style Retrieval-Augmented Generation (RAG) pipeline that ingests technical text, creates embeddings, stores them in FAISS, and benchmarks two retrieval strategies:

- **Strategy A** — Raw vector search (embed the query as-is, search FAISS)
- **Strategy B** — AI-Enhanced retrieval (rewrite/expand the query with a generative model first, then embed and search)

All GCP/Vertex AI components are mocked deterministically — no API keys, no internet access required.

---

## Repository Structure

```
rag_engine/
├── data/
│   └── sample_texts.json          # 10 technical paragraphs (dataset)
├── src/
│   ├── __init__.py
│   ├── mock_vertexai.py           # Mocks: TextEmbeddingModel, GenerativeModel
│   ├── embedding.py               # EmbeddingEngine wrapper
│   ├── vector_store.py            # FAISS-backed VectorStore + Chunk/SearchResult
│   ├── query_expander.py          # QueryExpander (uses mock GenerativeModel)
│   ├── retriever.py               # Retriever: Strategy A & B
│   ├── pipeline.py                # RAGPipeline: ingest → embed → index → benchmark
│   └── utils.py                   # Data loading, formatting helpers
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Shared pytest fixtures
│   ├── test_embedding.py
│   ├── test_vector_store.py
│   ├── test_retriever.py
│   ├── test_mock_vertexai.py
│   └── test_pipeline.py
├── run_benchmark.py               # CLI entry-point
├── benchmark_results.json         # Generated benchmark output (JSON)
├── retrieval_benchmark.md         # Human-readable benchmark analysis
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10 or 3.11 or 3.12
- pip

### 1. Clone / copy the repository

```bash
git clone <your-repo-url>
cd rag_engine
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# or
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on the embedding model:**  
> On first run, `sentence-transformers` will download `all-MiniLM-L6-v2` (~90 MB) from Hugging Face.  
> If your environment has no internet access, the `mock_vertexai.py` module automatically falls back to a **deterministic NumPy-based encoder** so the pipeline runs fully offline. Semantic quality is reduced but all logic, tests, and benchmark flows work identically.

---

## Running the Demo Benchmark

```bash
python run_benchmark.py
```

Optional flags:
```bash
python run_benchmark.py --dataset data/sample_texts.json --top-k 3 --output benchmark_results.json
python run_benchmark.py --quiet   # suppress table output, only save JSON
```

The script:
1. Ingests `data/sample_texts.json` (10 chunks)
2. Runs Strategy A vs Strategy B for 3 complex queries
3. Prints a formatted comparison table
4. Saves `benchmark_results.json`

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov-report=term-missing

# A single module
pytest tests/test_pipeline.py -v
```

All tests are deterministic and run fully offline — no network calls.

---

## Architecture Overview

```
User Query
    │
    ├──[Strategy A]──────────────────────────────────────────────┐
    │   embed(query)  →  FAISS search  →  Top-K SearchResults    │
    │                                                             │
    └──[Strategy B]──────────────────────────────────────────────┤
        GenerativeModel.expand(query)                            │
            → embed(expanded_query)                              │
            → FAISS search                                       │
            → Top-K SearchResults                                │
                                                                 ▼
                                                         Benchmark Table / JSON
```

### Component map — local mock → production Vertex AI

| Local component | Production equivalent |
|---|---|
| `mock_vertexai.TextEmbeddingModel` | `vertexai.language_models.TextEmbeddingModel` |
| `sentence-transformers/all-MiniLM-L6-v2` | `textembedding-gecko@003` on Vertex AI |
| `mock_vertexai.GenerativeModel` | `vertexai.generative_models.GenerativeModel("gemini-1.5-pro")` |
| `faiss.IndexFlatIP` (in-memory) | Vertex AI Vector Search (Matching Engine) |
| `VectorStore.search()` | `IndexEndpoint.find_neighbors()` |

---

## Similarity Metric: Cosine vs Euclidean

This project uses **cosine similarity** via FAISS `IndexFlatIP` on L2-normalised vectors.

**Why cosine?**

1. **Scale invariance** — Embedding magnitudes carry no semantic information. Cosine similarity measures only the *direction* of vectors, which encodes semantic meaning.
2. **Model alignment** — `sentence-transformers` and `textembedding-gecko` are trained with cosine similarity as the loss objective. Using Euclidean distance would misalign the metric with training.
3. **Normalisation trick** — By L2-normalising all vectors before storing, dot product equals cosine similarity. `IndexFlatIP` performs exact search with this property, giving perfect recall at the cost of O(N·D) per query — acceptable for corpora up to ~1M vectors.
4. **Interpretability** — Cosine scores lie in [−1, 1]; 1 = identical direction, 0 = orthogonal, −1 = opposite. This is intuitive for threshold-based filtering.

**When would Euclidean be better?** For tasks where absolute vector magnitude is meaningful (e.g., bag-of-words TF-IDF vectors), Euclidean distance is appropriate. For dense semantic embeddings, cosine is the standard choice.

---

## Migrating to Vertex AI Vector Search (Production)

### Step-by-step

```python
# 1. Replace mock imports in mock_vertexai.py / embedding.py:
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel

# 2. Initialise Vertex AI
vertexai.init(project="your-gcp-project", location="us-central1")

# 3. Embed with the real model
model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
embeddings = model.get_embeddings(["your text"])
vectors = [e.values for e in embeddings]

# 4. Create a Matching Engine index (one-time, or via Terraform)
from google.cloud import aiplatform
index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name="rag-index",
    contents_delta_uri="gs://your-bucket/embeddings/",
    dimensions=768,
    approximate_neighbors_count=150,
)

# 5. Deploy to an endpoint
endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name="rag-endpoint",
    public_endpoint_enabled=True,
)
endpoint.deploy_index(index=index, deployed_index_id="rag_deployed")

# 6. Query (replaces VectorStore.search)
response = endpoint.find_neighbors(
    deployed_index_id="rag_deployed",
    queries=[query_vector],
    num_neighbors=3,
)
```

No changes are needed in `retriever.py`, `query_expander.py`, `pipeline.py`, or any test file. The entire migration is isolated to `mock_vertexai.py` and `embedding.py`.

---

## Assumptions

1. **Dataset size** — The pipeline is designed for corpora up to ~1M vectors using `IndexFlatIP` (exact search). For larger corpora, switch to `IndexIVFFlat` or `IndexHNSWFlat` in `vector_store.py`.
2. **Offline fallback** — When Hugging Face is unreachable, a 128-dim hash-based NumPy encoder is used automatically. All tests pass with either encoder.
3. **Deterministic expansion** — The mock `GenerativeModel` uses rule-based expansion keyed to query fragments. A real LLM would produce richer, non-deterministic expansions. Tests are written to be compatible with both.
4. **No chunking strategy** — The dataset is pre-chunked. For raw documents, add a chunking step (e.g., by sentence, paragraph, or sliding window) before calling `pipeline.ingest()`.
5. **Single ingestion call** — `pipeline.ingest()` can be called multiple times to append more data; the FAISS index is additive.
