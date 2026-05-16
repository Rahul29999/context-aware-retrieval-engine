# Context-Aware Retrieval Engine — Semantic RAG & Vector Search

A local Retrieval-Augmented Generation (RAG) pipeline that ingests technical text, creates embeddings, stores them in FAISS, and benchmarks two retrieval strategies:

- **Strategy A** — Raw vector search (embed the query as-is, search FAISS)
- **Strategy B** — AI-Enhanced retrieval (rewrite/expand the query with a generative model first, then embed and search)


# Repository Structure

```bash
rag_engine/
├── data/
│   └── sample_texts.json         
├── src/
│   ├── __init__.py
│   ├── mock_vertexai.py          
│   ├── embedding.py               
│   ├── vector_store.py            
│   ├── query_expander.py         
│   ├── retriever.py              
│   ├── pipeline.py                
│   └── utils.py                  
├── tests/
│   ├── __init__.py
│   ├── conftest.py                
│   ├── test_embedding.py
│   ├── test_vector_store.py
│   ├── test_retriever.py
│   ├── test_mock_vertexai.py
│   └── test_pipeline.py
├── run_benchmark.py              
├── benchmark_results.json         
├── retrieval_benchmark.md        
├── requirements.txt
└── README.md
````

---

## Setup

### Prerequisites

* Python 3.11
* pip

### 1. Clone the repository

```bash
git clone https://github.com/Rahul29999/context-aware-retrieval-engine.git
cd context-aware-retrieval-engine
```

### 2. Create virtual environment

```bash
python -m venv .venv
```

### 3. Activate environment

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

The project uses the Hugging Face `all-MiniLM-L6-v2` embedding model through `sentence-transformers`.

---

## Local Development

The entire project was developed and tested locally in VS Code using Python virtual environments and PowerShell.

During setup and execution, the following issues were handled and resolved locally:

* PowerShell execution policy issue while activating `.venv`
* Torch version compatibility issue
* FAISS installation and dependency setup
* Hugging Face embedding model setup
* Benchmark execution and retrieval validation
* PyTest execution and debugging

---

## Running the Benchmark

```bash
python run_benchmark.py
```

Optional flags:

```bash
python run_benchmark.py --dataset data/sample_texts.json --top-k 3 --output benchmark_results.json
python run_benchmark.py --quiet
```

The benchmark pipeline:

1. Ingests the dataset
2. Generates embeddings
3. Stores vectors in FAISS
4. Runs Strategy A vs Strategy B retrieval
5. Saves benchmark results as JSON

---

## Benchmark Output

### Query

```text
How does the system handle peak load?
```

### Strategy A (Raw Retrieval)

| Rank | Score  | Topic       |
| ---- | ------ | ----------- |
| 1    | 0.5589 | scalability |
| 2    | 0.3799 | kubernetes  |
| 3    | 0.3561 | caching     |

### Strategy B (Expanded Retrieval)

| Rank | Score  | Topic       |
| ---- | ------ | ----------- |
| 1    | 0.7559 | kubernetes  |
| 2    | 0.6138 | scalability |
| 3    | 0.2922 | databases   |

Strategy B surfaced more relevant infrastructure-related chunks for peak-load scalability.

---

### Query

```text
What are the best strategies for semantic vector search?
```

### Strategy A (Raw Retrieval)

| Rank | Score  | Topic            |
| ---- | ------ | ---------------- |
| 1    | 0.5270 | embeddings       |
| 2    | 0.4853 | vector_databases |
| 3    | 0.4370 | vertex_ai        |

### Strategy B (Expanded Retrieval)

| Rank | Score  | Topic            |
| ---- | ------ | ---------------- |
| 1    | 0.7722 | embeddings       |
| 2    | 0.5894 | vector_databases |
| 3    | 0.4958 | vertex_ai        |

Strategy B retrieved more relevant chunks related to semantic vector search and FAISS retrieval.

---

### Query

```text
Explain how the RAG pipeline ingests and retrieves documents
```

### Strategy A (Raw Retrieval)

| Rank | Score  | Topic         |
| ---- | ------ | ------------- |
| 1    | 0.4269 | RAG           |
| 2    | 0.2539 | observability |
| 3    | 0.1668 | transformers  |

### Strategy B (Expanded Retrieval)

| Rank | Score  | Topic            |
| ---- | ------ | ---------------- |
| 1    | 0.6802 | RAG              |
| 2    | 0.4596 | embeddings       |
| 3    | 0.2792 | vector_databases |

Strategy B returned more RAG-specific chunks by expanding the retrieval query with additional semantic context.

---

## Running Tests

```bash
pytest tests/ -v
```

### Test Results

```bash
89 passed
```

The test suite validates:

* embedding generation
* vector search
* retrieval logic
* ingestion pipeline
* query expansion
* ranking correctness
* persistence
* mocked Vertex AI components

The following test modules were executed successfully:

* `test_embedding.py`
* `test_mock_vertexai.py`
* `test_pipeline.py`
* `test_retriever.py`
* `test_vector_store.py`

---

## Architecture Overview

```text
User Query
    │
    ├──[Strategy A]
    │   embed(query) → FAISS search → Top-K Results
    │
    └──[Strategy B]
        expand(query)
            → embed(expanded_query)
            → FAISS search
            → Top-K Results
```

---

## Similarity Metric

This project uses cosine similarity with FAISS `IndexFlatIP`.

Cosine similarity works well for semantic embeddings because it measures contextual similarity between vectors instead of absolute distance. All embeddings are L2-normalized before indexing to improve retrieval consistency.

---

## Vertex AI Migration

The current implementation uses local Hugging Face embeddings and FAISS for retrieval.

For production deployment, the same architecture can be migrated to Google Vertex AI by replacing:

* local sentence-transformers embeddings with Vertex AI `textembedding-gecko`
* FAISS vector storage with Vertex AI Matching Engine
* mocked query expansion with Gemini models

The retrieval pipeline structure remains mostly unchanged during migration.

---

## GitHub Repository

[https://github.com/Rahul29999/context-aware-retrieval-engine](https://github.com/Rahul29999/context-aware-retrieval-engine)

```
```

