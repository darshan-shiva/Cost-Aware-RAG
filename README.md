# Cost-Aware RAG: Semantic Caching & Intelligent Model Routing

**Phase 2 — Real Semantic Caching with Dense Local Embeddings**

A portfolio-grade AI infrastructure system designed to reduce LLM inference costs and response latencies through **genuine embedding-based semantic caching** and **complexity-based model routing**, coupled with a real-time observability telemetry dashboard.

---

## Architecture Overview

```
                                      +------------------------+
                                      |   Client / Frontend    |
                                      | (React + Vite + Recharts)|
                                      +-----------+------------+
                                                  |
                                           POST /api/query
                                                  v
                                      +------------------------+
                                      |      FastAPI App       |
                                      +-----------+------------+
                                                  |
                                                  v
                                      +------------------------+
                                      | SentenceTransformers   |
                                      |   (all-MiniLM-L6-v2)   |
                                      |    (384-dim Vectors)   |
                                      +-----------+------------+
                                                  |
                                                  v
                                      +------------------------+
                                      | Persistent Cache       |
                                      | (Cosine Similarity)    |
                                      | (SQLite + RAM Index)   |
                                      +-----+------------+-----+
                                            |            |
                                       Hit  |            |  Miss
                           (Latency ~1-3ms) |            |
                           (Cost: $0.00000) |            v
                                            |   +------------------------+
                                            |   |   Cost-Aware Router    |
                                            |   | (Complexity Heuristics)|
                                            |   +-----+------------+-----+
                                            |         |            |
                                            |    Low  |            |  High
                                            |         v            v
                                            |   [Lightweight]  [Frontier]
                                            |   (llama-3.1-8b)  (gpt-4o)
                                            |         |            |
                                            |         +-----+------+
                                            |               |
                                            +-------+-------+
                                                    |
                                                    v
                                      +------------------------+
                                      | SQLite Telemetry Store |
                                      |  (p50/p95, Spend, Logs)|
                                      +------------------------+
```

---

## Phase 2 Implementation Highlights

1. **Genuine Dense Semantic Embeddings (`SentenceTransformersEmbeddingProvider`)**:
   - Model: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
   - Loaded once via singleton caching and reused across requests for high throughput and zero cold-start penalty per query.
   - Normalized unit vectors (`normalize_embeddings=True`), allowing exact cosine similarity calculation via dot product.
   - Robust error handling for empty queries, model loading errors, and Windows runtime isolation.

2. **Persistent Semantic Cache (`SemanticCache`)**:
   - In-memory vector index backed by SQLite table `cache_entries`.
   - Persists query strings, responses, 384-dimensional embeddings (JSON format), timestamps, and hit counts across server restarts.
   - Enforces configurable maximum entries (`CACHE_MAX_ENTRIES`) with LRU eviction.
   - Bypassed on forced refresh requests (`force_refresh: true`).
   - Rejects empty queries or empty model responses.

3. **Cost-Aware Routing Engine (`CostAwareRouter`)**:
   - Evaluates query complexity (analytical keywords, token length, structured context).
   - Routes to **Lightweight Tier** (`llama-3.1-8b-instruct`) or **Frontier Tier** (`gpt-4o`).
   - Supplies human-readable routing rationale with every execution.
   - Transparent cost calculation: Cache hits incur **$0.000000** LLM spend; cache misses calculate input/output token costs based on model tier.

4. **Telemetry & Observability Dashboard (`React` + `TypeScript` + `Recharts`)**:
   - Live metrics: Total requests, cache hit rate, estimated LLM spend, and rank-ordered p50/p95 latencies.
   - Interactive pipeline runner to test semantic paraphrases and cache hits with real-time feedback.
   - Recharts visual breakdowns for Cache Outcomes and Model Distribution.
   - Audit trail table with expandable execution metadata and similarity scores.

---

## Quickstart

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm

### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Install dependencies
pip install -r requirements.txt

# Run backend test suite (18 automated tests)
python -m pytest tests -v

# Start FastAPI development server
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

The backend API will be live at `http://127.0.0.1:8000`. Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

### 2. Frontend Setup

In a separate terminal:

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start Vite development server
npm run dev
```

The dashboard will be accessible at `http://localhost:5173`.

---

## Configuration & Environment Variables

Settings can be customized via environment variables with the `RAG_` prefix:

| Variable | Default | Description |
|---|---|---|
| `RAG_EMBEDDING_PROVIDER` | `sentence-transformers` | Embedding provider (`sentence-transformers` or `simulated`) |
| `RAG_EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | SentenceTransformers model identifier |
| `RAG_SIMILARITY_THRESHOLD` | `0.82` | Cosine similarity threshold for semantic cache hits |
| `RAG_CACHE_MAX_ENTRIES` | `1000` | Maximum entries before LRU eviction |
| `RAG_LIGHTWEIGHT_MODEL_ID` | `llama-3.1-8b-instruct` | Model identifier for low-complexity queries |
| `RAG_COMPLEX_MODEL_ID` | `gpt-4o` | Model identifier for high-complexity queries |
| `RAG_DATABASE_PATH` | `./metrics.db` | SQLite database file location |

---

## Phase 2 Verification & Test Coverage

All 18 automated tests pass across unit and integration levels:
- `test_real_embedding_provider_properties`: 384-dimensional dense vectors, unit norm, non-simulated flag.
- `test_real_embedding_batch`: Batch embedding consistency.
- `test_real_semantic_paraphrase_hit`: Dense vector semantic matching for paraphrases.
- `test_real_semantic_unrelated_query_miss`: Unrelated queries yield low similarity (<0.40) and miss.
- `test_threshold_sensitivity`: Dynamic threshold adjustment switches between hit and miss.
- `test_cache_persistence_across_restarts`: Cache entries survive server restarts via SQLite.
- `test_empty_input_and_invalid_storage`: Empty queries and responses handled safely without crashes.
- `test_cache_bypass_via_force_refresh`: `force_refresh=True` bypasses cache for fresh generation.
- `test_health_check` & `test_query_flow_cache_miss_then_hit`: Full API contracts and telemetry persistence.
