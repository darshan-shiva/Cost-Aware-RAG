# Cost-Aware RAG: Semantic Caching & Intelligent Model Routing

**Phase 1 — Initial Foundation**

A portfolio-grade AI infrastructure system designed to reduce LLM inference costs and response latencies through **semantic caching** and **complexity-based model routing**, coupled with a real-time observability telemetry dashboard.

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
                                      |     Semantic Cache     |
                                      | (Cosine Similarity)    |
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

## Phase 1 Components

1. **Backend API (`FastAPI` & `Pydantic`)**:
   - `GET /health` — Application health status and configuration verification.
   - `POST /api/query` — Query-processing endpoint with cache lookup, model routing, and execution recording.
   - `GET /api/metrics/summary` — Aggregate request counts, cache hit rate, total estimated spend, p50 and p95 latencies.
   - `GET /api/metrics/requests` — Audit log of recent query executions and routing rationale.
   - `POST /api/metrics/reset` — Development utility to flush metrics and in-memory cache.

2. **Semantic Caching Foundation**:
   - Vector-based semantic similarity interface with cosine similarity calculations.
   - Threshold-configured matching (`similarity_threshold: 0.82` default).
   - In-memory vector store with LRU eviction and hit count tracking.
   - Phase 1 uses a deterministic 64-dimensional feature-hashing embedder (`SimulatedEmbeddingProvider`), generating reproducible unit-length vector embeddings without requiring external GPU/API keys.
   - Extension points documented for `sentence-transformers` or remote embedding APIs.

3. **Cost-Aware Routing Foundation**:
   - Query complexity analysis: evaluates lexical length, structural complexity, and multi-step reasoning keywords (`compare`, `analyze`, `tradeoffs`, `step-by-step`).
   - Routes to **Lightweight Tier** (`llama-3.1-8b-instruct` at $0.00015 / $0.00060 per 1k tokens) or **Frontier Tier** (`gpt-4o` at $0.00250 / $0.01000 per 1k tokens).
   - Provides explicit human-readable routing rationale for every decision.
   - Computes transparent estimated spend per request.

4. **Telemetry & Metrics Persistence**:
   - SQLite persistence layer for request audit trail.
   - Real-time calculation of exact p50 (median) and p95 (95th percentile) latencies using ordered rank calculation.
   - Clear separation between estimated token cost and actual provider-reported cost.

5. **Observability Dashboard (`React` + `TypeScript` + `Tailwind CSS` + `Recharts`)**:
   - Restrained, dark-mode AI infrastructure design.
   - Live KPI cards: Total Requests, Cache Hit Rate (with cost-savings badge), Estimated Spend, and p50/p95 Latency.
   - Interactive pipeline runner to test simple queries, complex reasoning, and cache hits.
   - Visual breakdown charts for Cache Hits vs Misses and Model Distribution.
   - Expandable request log table with detailed routing rationale.

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

# Run backend tests
python -m pytest tests -v

# Start FastAPI development server
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

The backend API will be live at `http://127.0.0.1:8000`. You can explore the interactive OpenAPI documentation at `http://127.0.0.1:8000/docs`.

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
| `RAG_SIMILARITY_THRESHOLD` | `0.82` | Cosine similarity threshold for semantic cache hits |
| `RAG_CACHE_MAX_ENTRIES` | `1000` | Maximum entries before LRU eviction |
| `RAG_LIGHTWEIGHT_MODEL_ID` | `llama-3.1-8b-instruct` | Model identifier for low-complexity queries |
| `RAG_COMPLEX_MODEL_ID` | `gpt-4o` | Model identifier for high-complexity queries |
| `RAG_DATABASE_PATH` | `./metrics.db` | SQLite database file location |

---

## Known Boundaries & Next Phase (Phase 2)

- **Phase 1 Status**: Foundation complete with simulated embeddings, complexity heuristics, mock generation, SQLite metrics, and full observability UI. No external API keys required.
- **Phase 2 Roadmap (Recommended Next)**:
  - Integration of live embedding providers (`sentence-transformers` local or OpenAI/Cohere).
  - Production vector index (FAISS, Chroma, or pgvector).
  - Dynamic routing models with confidence scores.
  - Live LLM streaming API integration with actual token usage parsing.
