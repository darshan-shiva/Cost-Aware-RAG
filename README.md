# Cost-Aware RAG: Semantic Caching & Intelligent Model Routing

A cost-aware AI infrastructure project that reduces repeated LLM inference through semantic caching and routes queries to different model tiers based on estimated complexity. It combines local sentence embeddings, Groq-hosted LLM inference, SQLite telemetry, structured logging, and an interactive React dashboard.

## Project Status

**Completed through Phase 5: Production-Grade Observability**

- **Phase 1 — Foundation:** Backend API, frontend dashboard, and SQLite telemetry.
- **Phase 2 — Real Semantic Caching:** Local `all-MiniLM-L6-v2` embeddings, cosine-similarity matching, persistent cache storage, and cache statistics.
- **Phase 3 — Real LLM Integration & Intelligent Routing:** Groq API integration, lightweight and complex model tiers, real response generation, token usage tracking, and estimated inference costs.
- **Phase 4 — Cache Quality & Performance Evaluation:** Semantic similarity evaluation, cache lookup benchmarking, and illustrative cost-savings projections.
- **Phase 5 — Production-Grade Observability:** Structured request logging, error telemetry, latency statistics, and an enhanced dashboard with real-time metrics.

Upcoming phases focus on reliability, security, scalability, and final deployment.

## Features

### 1. Semantic Caching

- Local embeddings using Sentence Transformers (`all-MiniLM-L6-v2`).
- Cosine-similarity matching with a configurable threshold.
- SQLite-backed cache persistence.
- Cache bypass through force refresh.
- Cache hits reuse stored responses and avoid another LLM generation call.
- Cache lookup performance evaluation using real embeddings.

### 2. Intelligent Model Routing

- Rule-based query complexity assessment.
- Lightweight routing for direct, straightforward questions.
- Complex-model routing for analytical and multi-step questions.
- Routing reasons recorded for observability.

**Configured model tiers:**

| Tier | Model | Intended use |
|---|---|---|
| Lightweight | `openai/gpt-oss-20b` | Direct questions and simpler requests |
| Complex | `openai/gpt-oss-120b` | Analytical queries and more involved reasoning |

Models are accessed through Groq. Model availability and rate limits depend on the provider account.

### 3. LLM Inference & Cost Estimation

- Real model-generated responses on cache misses.
- Provider-reported input and output token usage.
- Estimated inference costs calculated from configurable per-token rates.
- Cache hits recorded with zero additional LLM inference cost.
- Failed generation requests recorded in telemetry.

**Cost note:** Displayed costs are estimates based on configured rates. They are not verified provider billing amounts. `actual_cost` remains unavailable unless a verified source for actual charges is integrated.

### 4. Production Observability

- **Structured logging:** Captures HTTP request details, response status codes, request IDs, and request latency.
- **Request-level telemetry:** Stores query execution records, cache outcomes, selected models, token usage, estimated costs, and errors in SQLite.
- **Performance monitoring:** Tracks request volume, cache hit rate, average latency, p50 latency, and p95 latency.
- **Error monitoring:** Records failed requests and exposes error counts and error rates.
- **Routing visibility:** Tracks model selection and routing reasons.
- **Health monitoring:** Provides an API endpoint for application health and embedding-provider simulation status.
- **Automatic dashboard refresh:** Refreshes metrics periodically and after query execution.
- **Request history:** Displays recent request records for inspection and debugging.

### 5. Interactive Dashboard

- Total request count.
- Cache hit rate and cache outcomes.
- Estimated LLM spend.
- Latency statistics (p50, p95, and average).
- Error rate and failed request count.
- Model routing distribution and request history.
- Interactive query runner with similarity-threshold and force-refresh controls.
- Manual refresh and local metrics/cache reset controls.

## Architecture

```text
User Query
    |
    v
React + Vite Dashboard
    |
    v
FastAPI Backend
    |
    +---- Structured Request Logging
    |
    v
Semantic Cache Lookup
    |
    +---- Cache Hit ----> Return Stored Response
    |                    No LLM Generation
    |
    +---- Cache Miss ---> Complexity-Based Router
                              |
                              +--> Lightweight Groq Model
                              |
                              +--> Complex Groq Model
                                      |
                                      v
                                 Generated Response
                                      |
                                      v
                                 Store in Cache
                                      |
                                      v
                              SQLite Request Telemetry
                                      |
                                      v
                              Metrics API & Dashboard
```

## Technology Stack

**Backend**

- Python
- FastAPI
- Pydantic Settings
- Groq Python SDK
- Sentence Transformers
- PyTorch
- SQLite
- Python logging
- Pytest

**Frontend**

- React
- TypeScript
- Vite
- Tailwind CSS
- Recharts
- Lucide React

## Getting Started

### Prerequisites

- Python 3.11 or a compatible supported Python version.
- Node.js and npm.
- A Groq API key.
- Git.

### 1. Clone the repository

```bash
git clone https://github.com/darshan-shiva/Cost-Aware-RAG.git
cd Cost-Aware-RAG
```

### 2. Set up the backend

```bash
cd backend
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a local `.env` file in the backend directory. Configure the variables expected by `app/config.py`, including your provider, API key, and model IDs.

Example:

```dotenv
RAG_LLM_PROVIDER=groq
RAG_GROQ_API_KEY=your_groq_api_key_here
RAG_LIGHTWEIGHT_MODEL_ID=openai/gpt-oss-20b
RAG_COMPLEX_MODEL_ID=openai/gpt-oss-120b
RAG_LLM_MAX_COMPLETION_TOKENS=2048
```

Never commit `.env` or expose your API key publicly.

### 4. Start the backend

From the `backend` directory:

```bash
python -m uvicorn app.main:app --reload
```

The API documentation is available at:

`http://127.0.0.1:8000/docs`

### 5. Start the frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite, typically:

`http://localhost:5173`

## Observability API

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Application health and embedding-provider status |
| `/api/query` | POST | Process a query through caching and model routing |
| `/api/metrics/summary` | GET | Aggregate request, cache, latency, error, and estimated-cost metrics |
| `/api/metrics/requests` | GET | Retrieve recent request telemetry |
| `/api/metrics/reset` | POST | Clear local metrics and semantic cache |

The metrics endpoints expose operational telemetry to the dashboard. The reset endpoint clears local metrics and cache data; use it carefully.

## Testing

Run the backend tests from the `backend` directory:

```bash
python -m pytest -q
```

The test suite covers API behavior, semantic cache interactions, model routing, cost estimation, and error telemetry.

Tests for external LLM failures should mock provider errors so they can validate failure handling without depending on a live provider failure. Tests that invoke real inference require valid provider credentials and may incur usage costs.

The frontend production build can be checked from the `frontend` directory:

```bash
npm run build
```

A successful build confirms that TypeScript checking and Vite production bundling complete without errors.

## Cache Evaluation and Cost Analysis

### Semantic Similarity Evaluation

The semantic cache was evaluated using a small, manually labelled benchmark of 16 paraphrase pairs and 8 unrelated-query pairs with `sentence-transformers/all-MiniLM-L6-v2`.

- Mean paraphrase similarity: **0.733**
- Mean unrelated-query similarity: **0.025**

| Similarity threshold | Precision | Recall | F1 score |
|---:|---:|---:|---:|
| 0.70 | 1.000 | 0.625 | 0.769 |
| 0.82 | 1.000 | 0.250 | 0.400 |

On this benchmark, threshold `0.70` produced the best F1 score among the tested thresholds. The production threshold remains `0.82` until a larger benchmark can better assess the risk of false-positive cache matches. The small number of unrelated examples is not sufficient to establish that false positives will not occur in production.

### Cache Lookup Performance

The local in-memory benchmark used 3 warm-up runs and 20 measured runs per query type.

| Query type | Observed hits | Mean lookup latency | Median lookup latency |
|---|---:|---:|---:|
| Exact query | 20/20 | 20.792 ms | 19.871 ms |
| Paraphrased query | 0/20 | 17.815 ms | 17.701 ms |
| Unrelated query | 0/20 | 15.659 ms | 15.403 ms |

These timings include query embedding and cache lookup. They do not represent end-to-end request latency or production performance.

### Illustrative Cost-Savings Analysis

The cost benchmark models 1,000 queries, an average of 150 input tokens and 300 output tokens per uncached query, and a 70% lightweight / 30% complex model-routing mix.

The configured estimated costs per uncached query were approximately `$0.000101` for the lightweight tier and `$0.000202` for the complex tier. Under these assumptions, baseline estimated generation spend is **$0.131625 per 1,000 queries**.

| Assumed cache hit rate | Estimated remaining spend | Estimated savings | Savings rate |
|---:|---:|---:|---:|
| 10% | $0.118462 | $0.013163 | 10% |
| 25% | $0.098719 | $0.032906 | 25% |
| 40% | $0.078975 | $0.052650 | 40% |
| 60% | $0.052650 | $0.078975 | 60% |
| 80% | $0.026325 | $0.105300 | 80% |

This is a projection, not measured production savings. It assumes cache hits avoid all LLM generation charges and excludes embedding, storage, and infrastructure costs. Configured prices may differ from actual provider billing.

### Evaluation Limitations

- The semantic benchmark is small and manually labelled; results may not generalize to real traffic.
- The cost benchmark uses illustrative query volume, token usage, and routing distribution.
- Local cache lookup timings may differ in production and exclude end-to-end request latency.
- The benchmark scripts use an in-memory cache and do not modify production cache data or make LLM inference calls.

## Current Limitations

- Routing is rule-based rather than driven by a trained complexity classifier.
- Estimated costs depend on configured token prices and may differ from actual provider billing.
- No retrieval corpus or document-ingestion pipeline is documented as implemented here; the current focus is semantic response caching and model routing.
- Cache evaluation currently uses a small manually labelled benchmark; broader testing is needed before changing the production similarity threshold.
- Production authentication, rate limiting, and deployment hardening remain future work.
- Observability currently focuses on application-level telemetry; external monitoring integrations and alerting have not been documented as implemented.

## Roadmap

- [x] Phase 1 — Foundation
- [x] Phase 2 — Real Semantic Caching
- [x] Phase 3 — Real LLM Integration & Intelligent Routing
- [x] Phase 4 — Cache Quality & Performance Evaluation
- [x] Phase 5 — Production-Grade Observability
- [ ] Phase 6 — Reliability, Security & Scalability
- [ ] Phase 7 — Final Dashboard & Deployment

## Security

- Keep API keys in environment variables.
- Do not commit `.env`, virtual environments, or local database files.
- Avoid including secrets in logs, screenshots, or public issue reports.
- Restrict CORS origins and configure authentication before exposing the service publicly.
- Treat request queries and response previews as potentially sensitive data when retaining telemetry.

## License

Add a license file if you intend to distribute this project under an open-source license.
