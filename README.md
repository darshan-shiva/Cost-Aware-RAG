
# Cost-Aware RAG: Semantic Caching & Intelligent Model Routing

A cost-aware AI infrastructure project that reduces repeated LLM inference through semantic caching and routes queries to different model tiers based on estimated complexity. It combines local sentence embeddings, Groq-hosted LLM inference, SQLite telemetry, and an interactive React dashboard.

## Project Status

**Completed through Phase 3: Real LLM Integration & Intelligent Routing**

- **Phase 1 — Foundation:** Backend API, frontend dashboard, and SQLite telemetry.
- **Phase 2 — Real Semantic Caching:** Local `all-MiniLM-L6-v2` embeddings, cosine-similarity matching, persistent cache storage, and cache statistics.
- **Phase 3 — Real LLM Integration & Intelligent Routing:** Groq API integration, lightweight and complex model tiers, real response generation, token usage tracking, and estimated inference costs.

Upcoming phases cover cache quality evaluation, performance analysis, production-grade observability, reliability and security, and final deployment.

## Features

### 1. Semantic Caching

- Local embeddings using Sentence Transformers (`all-MiniLM-L6-v2`).
- Cosine-similarity matching with a configurable threshold.
- SQLite-backed cache persistence.
- Cache bypass through force refresh.
- Cache hits reuse stored responses and avoid another LLM generation call.

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

### 4. Observability Dashboard

- Total request count.
- Cache hit rate and cache outcomes.
- Estimated LLM spend.
- Request latency statistics.
- Model selection and routing reasons.
- Interactive query runner with cache threshold and force-refresh controls.

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

## Testing

Run the backend tests from the `backend` directory:

```bash
python -m pytest -q
```

Tests should use mocks for external LLM calls so the test suite does not require a live API key or incur inference costs. The project has previously collected 33 tests, and the semantic cache test file passed 14 tests during Phase 3 verification.

## Current Limitations

- Routing is rule-based rather than driven by a trained complexity classifier.
- Estimated costs depend on configured token prices and may differ from actual provider billing.
- No retrieval corpus or document-ingestion pipeline is documented as implemented here; the current focus is semantic response caching and model routing.
- Cache quality, false-positive matching, and performance benchmarks are planned for the next phase.
- Production authentication, rate limiting, and deployment hardening remain future work.

## Roadmap

- [x] Phase 1 — Foundation
- [x] Phase 2 — Real Semantic Caching
- [x] Phase 3 — Real LLM Integration & Intelligent Routing
- [ ] Phase 4 — Cache Quality & Performance Evaluation
- [ ] Phase 5 — Production-Grade Observability
- [ ] Phase 6 — Reliability, Security & Scalability
- [ ] Phase 7 — Final Dashboard & Deployment

## Security

- Keep API keys in environment variables.
- Do not commit `.env`, virtual environments, or local database files.
- Avoid including secrets in logs, screenshots, or public issue reports.

## License

Add a license file if you intend to distribute this project under an open-source license.
