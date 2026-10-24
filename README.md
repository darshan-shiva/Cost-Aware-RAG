# Cost-Aware RAG: grounded answers over your documents, with semantic caching and model routing

Upload PDF or TXT files, ask questions, and get answers that cite the exact
document and page they came from. Two cost controls sit in front of the LLM:
a semantic cache that reuses earlier answers only when the evidence is the
same, and a router that sends simple questions to a cheaper Groq model. Every
request is recorded with its retrieval, cache, routing, token, cost and
latency details, and the dashboard shows those records.

> **Status.** The document-grounded RAG pipeline described here lives on the
> `feature/document-grounded-rag` branch and has **not been deployed** yet.
> The live demo at https://cost-aware-rag.vercel.app still runs the earlier
> cache-and-routing version (see [History](#history)). All numbers below come
> from local runs; nothing here was measured on the hosted service.

## What problem this solves

A plain LLM answers from training data, cannot say where an answer came from,
and costs the same for a trivial question as for a hard one. This project:

1. **Grounds answers** in documents you upload, with `[S1]`-style citations
   that map to stored chunks (filename, page, excerpt, relevance score).
2. **Declines** when the documents do not contain the answer, instead of
   guessing, and skips the LLM call entirely when nothing relevant was found.
3. **Avoids repeat spend** by caching answers per collection and document
   version, while refusing to reuse an answer for a lookalike question with a
   different answer (X2 vs X3, full-time vs part-time).
4. **Routes by complexity** so short factual questions use the lightweight
   model and analytical, long-context or multi-document questions use the
   larger one, with a single fallback on rate limits.
5. **Measures itself** with telemetry and a reproducible evaluation that
   compares an unoptimized baseline with the optimized configuration.

## Architecture

```text
            Upload PDF/TXT                                Ask a question
                  |                                              |
                  v                                              v
  validate (type, size, pages, header,             validate query, collection,
  encryption, duplicate sha256)                    document filter
                  |                                              |
  extract text per page (pypdf)                     embed query (same model)
                  |                                              |
  chunk within each page (900 chars,               semantic cache lookup, scoped to
  150 overlap, break at sentence)                  collection + doc filter + corpus
                  |                                fingerprint ----- valid hit ---> cached answer
  embed chunks (MiniLM or hashing)                               |                 + original citations
                  |                                         miss / rejected
  store pages, chunks, float32 vectors                           |
  in SQLite; bump version;  -----------> invalidates    top-k exact cosine search
  invalidate collection cache            that cache     (numpy, per collection)
                                                                 |
                                                   no chunk >= min_relevance?
                                                      yes -> "insufficient evidence"
                                                             (no LLM call)
                                                                 |
                                                   grounded prompt: system rules +
                                                   <source id=S1 document=.. page=..>
                                                                 |
                                                   router: lightweight | complex
                                                   (keywords, length, context size,
                                                    multi-document comparisons)
                                                                 |
                                                   Groq gateway: one fallback on
                                                   429 / model unavailable
                                                                 |
                                                   parse [S#], drop invalid labels,
                                                   "uncited" answers are flagged
                                                   and never cached
                                                                 |
                                                   store in cache, record telemetry
                                                                 |
                                                   React UI: answer, citations,
                                                   cache/fresh badge, pipeline timings
```

One SQLite file (WAL mode) holds documents, page text, chunk vectors, the
semantic cache and telemetry. Raw uploads are not kept; re-indexing rebuilds
chunks from the stored page text.

## How it works

### Ingestion

- **Accepted:** `.pdf` and `.txt` up to `RAG_MAX_UPLOAD_MB` (10 MB) and
  `RAG_MAX_PDF_PAGES` (300). Files are checked by extension, content type
  and content: a PDF needs a `%PDF-` header and must not be encrypted, a TXT
  must decode as UTF-8 or UTF-16 and must not be binary. Filenames are
  sanitized; collection names must match `[A-Za-z0-9][A-Za-z0-9_-]{0,63}`.
- **Duplicates:** the same bytes already in the collection return the
  existing document (`outcome: "duplicate"`). A different file with the same
  filename replaces it as a new version (`outcome: "replaced"`), keeping the
  document ID.
- **Extraction:** pypdf per page; text is NFKC-normalized, hyphenated line
  breaks are rejoined, and wrapped PDF lines are joined. A document with no
  extractable text (a scan) fails with a message saying OCR is not supported.
- **Chunking:** chunks never cross a page, so every citation has exactly one
  page number. Target 900 characters with 150 overlap, preferring a paragraph
  or sentence break in the last 30% of the window.
- **Embedding:** `all-MiniLM-L6-v2` via Sentence Transformers, or the
  simulated hashing embedder for low-memory hosts (lexical, not semantic).
  The model name is stored with each chunk; searching only uses vectors from
  the current model.
- **Processing** runs in the background. The document moves through
  `pending -> processing -> ready | failed`, and the UI polls it. Documents
  left in `processing` by a restart are marked failed and can be re-indexed.

### Retrieval and grounding

- Exact cosine similarity over the collection's vectors (cached in memory per
  collection and rebuilt after any change). Defaults: `top_k=5`,
  `min_relevance=0.30` for MiniLM (about 0.15 for the hashing embedder).
- Passages are wrapped in `<source id="S1" document="..." page="...">` tags.
  Any tag inside a passage is escaped, so a document cannot close its own
  wrapper. The system prompt says passages are untrusted data and instructions
  inside them must be ignored. Passages that look like injections are counted
  in `retrieval.suspicious_passages`.
- The model must cite `[S#]` and may reply `INSUFFICIENT_EVIDENCE`. Labels
  that do not match a retrieved passage are removed. An answer left with no
  valid citation gets `outcome: "uncited"`, a visible warning, and is not
  cached.
- Outcomes: `answered`, `insufficient_evidence`, `uncited`, `no_documents`.

### Semantic cache and invalidation

A cached answer is reused only if all of these hold:

1. **Same scope:** collection and document filter.
2. **Same corpus fingerprint:** a hash of every document ID, version,
   content hash and embedding model in the scope, plus `top_k`, the relevance
   threshold, chunk settings and prompt version. Uploading, replacing,
   re-indexing or deleting a document also deletes that collection's cached
   answers outright.
3. **Close enough:** an exact match after normalization (case, whitespace,
   trailing punctuation), or cosine similarity >= `RAG_SIMILARITY_THRESHOLD`
   (0.82) and:
   - no conflicting numbers or named entities ("France" vs "Germany",
     "X2" vs "X3");
   - the new question retrieves mostly the same top evidence chunks
     (overlap >= `RAG_CACHE_MIN_EVIDENCE_OVERLAP`, 0.5);
   - no contrasting terms that both appear in the evidence (a "full-time"
     question will not reuse a "part-time" answer when the handbook covers
     both);
   - the cached answer mentions every topic the new question adds that the
     passages also use (a "warranty" question will not reuse a capacity-only
     answer).

Cache hits return the original citations, are labelled "cached" in the UI,
record zero tokens, and report the estimated cost they avoided. If a cited
chunk has since disappeared, the entry is dropped and a fresh answer is
generated. `force_refresh` bypasses the cache.

### Routing

`RAG_ROUTING_POLICY` is `complexity` (default), `fixed-lightweight` or
`fixed-complex`. The complexity policy picks the complex model for analytical
keywords (compare, explain why, trade-off ...), long questions, retrieved
context above `RAG_ROUTING_CONTEXT_TOKEN_THRESHOLD` (2500 estimated tokens) or
above the lightweight model's context limit, and comparisons that span
`RAG_ROUTING_MULTI_DOC_THRESHOLD` or more documents. Each response says which
factors fired.

On a rate-limit or model-unavailable error the gateway tries the other tier
once; it never loops. Timeouts, auth errors and bad requests are not retried.
Models the account reports as unavailable are skipped up front.

**Limitations:** the heuristic is hand-written, not trained, and has not been
validated against real answer quality. A question with simple wording but a
hard answer goes to the small model.

### Cost methodology

`estimated_cost = input_tokens / 1000 * input_price + output_tokens / 1000 * output_price`
using the per-1K prices in `RAG_*_COST_*_PER_1K`. Tokens are provider-reported
when Groq returns usage and a 4-characters-per-token estimate otherwise;
`token_source` says which. Prices are values you configure, so every cost in
the UI and API is labelled an estimate. Embedding, storage and hosting costs
are not included, and there is no billing integration.

The comparison baseline is **no cache and always the complex model**, the
setup you would run if cost did not matter. See [Evaluation](#evaluation).

### Telemetry

Each request records mode (`rag` or `general`), collection, outcome, cache
decision and match type, similarity, rejection reason, selected model,
routing reasons, fallback and retry count, retrieved and cited chunk counts,
top score, tokens, token source, estimated cost and cost avoided, and stage
timings (embedding, cache lookup, retrieval, generation, total). Errors store
a category (`rate_limited`, `timeout`, ...) and never the provider's raw
message. Set `RAG_TELEMETRY_STORE_QUERY_TEXT=false` to store a hash instead
of the question.

## Security

- The Groq key is read only by the backend. It is never sent to the
  browser, written to logs, or included in error responses.
  `GET /api/config` returns only non-secret settings and whether a key is set.
- Uploads are validated by size, page count, header and encoding before
  parsing; encrypted PDFs are rejected; filenames are sanitized and never used
  as paths.
- Rate limits: `RAG_QUERY_RATE_LIMIT` (30/minute) on both query endpoints and
  `RAG_UPLOAD_RATE_LIMIT` (10/minute) on upload and re-index.
- In production, `POST /api/metrics/reset` requires the `X-Admin-Token`
  header to match `RAG_ADMIN_TOKEN`, and it is disabled when no token is set.
- CORS allows only `RAG_CORS_ORIGINS`, without credentials. Responses send
  `X-Content-Type-Options: nosniff` and a strict `Referrer-Policy`. Request
  IDs from clients are sanitized.
- Prompt injection: see [Retrieval and grounding](#retrieval-and-grounding).
  The defences reduce the risk but cannot rule it out with any LLM.
- **There is no user authentication.** Collections separate documents, but
  anyone who can reach the API can read or delete any collection. Do not
  upload confidential documents to a public deployment.

## Evaluation

`backend/eval/` contains a synthetic corpus (three documents with lookalike
facts) and 38 requests in cold, warm, changed-document and deleted-document
phases. `python -m eval.run_eval` runs four configurations on identical
retrieval settings. See [backend/eval/README.md](backend/eval/README.md) for
the method.

**Measured offline** (run `20261024T142916Z`, hashing embedder,
**extractive stub instead of an LLM**, estimated tokens):

| Metric | Baseline (no cache, always complex) | Optimized (cache + routing) |
|---|---:|---:|
| Retrieval hit@5, cold questions | 100% | 100% |
| Citation precision | 80.7% | 79.6% |
| Cache hit rate | 0% | 18.4% (4 exact, 3 semantic) |
| Incorrect cache hits | 0 | 0 |
| Lookalike questions answered correctly | 6/6 | 6/6 |
| Stale answers after a document change | 0 | 0 |
| Citations to a deleted document | 0 | 0 |
| Unsupported questions declined | 2/4 | 2/4 |
| Model calls | 38 | 31 |
| Estimated cost (USD, 38 requests) | 0.004236 | 0.001729 (-59.2%) |

Ablation: cache alone -21.8%, routing alone -48.3%.

How to read this:

- **Routing's share is arithmetic, not evidence.** The lightweight model is
  configured at half the complex price; the stub ignores the model, so the
  run cannot show whether the small model answers as well.
- **Caching's share is workload-dependent.** 11 of 38 requests are repeats or
  paraphrases by design; real traffic may repeat far less.
- The stub cannot recognise on-topic but unanswerable questions, which is why
  only 2 of 4 unsupported questions were declined. A real model should do
  better there, but that has not been measured.
- Latency figures (about 5 ms per request) cover only the pipeline: there is
  no real generation time in them.
- The relevance threshold was tuned on this same workload.

To measure real quality, tokens and latency, run with a Groq key and MiniLM:
`python -m eval.run_eval --embedding sentence-transformers --llm groq`.
That run has not been done yet.

### Example output (offline stub)

Two rows from the run above, trimmed. The answers are sentences picked by the
extractive stub, not LLM output, which is why they include extra sentences.

```json
{"id": "M2", "query": "Compare the fast charging wattage and the warranty length of the Aurora X3.",
 "outcome": "answered", "model": "openai/gpt-oss-120b", "cache_hit": false,
 "answer": "The Aurora X3 supports 65 watt fast charging and reaches a full charge in about 1.5 hours. [S3] The Aurora X3 carries a 24 month limited warranty. [S1] ...",
 "citations": [["aurora-battery-manual.pdf", 2], ["aurora-battery-manual.pdf", 3], ["aurora-battery-manual.pdf", 1]]}

{"id": "L2", "query": "What is the warranty period of the Aurora X3?",
 "outcome": "answered", "cache_hit": true, "cache_match": "semantic", "cache_similarity": 0.8347,
 "reused_from": "A7 (What is the capacity of the Aurora X3?)",
 "answer": "The Aurora X3 is a 10,000 mAh portable battery ... [S1] The Aurora X3 carries a 24 month limited warranty. [S2] ..."}
```

The second row shows a limit of the hashing embedder: a capacity question and
a warranty question scored 0.83. The reuse was allowed only because the cached
answer happens to state the warranty; the cache now also rejects a semantic
match when the new question asks about something the passages cover but the
cached answer never mentions, so a real LLM's capacity-only answer would not
be reused for the warranty question.

Other questions from the eval set worth trying: "What is the warranty period
of the Aurora X2?" (12 months, must not reuse the X3 answer), "How many
vacation days do part-time employees receive?", and "What do Northwind
expense claims require, and can Greenfield final sale items be returned?"
(two documents).

## Implemented vs planned

| Implemented (on this branch, tested locally) | Planned / not done |
|---|---|
| PDF and TXT upload, validation, duplicates, versioned replace, re-index, delete | OCR for scanned PDFs; DOCX/HTML |
| Page-bounded chunking, MiniLM or hashing embeddings, SQLite vectors | ANN index (FAISS, pgvector) for large corpora |
| Grounded answers with verified citations and insufficient-evidence handling | Answer-quality evaluation with a real LLM (not yet run) |
| Collection- and version-scoped semantic cache with evidence checks | User accounts and per-user access control |
| Configurable routing with a single fallback | Learned or quality-validated routing |
| Telemetry with estimate labelling, dashboard on real data | Billing integration; alerting |
| Documents page, grounded chat, citations, mobile layout | Streaming responses |
| Offline evaluation harness with baseline and ablations | Deployment of this branch (pending approval) |

## Tech stack

Backend: Python 3.11+, FastAPI, pydantic-settings, SQLite, numpy, pypdf,
Sentence Transformers (optional), Groq SDK, SlowAPI, pytest.
Frontend: React, TypeScript, Vite, Tailwind CSS, Recharts, Lucide, Vitest,
Testing Library.

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Status, embedding provider, LLM key configured (yes/no), document counts |
| `/api/config` | GET | Non-secret settings the UI needs |
| `/api/rag/query` | POST | Grounded answer over a collection (optional `document_ids`, `top_k`, `min_relevance`, `force_refresh`) |
| `/api/query` | POST | General-knowledge mode, **not grounded** (the original endpoint) |
| `/api/collections` | GET | Collections with document and chunk counts |
| `/api/documents` | GET / POST | List documents (`?collection=`) / upload (multipart `file`, `collection`) |
| `/api/documents/{id}` | GET / DELETE | Document status and metadata / delete it and its chunks |
| `/api/documents/{id}/reindex` | POST | Rebuild chunks and embeddings |
| `/api/metrics/summary` | GET | Aggregates (`?mode=rag|general`) |
| `/api/metrics/requests` | GET | Recent request records |
| `/api/metrics/reset` | POST | Clear telemetry and cache (admin token in production) |

Interactive docs at `/docs` when the backend is running.

## Local setup

```bash
git clone https://github.com/darshan-shiva/Cost-Aware-RAG.git
cd Cost-Aware-RAG/backend
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt        # or requirements-lite.txt (no PyTorch, hashing embedder)
cp .env.example .env                   # then set RAG_GROQ_API_KEY
python -m uvicorn app.main:app --reload
```

```bash
cd frontend
npm install
npm run dev            # http://localhost:5173, proxies /api to :8000
```

Without a Groq key the app still starts: uploads, retrieval and the
insufficient-evidence path work, and generation returns a 503 that names the
missing setting.

### Environment variables

All backend settings use the `RAG_` prefix; [backend/.env.example](backend/.env.example)
lists every one with its default. The ones you will most likely change:

| Variable | Default | Notes |
|---|---|---|
| `RAG_GROQ_API_KEY` | (none) | Required for generation. Backend only. |
| `RAG_LIGHTWEIGHT_MODEL_ID` / `RAG_COMPLEX_MODEL_ID` | `openai/gpt-oss-20b` / `openai/gpt-oss-120b` | Must exist on your Groq account |
| `RAG_*_COST_*_PER_1K` | see example | Your price estimates; confirm against Groq pricing |
| `RAG_EMBEDDING_PROVIDER` | `sentence-transformers` | `simulated` for hosts under ~1 GB RAM |
| `RAG_RETRIEVAL_MIN_RELEVANCE` | `0.30` | About `0.15` with `simulated` |
| `RAG_SIMILARITY_THRESHOLD` | `0.82` | Cache reuse threshold |
| `RAG_ROUTING_POLICY` | `complexity` | or `fixed-lightweight`, `fixed-complex` |
| `RAG_DATABASE_PATH` | `backend/metrics.db` | Point at a persistent disk in production |
| `RAG_CORS_ORIGINS` | localhost:5173 | JSON list |
| `RAG_ENVIRONMENT` / `RAG_ADMIN_TOKEN` | `development` / empty | Production locks the reset endpoint |
| `VITE_API_BASE_URL` (frontend) | empty | Public backend URL; build-time; never a secret |

## Tests

```bash
cd backend && python -m pytest -q      # 134 passed, 18 skipped (last local run)
cd frontend && npm test && npm run build   # 11 passed; build succeeds
```

All LLM calls are mocked, so the tests need no key, network or money. The
18 skipped tests use the real MiniLM model and run automatically when its
weights are available locally (they were not in the environment used for the
last run). Coverage includes ingestion and validation, retrieval accuracy and
filters, citation verification, insufficient evidence, cache scoping,
lookalike rejection and invalidation on replace/delete/re-index, routing and
fallback, telemetry, rate limits, prompt injection, and secret handling. The
frontend tests cover the documents page, the chat with citations, the
cache badge, and loading, empty and error states.

## Deployment

**Backend (Render).** `render.yaml` is a reference Blueprint; it is not
applied to the existing service automatically. Decide on storage first:

- **Persistent:** a paid instance with a disk mounted at `/var/data` and
  `RAG_DATABASE_PATH=/var/data/rag.db`. Documents, cache and telemetry
  survive restarts.
- **Ephemeral (free tier):** works, but every deploy or restart deletes all
  uploaded documents and telemetry. Fine for a demo; say so in the UI or docs.

On about 512 MB of RAM use `requirements-lite.txt` with
`RAG_EMBEDDING_PROVIDER=simulated` and `RAG_RETRIEVAL_MIN_RELEVANCE=0.15`;
retrieval is then lexical, not semantic. Use MiniLM on 2 GB or more. Run a
single worker (background ingestion and the vector cache are per process),
set `RAG_CORS_ORIGINS` to the Vercel origin, `RAG_ENVIRONMENT=production`,
and `RAG_GROQ_API_KEY` / `RAG_ADMIN_TOKEN` as secrets. Health check: `/health`.

**Frontend (Vercel).** Set `VITE_API_BASE_URL` to the backend URL for
Production and redeploy, because Vite embeds it at build time.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Generation returns 503 "RAG_GROQ_API_KEY is not set" | No key in the backend environment |
| 502 with `error_category: model_unavailable` | Model ID not enabled on your Groq account |
| Every question gets "insufficient evidence" | `RAG_RETRIEVAL_MIN_RELEVANCE` too high for the embedder (use ~0.15 with `simulated`) or document still processing |
| Document stuck in `failed` after a restart | Ingestion was interrupted; press Re-index |
| PDF fails with "No extractable text was found" | Scanned PDF with no text layer (no OCR) |
| Documents vanish after deploy | Ephemeral filesystem; add a disk |
| Browser shows CORS errors | Frontend origin missing from `RAG_CORS_ORIGINS` |
| Process killed on startup | MiniLM + PyTorch exceeded memory; use the lite setup |

## Known limitations

- No authentication or per-user isolation.
- Exact vector search loads a collection's vectors into memory; fine for
  thousands of chunks, not millions.
- Single process: background ingestion is not a durable job queue.
- No OCR, tables are extracted as plain text, and scanned PDFs yield nothing.
- The hashing embedder matches words, not meaning; paraphrases with different
  vocabulary may miss.
- Routing quality and LLM answer quality are unmeasured (offline stub only).
- Costs are estimates from configured prices.

## Future work

Run the live evaluation with MiniLM and Groq and publish it; add auth with
per-user collections; move vectors to pgvector or FAISS for large corpora;
add OCR; stream responses; validate routing against answer quality; add a
reranker.

## History

Before the RAG upgrade, the project cached LLM answers to general questions
and routed them by complexity (Phases 1 to 7: foundation, MiniLM semantic
cache, Groq integration and routing, cache evaluation, observability,
hardening, and deployment to Vercel and Render). That mode is still available
as "General" in the UI and `POST /api/query`.

Its cache evaluation, on 16 paraphrase pairs and 8 unrelated pairs with
MiniLM, found mean similarity 0.733 for paraphrases and 0.025 for unrelated
pairs; threshold 0.70 gave precision 1.000 and recall 0.625, and 0.82 gave
precision 1.000 and recall 0.250. The set is too small to rule out false
positives, which is one reason the RAG cache adds evidence and entity checks
on top of similarity.

## License

No license file yet. Add one if you intend to distribute the project.
