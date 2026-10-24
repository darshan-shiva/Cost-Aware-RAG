# RAG evaluation suite

A small, reproducible workload that compares the pipeline with and without
its cost optimizations. Everything runs against a temporary database; it does
not touch `metrics.db` or any deployed service.

```bash
cd backend
python -m eval.run_eval                                   # offline (default)
python -m eval.run_eval --embedding sentence-transformers # real MiniLM embeddings
python -m eval.run_eval --embedding sentence-transformers --llm groq   # live; costs a few cents at most
```

Results go to `eval/results/<run-id>-<generator>-<embedding>.{json,md}`. The
JSON has one row per request (answer, citations, cache decision, tokens,
latency) so every summary number can be traced back.

## Corpus and questions

`corpus/` holds three synthetic documents (see `corpus/README.md`); PDFs are
generated from `corpus/corpus.json`. `dataset.json` defines 38 requests in
four phases:

| Phase | Requests | What it checks |
|---|---:|---|
| cold | 23 | 10 single-fact questions, 3 multi-chunk questions (two pages or two documents), 6 lookalike questions in pairs with different answers (X2 vs X3 warranty and charging, full-time vs part-time vacation), 4 unsupported questions (3 off-topic, 1 on-topic but unanswerable) |
| warm | 11 | 5 exact repeats (one only differs in case and punctuation) and 6 paraphrases of cold questions |
| changed | 2 | the refund policy is replaced by v2 under the same filename; answers must use v2 and not be served from v1 cache entries |
| deleted | 2 | the battery manual is deleted; nothing may cite it afterwards |

Each supported question lists its expected evidence `(file, page)` and the
keywords a correct answer must contain.

## Configurations

| Name | Semantic cache | Routing |
|---|---|---|
| baseline | off (`force_refresh` on every request) | `fixed-complex`: always the larger configured model |
| cache_only | on | `fixed-complex` |
| routing_only | off | `complexity` heuristic |
| optimized | on | `complexity` heuristic |

All four use the same documents, chunking, embedding model, top-k and
relevance threshold, so retrieval is identical and only caching and model
choice differ. The baseline is a "quality-first" fixed-model setup; pass
`--baseline-policy fixed-lightweight` to compare against the cheap model.

## Metrics

- **Retrieval hit@k / MRR**: whether the expected page is among the top-k
  retrieved passages, and its reciprocal rank. Computed on cold-phase
  supported questions that were freshly retrieved (cache hits are excluded
  because they report the original request's retrieval).
- **Expected evidence above threshold**: fraction of expected pages that
  cleared the relevance threshold and were therefore given to the model.
- **Citation precision**: fraction of returned citations whose page is
  expected evidence for that question.
- **Correct**: supported questions need outcome `answered` and every
  expected keyword in the answer; unsupported and deleted-document questions
  need `insufficient_evidence` / `no_documents`. Reported separately for
  freshly generated and cache-served requests so cache hits cannot inflate
  quality.
- **Incorrect cache hits**: cache-served answers that fail the check above
  (for example a lookalike question receiving its pair's answer).
- **Stale answers**: changed-phase answers that are cache hits or contain the
  v1 value.
- **Latency**: mean and median per phase; p95 only when a group has 20 or
  more requests. Cold and warm phases and cache hits are reported separately.
- **Tokens and cost**: tokens come from the provider in live runs and from a
  4-characters-per-token estimate in stub runs; cost applies the configured
  per-1K prices from `app/config.py`. Cost reduction is
  `1 - optimized / baseline` over the whole workload.

## The offline generator

The default `--llm stub` is an extractive heuristic, not a language model:
it returns the source sentences that share the most content words with the
question (at least two), cites them, and replies `INSUFFICIENT_EVIDENCE` when
no sentence overlaps enough. It makes the pipeline testable without an API
key, but:

- answer correctness then measures retrieval and the stub's sentence pick,
  not LLM answer quality;
- it cannot recognise on-topic but unanswerable questions, which a real model
  should decline;
- routing cannot change answer quality, because the stub ignores the model ID;
  the routing-only cost reduction is simply the configured price ratio
  between the two tiers applied to the share of queries routed to the
  lightweight model;
- generation latency is near zero, so latency numbers describe pipeline
  overhead only.

## Limitations

- 38 requests over 3 short documents: enough to exercise every code path and
  catch regressions, far too few to estimate production hit rates, quality or
  cost. Do not read the percentages as general claims.
- The relevance threshold for the hashing embedder (0.15) was chosen by a
  sweep on this same workload, so the offline quality numbers are optimistic.
- Questions and expected keywords were written by the same person who wrote
  the documents; there is no independent labelling.
- Costs are estimates from configured prices, never billing data.
