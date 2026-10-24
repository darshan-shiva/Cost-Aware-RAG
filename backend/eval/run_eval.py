"""Baseline vs optimized evaluation of the RAG pipeline.

Run from backend/:

    python -m eval.run_eval                       # offline: hashing embedder + extractive stub
    python -m eval.run_eval --embedding sentence-transformers
    python -m eval.run_eval --embedding sentence-transformers --llm groq   # live, needs RAG_GROQ_API_KEY

Both configurations run the same workload (eval/dataset.json) on a fresh,
temporary database:

  baseline   cache disabled (every request generates), fixed routing policy
             (default: fixed-complex, i.e. always the larger model)
  optimized  semantic cache enabled, complexity routing

The offline "stub" generator is extractive: it copies the source sentence
that best overlaps the question and cites it, or replies INSUFFICIENT_EVIDENCE
when no sentence overlaps enough. It exercises retrieval, prompting,
citation mapping and caching deterministically, but it is NOT a language
model, so answer-quality numbers from stub runs describe the pipeline and the
retrieval, not LLM quality, and its token counts are character-based
estimates. Results are written to eval/results/.
"""

import argparse
import json
import os
import re
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import HTTPException

from app.api.routes import build_rag_service
from app.cache.embeddings import get_embedding_provider
from app.cache.semantic import SemanticCache
from app.config import settings
from app.llm import LLMGateway
from app.metrics.storage import SQLiteMetricsStore
from app.rag.prompt import INSUFFICIENT_EVIDENCE_TOKEN, estimate_tokens
from app.routing import CostAwareRouter
from app.schemas import RAGQueryRequest
from app.services.query_service import QueryService, build_groq_client

EVAL_DIR = Path(__file__).resolve().parent
CORPUS = EVAL_DIR / "corpus"
RESULTS = EVAL_DIR / "results"

_WORD = re.compile(r"[a-z0-9][a-z0-9.,:'-]*")
_STOP = {
    "the", "a", "an", "is", "are", "of", "to", "do", "does", "how", "what", "many",
    "much", "for", "in", "on", "and", "at", "be", "can", "with", "it", "that", "this",
    "after", "before", "must", "within", "long", "get", "each", "per",
}


def _content(text: str) -> set:
    words = set()
    for w in _WORD.findall(text.lower()):
        w = w.strip(".,:'")
        if w and w not in _STOP:
            words.add(w[:-1] if len(w) > 3 and w.endswith("s") else w)
    return words


class ExtractiveStub:
    """Deterministic offline stand-in for the chat model (see module doc)."""

    name = "extractive-stub"

    def __call__(self, messages, model_id):
        user = messages[-1]["content"]
        question = user.rsplit("Question:", 1)[-1].split("\n")[0]
        q = _content(question)
        sources = re.findall(r'<source id="(S\d+)"[^>]*>\n(.*?)\n</source>', user, re.S)
        scored = []
        for label, text in sources:
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                overlap = len(q & _content(sentence))
                scored.append((overlap, label, sentence.strip()))
        scored.sort(key=lambda s: -s[0])
        if not scored or scored[0][0] < 2:
            answer = INSUFFICIENT_EVIDENCE_TOKEN
        else:
            best = scored[0][0]
            picked, seen = [], set()
            for overlap, label, sentence in scored:
                if overlap >= max(2, best * 0.6) and sentence not in seen and len(picked) < 3:
                    picked.append(f"{sentence} [{label}]")
                    seen.add(sentence)
            answer = " ".join(picked)
        return answer, estimate_tokens(messages[0]["content"] + user), estimate_tokens(answer), "stop"


def _percentile(values: List[float], pct: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(-(-pct * len(ordered) // 1)) - 1))
    return ordered[index]


def latency_stats(values: List[float]) -> Dict[str, object]:
    if not values:
        return {"n": 0}
    out: Dict[str, object] = {
        "n": len(values),
        "mean_ms": round(statistics.fmean(values), 2),
        "median_ms": round(statistics.median(values), 2),
    }
    # p95 from fewer than 20 observations is just the max; do not report it.
    out["p95_ms"] = round(_percentile(values, 0.95), 2) if len(values) >= 20 else None
    return out


def build_pipeline(args, policy: str, db_path: str):
    provider = get_embedding_provider(
        provider_type=args.embedding,
        model_name=settings.EMBEDDING_MODEL_NAME,
        dimension=settings.SIMULATED_EMBEDDING_DIM,
    )
    cache = SemanticCache(embedding_provider=provider, default_threshold=args.cache_threshold, db_path=db_path)
    client = build_groq_client() if args.llm == "groq" else None
    if args.llm == "groq" and client is None:
        sys.exit("--llm groq needs RAG_GROQ_API_KEY.")
    qs = QueryService(
        cache=cache,
        router=CostAwareRouter(policy=policy),
        metrics_store=SQLiteMetricsStore(db_path),
        gateway=LLMGateway(client=client, max_completion_tokens=settings.LLM_MAX_COMPLETION_TOKENS),
    )
    rag = build_rag_service(qs, db_path)
    if args.llm == "stub":
        rag._generate = ExtractiveStub()
    return rag


def ingest(rag, filename: str, source: Optional[str] = None) -> str:
    data = (CORPUS / (source or filename)).read_bytes()
    outcome = rag.ingestion.register_upload("eval", filename, data)
    doc = outcome.document
    if outcome.outcome != "duplicate":
        rag.ingestion.process(doc.document_id, doc.version, data)
    doc = rag.store.get(doc.document_id)
    if doc.status != "ready":
        raise RuntimeError(f"{filename} failed to index: {doc.error_message}")
    return doc.document_id


def run_config(args, name: str, policy: str, use_cache: bool, dataset: dict) -> dict:
    fd, db_path = tempfile.mkstemp(suffix=f"-{name}.db")
    os.close(fd)
    try:
        rag = build_pipeline(args, policy, db_path)
        doc_ids = {f: ingest(rag, f) for f in dataset["documents"]}
        id_by_request: Dict[str, str] = {}
        rows = []
        for phase in dataset["phases"]:
            action = phase.get("action") or {}
            if "replace" in action:
                doc_ids[action["replace"]] = ingest(rag, action["replace"], action["with"])
            if "delete" in action:
                rag.ingestion.delete(doc_ids.pop(action["delete"]))
            for item in phase["queries"]:
                req = RAGQueryRequest(
                    query=item["query"],
                    collection="eval",
                    top_k=args.top_k,
                    min_relevance=args.min_relevance,
                    similarity_threshold=args.cache_threshold,
                    force_refresh=not use_cache,
                )
                started = time.perf_counter()
                try:
                    res = rag.answer(req)
                    error = None
                except HTTPException as exc:
                    res, error = None, f"{exc.status_code}: {exc.detail}"
                wall_ms = (time.perf_counter() - started) * 1000
                rows.append(score_row(item, phase["name"], res, error, wall_ms, id_by_request))
                if res is not None:
                    id_by_request[res.request_id] = item["id"]
        return {"name": name, "policy": policy, "cache_enabled": use_cache, "rows": rows}
    finally:
        for suffix in ("", "-wal", "-shm"):
            try:
                os.remove(db_path + suffix)
            except OSError:
                pass


def score_row(item, phase, res, error, wall_ms, id_by_request) -> dict:
    expected = {tuple(e) for e in item.get("evidence", [])}
    row = {
        "id": item["id"], "phase": phase, "category": item["category"], "query": item["query"],
        "error": error, "wall_ms": round(wall_ms, 2),
    }
    if res is None:
        return row
    sources = res.retrieval.sources
    relevant = [s for s in sources if s.relevance_score >= res.retrieval.min_relevance]
    ranks = [i + 1 for i, s in enumerate(sources) if (s.filename, s.page_number) in expected]
    found = {(s.filename, s.page_number) for s in relevant} & expected
    answer_l = res.answer.lower()
    keywords_ok = all(k.lower() in answer_l for k in item.get("keywords", []))
    cited = [(c.filename, c.page_number) for c in res.citations]
    supported = bool(expected)
    if supported:
        correct = res.outcome == "answered" and keywords_ok
    else:
        correct = res.outcome in ("insufficient_evidence", "no_documents")
    original = id_by_request.get(res.cache.original_request_id or "", None)
    row.update({
        "outcome": res.outcome,
        "answer": res.answer[:300],
        "cache_hit": res.cache.hit,
        "cache_match": res.cache.match_type,
        "cache_similarity": res.cache.similarity,
        "cache_rejection": res.cache.rejection_reason,
        "reused_from": original,
        "model": res.routing.selected_model,
        "fallback": res.routing.fallback_used,
        "input_tokens": res.usage.input_tokens,
        "output_tokens": res.usage.output_tokens,
        "estimated_cost": res.usage.estimated_cost,
        "estimated_cost_avoided": res.usage.estimated_cost_avoided,
        "latency_ms": res.latency.total_ms,
        "generation_ms": res.latency.generation_ms,
        "retrieval_ms": res.latency.retrieval_ms,
        "top_score": res.retrieval.top_score,
        "expected_rank": ranks[0] if ranks else None,
        "evidence_recall": (len(found) / len(expected)) if expected else None,
        "citations": cited,
        "citation_precision": (sum(1 for c in cited if c in expected) / len(cited)) if cited and expected else None,
        "correct": correct,
        "stale": any(k.lower() in answer_l for k in item.get("stale_keywords", [])),
        "cites_deleted": any(c[0] == item.get("deleted") for c in cited),
    })
    return row


def _mean(values) -> Optional[float]:
    values = [v for v in values if v is not None]
    return round(statistics.fmean(values), 4) if values else None


def summarize(config: dict) -> dict:
    rows = [r for r in config["rows"] if not r.get("error")]
    errors = [r for r in config["rows"] if r.get("error")]
    cold = [r for r in rows if r["phase"] == "cold"]
    # Retrieval quality is measured on freshly retrieved requests only; a cache
    # hit shows the original request's retrieval and would double-count it.
    supported_cold = [
        r for r in cold
        if r["category"] in ("answerable", "multi_chunk", "lookalike") and not r["cache_hit"]
    ]
    hits = [r for r in rows if r["cache_hit"]]
    cross = [r for r in hits if r["reused_from"] not in (r["id"], None) and r["reused_from"] != _of(config, r)]
    by_cat: Dict[str, List[dict]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)

    def phase_latency(predicate):
        return latency_stats([r["latency_ms"] for r in rows if predicate(r)])

    total_cost = sum(r["estimated_cost"] for r in rows)
    return {
        "requests": len(config["rows"]),
        "errors": len(errors),
        "retrieval_cold": {
            "n": len(supported_cold),
            "hit_rate_at_k": _mean([1.0 if r["expected_rank"] else 0.0 for r in supported_cold]),
            "mrr": _mean([1.0 / r["expected_rank"] if r["expected_rank"] else 0.0 for r in supported_cold]),
            "evidence_recall_above_threshold": _mean([r["evidence_recall"] for r in supported_cold]),
        },
        "answer_correct_by_category": {
            cat: {"n": len(rs), "correct": sum(1 for r in rs if r["correct"])} for cat, rs in sorted(by_cat.items())
        },
        "citation_precision": _mean([r["citation_precision"] for r in rows]),
        "correct_fresh": [sum(1 for r in rows if not r["cache_hit"] and r["correct"]), sum(1 for r in rows if not r["cache_hit"])],
        "correct_cache_hits": [sum(1 for r in hits if r["correct"]), len(hits)],
        "answers_with_citations": sum(1 for r in rows if r["citations"]),
        "unsupported_declined": {
            "n": len(by_cat.get("unsupported", [])) + len(by_cat.get("unsupported_on_topic", [])),
            "declined": sum(1 for r in by_cat.get("unsupported", []) + by_cat.get("unsupported_on_topic", []) if r["correct"]),
        },
        "stale_answers_after_change": sum(1 for r in by_cat.get("changed_document", []) if r["stale"] or r["cache_hit"]),
        "citations_to_deleted_document": sum(1 for r in by_cat.get("deleted_document", []) if r["cites_deleted"]),
        "cache": {
            "hits": len(hits),
            "hit_rate": round(len(hits) / len(rows), 4) if rows else 0.0,
            "exact": sum(1 for r in hits if r["cache_match"] == "exact"),
            "semantic": sum(1 for r in hits if r["cache_match"] == "semantic"),
            "hits_by_phase": {p: sum(1 for r in hits if r["phase"] == p) for p in ("cold", "warm", "changed", "deleted")},
            "hits_reused_from_a_different_question": len(cross),
            "incorrect_cache_hits": sum(1 for r in hits if not r["correct"]),
            "rejections": sorted({r["cache_rejection"].split(" (")[0] for r in rows if r.get("cache_rejection")}),
        },
        "latency": {
            "all": phase_latency(lambda r: True),
            "cold_phase": phase_latency(lambda r: r["phase"] == "cold"),
            "warm_phase": phase_latency(lambda r: r["phase"] == "warm"),
            "cache_hits": phase_latency(lambda r: r["cache_hit"]),
            "fresh_generations": phase_latency(lambda r: r["generation_ms"] is not None),
        },
        "tokens": {
            "input": sum(r["input_tokens"] for r in rows),
            "output": sum(r["output_tokens"] for r in rows),
        },
        "model_calls": sum(1 for r in rows if r["generation_ms"] is not None),
        "models": {m: sum(1 for r in rows if r["model"] == m and not r["cache_hit"]) for m in sorted({r["model"] for r in rows if r["model"]})},
        "estimated_cost_total": round(total_cost, 6),
        "estimated_cost_per_query": round(total_cost / len(rows), 8) if rows else None,
    }


def _of(config, row) -> Optional[str]:
    return next((q.get("of") for q in _DATASET_INDEX if q["id"] == row["id"]), None)


_DATASET_INDEX: List[dict] = []


def score_distribution(rows: List[dict]) -> dict:
    cold = [r for r in rows if r["phase"] == "cold" and r.get("top_score") is not None]
    sup = [r["top_score"] for r in cold if r["category"] in ("answerable", "multi_chunk", "lookalike")]
    uns = [r["top_score"] for r in cold if r["category"].startswith("unsupported")]
    return {
        "supported_top_score": {"min": min(sup), "median": round(statistics.median(sup), 4)} if sup else None,
        "unsupported_top_score": {"max": max(uns), "median": round(statistics.median(uns), 4)} if uns else None,
    }


def render_markdown(meta: dict, base: dict, opt: dict, ablations: List[dict]) -> str:
    def pct(x):
        return "n/a" if x is None else f"{x * 100:.1f}%"

    def ms(stats, key):
        v = stats.get(key)
        return "n/a" if v is None else f"{v:.2f}"

    b, o = base["summary"], opt["summary"]
    reduction = (
        1 - o["estimated_cost_total"] / b["estimated_cost_total"] if b["estimated_cost_total"] else None
    )
    lines = [
        f"# Evaluation run {meta['run_id']}",
        "",
        f"- Embeddings: `{meta['embedding_provider']}`",
        f"- Generator: `{meta['generator']}`" + (" (offline extractive stub, not an LLM)" if meta["generator"] == "extractive-stub" else ""),
        f"- Retrieval: top_k={meta['top_k']}, min_relevance={meta['min_relevance']}; cache similarity threshold={meta['cache_threshold']}",
        f"- Workload: {b['requests']} requests per configuration ({meta['phase_counts']})",
        f"- Baseline: cache disabled, routing `{base['policy']}`. Optimized: cache enabled, routing `{opt['policy']}`.",
        f"- Prices: configured estimates (USD per 1K tokens) {meta['prices']}",
        "",
        "| Metric | Baseline | Optimized |",
        "|---|---:|---:|",
        f"| Retrieval hit@k, cold supported questions, fresh retrieval only (n={b['retrieval_cold']['n']} / {o['retrieval_cold']['n']}) | {pct(b['retrieval_cold']['hit_rate_at_k'])} | {pct(o['retrieval_cold']['hit_rate_at_k'])} |",
        f"| Retrieval MRR, cold | {b['retrieval_cold']['mrr']} | {o['retrieval_cold']['mrr']} |",
        f"| Expected evidence above relevance threshold, cold | {pct(b['retrieval_cold']['evidence_recall_above_threshold'])} | {pct(o['retrieval_cold']['evidence_recall_above_threshold'])} |",
        f"| Citation precision (cited page is expected evidence) | {pct(b['citation_precision'])} | {pct(o['citation_precision'])} |",
        f"| Unsupported questions declined | {b['unsupported_declined']['declined']}/{b['unsupported_declined']['n']} | {o['unsupported_declined']['declined']}/{o['unsupported_declined']['n']} |",
        f"| Stale answers after document change | {b['stale_answers_after_change']} | {o['stale_answers_after_change']} |",
        f"| Citations to a deleted document | {b['citations_to_deleted_document']} | {o['citations_to_deleted_document']} |",
        f"| Cache hit rate (all requests) | {pct(b['cache']['hit_rate'])} | {pct(o['cache']['hit_rate'])} |",
        f"| Cache hits exact / semantic | {b['cache']['exact']} / {b['cache']['semantic']} | {o['cache']['exact']} / {o['cache']['semantic']} |",
        f"| Incorrect cache hits | {b['cache']['incorrect_cache_hits']} | {o['cache']['incorrect_cache_hits']} |",
        f"| Correct, freshly generated requests | {b['correct_fresh'][0]}/{b['correct_fresh'][1]} | {o['correct_fresh'][0]}/{o['correct_fresh'][1]} |",
        f"| Correct, cache-served requests | {b['correct_cache_hits'][0]}/{b['correct_cache_hits'][1]} | {o['correct_cache_hits'][0]}/{o['correct_cache_hits'][1]} |",
        f"| Model calls | {b['model_calls']} | {o['model_calls']} |",
        f"| Model mix (fresh calls) | {b['models']} | {o['models']} |",
        f"| Input / output tokens | {b['tokens']['input']} / {b['tokens']['output']} | {o['tokens']['input']} / {o['tokens']['output']} |",
        f"| Estimated cost, total (USD) | {b['estimated_cost_total']:.6f} | {o['estimated_cost_total']:.6f} |",
        f"| Estimated cost per query (USD) | {b['estimated_cost_per_query']:.8f} | {o['estimated_cost_per_query']:.8f} |",
        f"| Estimated cost reduction vs baseline | - | {pct(reduction)} |",
        f"| Latency mean / median, all (ms) | {ms(b['latency']['all'], 'mean_ms')} / {ms(b['latency']['all'], 'median_ms')} | {ms(o['latency']['all'], 'mean_ms')} / {ms(o['latency']['all'], 'median_ms')} |",
        f"| Latency p95, all (ms) | {ms(b['latency']['all'], 'p95_ms')} | {ms(o['latency']['all'], 'p95_ms')} |",
        f"| Latency mean, cold phase (ms) | {ms(b['latency']['cold_phase'], 'mean_ms')} | {ms(o['latency']['cold_phase'], 'mean_ms')} |",
        f"| Latency mean, warm phase (ms) | {ms(b['latency']['warm_phase'], 'mean_ms')} | {ms(o['latency']['warm_phase'], 'mean_ms')} |",
        f"| Latency mean, cache hits only (ms, n={o['latency']['cache_hits'].get('n', 0)}) | n/a | {ms(o['latency']['cache_hits'], 'mean_ms')} |",
        "",
        "Answers judged correct by category (all expected keywords present for supported questions; declined for unsupported ones):",
        "",
        "| Category | Baseline | Optimized |",
        "|---|---:|---:|",
    ]
    for cat in b["answer_correct_by_category"]:
        bc, oc = b["answer_correct_by_category"][cat], o["answer_correct_by_category"].get(cat, {"n": 0, "correct": 0})
        lines.append(f"| {cat} | {bc['correct']}/{bc['n']} | {oc['correct']}/{oc['n']} |")
    lines += [
        "",
        "Ablation (same workload; isolates where the estimated cost change comes from):",
        "",
        "| Configuration | Cache | Routing | Model calls | Est. cost (USD) | vs baseline | Correct (all categories) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for cfg in [base] + ablations + [opt]:
        s_ = cfg["summary"]
        correct = sum(v["correct"] for v in s_["answer_correct_by_category"].values())
        total = sum(v["n"] for v in s_["answer_correct_by_category"].values())
        change = (
            "-" if cfg is base or not b["estimated_cost_total"]
            else f"{(s_['estimated_cost_total'] / b['estimated_cost_total'] - 1) * 100:+.1f}%"
        )
        lines.append(
            f"| {cfg['name']} | {'on' if cfg['cache_enabled'] else 'off'} | {cfg['policy']} | "
            f"{s_['model_calls']} | {s_['estimated_cost_total']:.6f} | {change} | {correct}/{total} |"
        )
    dist = meta["score_distribution"]
    lines += [
        "",
        f"Cold-phase top retrieval scores (optimized run): supported min/median {dist['supported_top_score']}, "
        f"unsupported max/median {dist['unsupported_top_score']}.",
        f"Cache rejection reasons seen: {o['cache']['rejections'] or 'none'}.",
        "",
    ]
    if meta["generator"] == "extractive-stub":
        lines += [
            "**Reading these numbers.** The generator is an offline extractive stub, so answer correctness reflects",
            "whether retrieval surfaced the right sentence, not LLM quality; token counts are ~4-characters-per-token",
            "estimates and generation latency is near zero, so latency differences here reflect only the pipeline",
            "(embedding, cache lookup, retrieval, SQLite writes). Cost figures apply configured prices to those",
            "estimated tokens. Re-run with `--llm groq` for provider-reported tokens and real latency.",
        ]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embedding", default="simulated", choices=["simulated", "sentence-transformers"])
    parser.add_argument("--llm", default="stub", choices=["stub", "groq"])
    parser.add_argument("--baseline-policy", default="fixed-complex", choices=list(CostAwareRouter.POLICIES))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-relevance", type=float, default=None,
                        help="default: 0.30 for sentence-transformers, 0.15 for the hashing embedder")
    parser.add_argument("--cache-threshold", type=float, default=None,
                        help="default: 0.82 for sentence-transformers, 0.80 for the hashing embedder")
    parser.add_argument("--tag", default="")
    args = parser.parse_args(argv)
    simulated = args.embedding == "simulated"
    if args.min_relevance is None:
        args.min_relevance = 0.15 if simulated else 0.30
    if args.cache_threshold is None:
        args.cache_threshold = 0.80 if simulated else 0.82

    dataset = json.loads((EVAL_DIR / "dataset.json").read_text(encoding="utf-8"))
    _DATASET_INDEX.extend(q for p in dataset["phases"] for q in p["queries"])
    from eval.build_corpus import build

    build()

    base = run_config(args, "baseline", args.baseline_policy, use_cache=False, dataset=dataset)
    opt = run_config(args, "optimized", "complexity", use_cache=True, dataset=dataset)
    # Ablations separate the routing effect from the caching effect.
    cache_only = run_config(args, "cache_only", args.baseline_policy, use_cache=True, dataset=dataset)
    routing_only = run_config(args, "routing_only", "complexity", use_cache=False, dataset=dataset)
    for cfg in (base, opt, cache_only, routing_only):
        cfg["summary"] = summarize(cfg)

    router = CostAwareRouter()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + (f"-{args.tag}" if args.tag else "")
    provider = get_embedding_provider(args.embedding, settings.EMBEDDING_MODEL_NAME, settings.SIMULATED_EMBEDDING_DIM)
    meta = {
        "run_id": run_id,
        "embedding_provider": provider.provider_name,
        "generator": "groq" if args.llm == "groq" else ExtractiveStub.name,
        "top_k": args.top_k,
        "min_relevance": args.min_relevance,
        "cache_threshold": args.cache_threshold,
        "phase_counts": {p["name"]: len(p["queries"]) for p in dataset["phases"]},
        "prices": {m.model_id: [m.cost_input_per_1k, m.cost_output_per_1k] for m in router.profiles},
        "score_distribution": score_distribution(opt["rows"]),
    }
    RESULTS.mkdir(exist_ok=True)
    stem = f"{run_id}-{'stub' if args.llm == 'stub' else 'groq'}-{'hashing' if simulated else 'minilm'}"
    (RESULTS / f"{stem}.json").write_text(json.dumps(
        {"meta": meta, "baseline": base, "optimized": opt, "cache_only": cache_only, "routing_only": routing_only},
        indent=2,
    ))
    report = render_markdown(meta, base, opt, [cache_only, routing_only])
    (RESULTS / f"{stem}.md").write_text(report)
    print(report)
    print(f"Wrote eval/results/{stem}.json and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
