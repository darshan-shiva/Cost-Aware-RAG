import React, { useState } from 'react';
import { AlertTriangle, CheckCircle2, ChevronDown, ChevronRight, Database, ShieldAlert, Zap } from 'lucide-react';
import { RAGQueryResponse } from '../../api/types';
import { fmtMs, fmtUsd, OUTCOME_LABELS } from '../../lib/format';
import { AnswerText } from './AnswerText';
import { CitationList } from './CitationList';

const OUTCOME_STYLES: Record<string, string> = {
  answered: 'border-emerald-800/70 bg-emerald-950/50 text-emerald-300',
  insufficient_evidence: 'border-amber-800/70 bg-amber-950/50 text-amber-300',
  uncited: 'border-rose-800/70 bg-rose-950/50 text-rose-300',
  no_documents: 'border-slate-700 bg-slate-900 text-slate-300',
};

const chip = 'inline-flex items-center gap-1 rounded border px-2 py-0.5 font-mono text-[10px] font-semibold';

export const AnswerCard: React.FC<{ result: RAGQueryResponse }> = ({ result }) => {
  const [openLabel, setOpenLabel] = useState<string | null>(null);
  const [showDetails, setShowDetails] = useState(false);
  const { cache, routing, usage, latency, retrieval } = result;
  const toggle = (label: string) => setOpenLabel((cur) => (cur === label ? null : label));

  return (
    <article className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 space-y-3" data-testid="answer-card">
      <div className="flex flex-wrap items-center gap-2">
        <span className={`${chip} ${OUTCOME_STYLES[result.outcome] ?? OUTCOME_STYLES.no_documents}`}>
          {result.outcome === 'answered' ? <CheckCircle2 className="h-3 w-3" /> : <AlertTriangle className="h-3 w-3" />}
          {OUTCOME_LABELS[result.outcome] ?? result.outcome}
        </span>
        {cache.hit ? (
          <span className={`${chip} border-emerald-800/70 bg-emerald-950/50 text-emerald-300`}>
            <Database className="h-3 w-3" />
            CACHED · {cache.match_type}
            {cache.similarity != null && cache.match_type === 'semantic' ? ` · sim ${cache.similarity.toFixed(3)}` : ''}
          </span>
        ) : routing.selected_model ? (
          <span className={`${chip} border-amber-800/70 bg-amber-950/50 text-amber-300`}>
            <Zap className="h-3 w-3" />
            FRESH · {routing.selected_model}
          </span>
        ) : (
          <span className={`${chip} border-slate-700 bg-slate-900 text-slate-400`}>NO MODEL CALL</span>
        )}
        {routing.fallback_used && (
          <span className={`${chip} border-purple-800/70 bg-purple-950/50 text-purple-300`}>FALLBACK</span>
        )}
        {retrieval.suspicious_passages > 0 && (
          <span
            className={`${chip} border-rose-800/70 bg-rose-950/50 text-rose-300`}
            title="Some retrieved passages contain instruction-like text. They were passed to the model as untrusted content."
          >
            <ShieldAlert className="h-3 w-3" />
            {retrieval.suspicious_passages} suspicious passage{retrieval.suspicious_passages > 1 ? 's' : ''}
          </span>
        )}
        <span className="ml-auto font-mono text-[11px] text-slate-400">
          {fmtMs(latency.total_ms)} · est. {fmtUsd(usage.estimated_cost)}
          {usage.estimated_cost_avoided > 0 && ` · est. avoided ${fmtUsd(usage.estimated_cost_avoided)}`}
        </span>
      </div>

      <AnswerText text={result.answer} citations={result.citations} onCite={toggle} />

      <CitationList citations={result.citations} openLabel={openLabel} onToggle={toggle} />

      <button
        type="button"
        onClick={() => setShowDetails((v) => !v)}
        aria-expanded={showDetails}
        className="flex items-center gap-1 text-[11px] font-mono text-slate-400 hover:text-slate-200"
      >
        {showDetails ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
        Pipeline details
      </button>

      {showDetails && (
        <div className="grid gap-3 text-[11px] font-mono md:grid-cols-2">
          <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-3 space-y-1">
            <div className="uppercase tracking-wider text-slate-500">Latency (measured)</div>
            <Row k="Embedding" v={fmtMs(latency.embedding_ms)} />
            <Row k="Cache lookup" v={fmtMs(latency.cache_lookup_ms)} />
            <Row k="Retrieval" v={fmtMs(latency.retrieval_ms)} />
            <Row k="Generation" v={fmtMs(latency.generation_ms)} />
            <Row k="Total" v={fmtMs(latency.total_ms)} />
          </div>
          <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-3 space-y-1">
            <div className="uppercase tracking-wider text-slate-500">Tokens and cost</div>
            <Row k="Input tokens" v={usage.token_source === 'provider' ? String(usage.input_tokens) : 'none used'} />
            <Row k="Output tokens" v={usage.token_source === 'provider' ? String(usage.output_tokens) : 'none used'} />
            <Row k="Estimated cost" v={fmtUsd(usage.estimated_cost)} />
            <Row k="Actual billed cost" v="unavailable" />
            {cache.hit && <Row k="Est. cost avoided" v={fmtUsd(usage.estimated_cost_avoided)} />}
          </div>
          <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-3 space-y-1 md:col-span-2">
            <div className="uppercase tracking-wider text-slate-500">Routing and cache</div>
            <p className="font-sans text-xs leading-relaxed text-slate-300">{routing.reason}</p>
            {!cache.hit && cache.rejection_reason && (
              <p className="text-amber-300/90">
                Cache candidate rejected (similarity {cache.similarity?.toFixed(3)}): {cache.rejection_reason}
              </p>
            )}
            {cache.bypassed && <p className="text-slate-400">Cache bypassed by request.</p>}
          </div>
          {retrieval.sources.length > 0 && (
            <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-3 md:col-span-2 overflow-x-auto">
              <div className="mb-1 uppercase tracking-wider text-slate-500">
                Retrieved passages (top {retrieval.top_k}, threshold {retrieval.min_relevance})
                {!retrieval.performed && cache.hit ? ' · from original request' : ''}
              </div>
              <table className="w-full text-left">
                <thead className="text-slate-500">
                  <tr>
                    <th className="py-1 pr-3">Label</th>
                    <th className="py-1 pr-3">Document</th>
                    <th className="py-1 pr-3">Page</th>
                    <th className="py-1 pr-3 text-right">Score</th>
                    <th className="py-1">Use</th>
                  </tr>
                </thead>
                <tbody className="text-slate-300">
                  {retrieval.sources.map((s) => (
                    <tr key={s.chunk_id} className={s.relevance_score < retrieval.min_relevance ? 'text-slate-500' : ''}>
                      <td className="py-0.5 pr-3">{s.label ?? '-'}</td>
                      <td className="py-0.5 pr-3 max-w-[14rem] truncate">{s.filename}</td>
                      <td className="py-0.5 pr-3">{s.page_number}</td>
                      <td className="py-0.5 pr-3 text-right">{s.relevance_score.toFixed(3)}</td>
                      <td className="py-0.5">{s.cited
                          ? 'cited'
                          : s.in_prompt
                            ? 'in prompt'
                            : s.relevance_score < retrieval.min_relevance
                              ? 'below threshold'
                              : 'not in prompt'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </article>
  );
};

const Row: React.FC<{ k: string; v: string }> = ({ k, v }) => (
  <div className="flex justify-between gap-3">
    <span className="text-slate-400">{k}</span>
    <span className="text-slate-200">{v}</span>
  </div>
);
