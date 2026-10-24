import React, { useEffect, useState } from 'react';
import { AlertTriangle, BookOpen, Loader2, Send, Settings2 } from 'lucide-react';
import { submitRagQuery } from '../../api/client';
import { DocumentInfo, PublicConfig, RAGQueryResponse } from '../../api/types';
import { AnswerCard } from './AnswerCard';

interface AskPanelProps {
  collection: string;
  collections: string[];
  onCollectionChange: (name: string) => void;
  documents: DocumentInfo[];
  config: PublicConfig | null;
  onAsked: () => void;
  onGoToDocuments: () => void;
}

interface Turn {
  id: number;
  question: string;
  collection: string;
  result?: RAGQueryResponse;
  error?: string;
}

let turnId = 0;

export const AskPanel: React.FC<AskPanelProps> = ({
  collection,
  collections,
  onCollectionChange,
  documents,
  config,
  onAsked,
  onGoToDocuments,
}) => {
  const [question, setQuestion] = useState('');
  const [turns, setTurns] = useState<Turn[]>([]);
  const [pending, setPending] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [docFilter, setDocFilter] = useState<string[]>([]);
  const [topK, setTopK] = useState<number | null>(null);
  const [minRelevance, setMinRelevance] = useState<number | null>(null);
  const [cacheThreshold, setCacheThreshold] = useState<number | null>(null);
  const [forceRefresh, setForceRefresh] = useState(false);

  const ready = documents.filter((d) => d.status === 'ready' && d.collection === collection);
  const processing = documents.filter(
    (d) => d.collection === collection && (d.status === 'pending' || d.status === 'processing')
  );

  // Drop filters that point at documents no longer in this collection.
  useEffect(() => {
    setDocFilter((cur) => cur.filter((id) => ready.some((d) => d.document_id === id)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [collection, documents]);

  const effTopK = topK ?? config?.retrieval_top_k ?? 5;
  const effMinRel = minRelevance ?? config?.retrieval_min_relevance ?? 0.3;
  const effThreshold = cacheThreshold ?? config?.cache_similarity_threshold ?? 0.8;

  const ask = async (e?: React.FormEvent) => {
    e?.preventDefault();
    const q = question.trim();
    if (!q || pending) return;
    const id = ++turnId;
    setTurns((t) => [{ id, question: q, collection }, ...t]);
    setPending(true);
    try {
      const result = await submitRagQuery({
        query: q,
        collection,
        document_ids: docFilter.length ? docFilter : undefined,
        top_k: topK ?? undefined,
        min_relevance: minRelevance ?? undefined,
        similarity_threshold: cacheThreshold ?? undefined,
        force_refresh: forceRefresh,
      });
      setTurns((t) => t.map((x) => (x.id === id ? { ...x, result } : x)));
      setQuestion('');
      onAsked();
    } catch (err) {
      const message = err instanceof Error ? err.message : 'The question could not be answered.';
      setTurns((t) => t.map((x) => (x.id === id ? { ...x, error: message } : x)));
    } finally {
      setPending(false);
    }
  };

  const toggleDoc = (id: string) =>
    setDocFilter((cur) => (cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id]));

  return (
    <div className="space-y-5">
      <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800/80 pb-4">
          <div className="flex items-center gap-2">
            <BookOpen className="h-4 w-4 text-emerald-400" />
            <h2 className="text-sm font-semibold text-slate-200">Ask your documents</h2>
            <span className="rounded border border-emerald-800/60 bg-emerald-950/40 px-1.5 py-0.5 font-mono text-[10px] text-emerald-300">
              GROUNDED · CITED
            </span>
          </div>
          <label className="flex items-center gap-2 text-xs text-slate-400">
            Knowledge source
            <select
              aria-label="Knowledge collection"
              value={collection}
              onChange={(e) => onCollectionChange(e.target.value)}
              className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1 font-mono text-xs text-slate-200 focus:border-emerald-500/60 focus:outline-none"
            >
              {collections.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </label>
        </div>

        {ready.length === 0 ? (
          <div className="py-10 text-center">
            <p className="text-sm text-slate-300">
              {processing.length > 0
                ? `${processing.length} document${processing.length > 1 ? 's are' : ' is'} still being processed in "${collection}".`
                : `No indexed documents in "${collection}" yet.`}
            </p>
            <button
              type="button"
              onClick={onGoToDocuments}
              className="mt-3 rounded-lg border border-slate-700 bg-slate-800/80 px-3 py-1.5 text-xs text-slate-200 hover:bg-slate-700/80"
            >
              Upload a PDF or TXT file
            </button>
          </div>
        ) : (
          <form onSubmit={ask} className="mt-4 space-y-3">
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  ask();
                }
              }}
              rows={2}
              maxLength={4000}
              aria-label="Question"
              placeholder={`Ask a question about the ${ready.length} document${ready.length > 1 ? 's' : ''} in "${collection}"…`}
              className="w-full resize-none rounded-lg border border-slate-800 bg-slate-950 p-3 text-sm text-slate-100 placeholder-slate-500 focus:border-emerald-500/60 focus:outline-none focus:ring-1 focus:ring-emerald-500/40"
            />

            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex flex-wrap items-center gap-2 text-xs text-slate-400">
                <span>Search in:</span>
                <button
                  type="button"
                  onClick={() => setDocFilter([])}
                  className={`rounded-full border px-2.5 py-0.5 ${docFilter.length === 0 ? 'border-emerald-700 bg-emerald-950/50 text-emerald-300' : 'border-slate-700 text-slate-400 hover:text-slate-200'}`}
                >
                  All documents
                </button>
                {ready.map((d) => (
                  <button
                    type="button"
                    key={d.document_id}
                    onClick={() => toggleDoc(d.document_id)}
                    aria-pressed={docFilter.includes(d.document_id)}
                    className={`max-w-[12rem] truncate rounded-full border px-2.5 py-0.5 ${docFilter.includes(d.document_id) ? 'border-sky-700 bg-sky-950/50 text-sky-300' : 'border-slate-700 text-slate-400 hover:text-slate-200'}`}
                  >
                    {d.filename}
                  </button>
                ))}
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setShowSettings((v) => !v)}
                  aria-expanded={showSettings}
                  className="flex items-center gap-1 rounded-lg border border-slate-700 bg-slate-800/60 px-2.5 py-2 text-xs text-slate-300 hover:bg-slate-700/60"
                >
                  <Settings2 className="h-3.5 w-3.5" />
                  Retrieval & cache
                </button>
                <button
                  type="submit"
                  disabled={!question.trim() || pending}
                  className="flex items-center gap-2 rounded-lg bg-emerald-600 px-4 py-2 text-xs font-semibold text-white hover:bg-emerald-500 disabled:opacity-50"
                >
                  {pending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Send className="h-3 w-3" />}
                  {pending ? 'Answering…' : 'Ask'}
                </button>
              </div>
            </div>

            {showSettings && (
              <div className="grid gap-4 rounded-lg border border-slate-800 bg-slate-950/60 p-3 text-xs text-slate-400 sm:grid-cols-2 lg:grid-cols-4">
                <Slider
                  label="Top-k passages"
                  value={effTopK}
                  min={1}
                  max={config?.retrieval_max_top_k ?? 12}
                  step={1}
                  onChange={setTopK}
                  format={(v) => String(v)}
                />
                <Slider
                  label="Min relevance"
                  value={effMinRel}
                  min={0}
                  max={0.9}
                  step={0.01}
                  onChange={setMinRelevance}
                  format={(v) => v.toFixed(2)}
                />
                <Slider
                  label="Cache similarity"
                  value={effThreshold}
                  min={0.5}
                  max={0.99}
                  step={0.01}
                  onChange={setCacheThreshold}
                  format={(v) => v.toFixed(2)}
                />
                <label className="flex cursor-pointer select-none items-center gap-2 self-end pb-1">
                  <input
                    type="checkbox"
                    checked={forceRefresh}
                    onChange={(e) => setForceRefresh(e.target.checked)}
                    className="rounded border-slate-700 bg-slate-950 text-emerald-500"
                  />
                  <span className={forceRefresh ? 'text-amber-400' : ''}>Bypass cache</span>
                </label>
              </div>
            )}
          </form>
        )}
      </section>

      {turns.length === 0 && ready.length > 0 && (
        <p className="text-center text-xs text-slate-500">
          Answers are generated only from retrieved passages and cite them as [S1], [S2]… Questions the documents do not
          support get an explicit insufficient-evidence reply.
        </p>
      )}

      <div className="space-y-4">
        {turns.map((t) => (
          <div key={t.id} className="space-y-2">
            <div className="flex items-start justify-end">
              <div className="max-w-[85%] rounded-xl rounded-tr-sm border border-slate-700 bg-slate-800/70 px-3 py-2 text-sm text-slate-100">
                {t.question}
                <div className="mt-0.5 text-right font-mono text-[10px] text-slate-500">{t.collection}</div>
              </div>
            </div>
            {t.result ? (
              <AnswerCard result={t.result} />
            ) : t.error ? (
              <div role="alert" className="flex items-center gap-2 rounded-lg border border-rose-900/60 bg-rose-950/40 p-3 text-xs text-rose-300">
                <AlertTriangle className="h-4 w-4 flex-shrink-0" />
                {t.error}
              </div>
            ) : (
              <div className="flex items-center gap-2 rounded-lg border border-slate-800 bg-slate-900/60 p-3 text-xs text-slate-400" role="status">
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                Retrieving passages and generating a grounded answer…
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

const Slider: React.FC<{
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format: (v: number) => string;
}> = ({ label, value, min, max, step, onChange, format }) => (
  <label className="space-y-1">
    <span className="flex justify-between">
      <span>{label}</span>
      <span className="font-mono text-emerald-400">{format(value)}</span>
    </span>
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      className="w-full accent-emerald-500"
    />
  </label>
);
