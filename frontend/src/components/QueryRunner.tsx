
import React, { useState } from 'react';
import {
  Send,
  Zap,
  Sliders,
  CheckCircle2,
  AlertTriangle,
  Sparkles,
} from 'lucide-react';
import { QueryResponse } from '../api/types';

interface QueryRunnerProps {
  onRunQuery: (
    query: string,
    threshold: number,
    forceRefresh: boolean
  ) => Promise<QueryResponse>;
  isLoading: boolean;
}

const PRESET_QUERIES = [
  {
    label: '1. Simple Query (Fast Tier)',
    query: 'What is semantic caching in RAG pipelines?',
    desc: 'Routes to Groq lightweight model (GPT-OSS 20B)',
  },
  {
    label: '2. Complex Reasoning (Capable Tier)',
    query:
      'Compare the tradeoffs between semantic caching and prompt distillation for latency sensitive multi-tenant systems.',
    desc: 'Routes to Groq reasoning model (GPT-OSS 120B)',
  },
  {
    label: '3. Paraphrase Cache Hit',
    query: 'Can you explain how semantic caching works in RAG pipelines?',
    desc: 'Tests semantic similarity matching and cache reuse',
  },
];

export const QueryRunner: React.FC<QueryRunnerProps> = ({
  onRunQuery,
  isLoading,
}) => {
  const [query, setQuery] = useState('');
  const [threshold, setThreshold] = useState(0.82);
  const [forceRefresh, setForceRefresh] = useState(false);
  const [lastResult, setLastResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!query.trim() || isLoading) return;

    setError(null);
    try {
      const res = await onRunQuery(query, threshold, forceRefresh);
      setLastResult(res);
    } catch (err: unknown) {
      setError(
        err instanceof Error ? err.message : 'Execution error'
      );
    }
  };

  const selectPreset = (q: string) => {
    setQuery(q);
    setError(null);
  };

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5">
      <div className="flex items-center justify-between pb-4 border-b border-slate-800/80">
        <div className="flex items-center space-x-2">
          <Sparkles className="w-4 h-4 text-emerald-400" />
          <h2 className="text-sm font-semibold text-slate-200">
            Interactive Pipeline Runner
          </h2>
        </div>

        <span className="text-[11px] text-slate-400 font-mono">
          Groq LLM Inference • Sentence Transformers Cache • SQLite Telemetry
        </span>
      </div>

      <div className="mt-4">
        <label className="text-xs text-slate-400 mb-2 block">
          Quick Test Scenarios:
        </label>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
          {PRESET_QUERIES.map((preset, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => selectPreset(preset.query)}
              className="text-left p-2.5 rounded-lg border border-slate-800 bg-slate-950/60 hover:bg-slate-800/60 hover:border-slate-700 transition-all text-xs"
            >
              <div className="font-medium text-slate-300">
                {preset.label}
              </div>
              <div className="text-[11px] text-slate-400 mt-0.5 line-clamp-1">
                {preset.desc}
              </div>
            </button>
          ))}
        </div>
      </div>

      <form onSubmit={handleSubmit} className="mt-4 space-y-3">
        <div className="relative">
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter a question to process through semantic caching and intelligent model routing..."
            rows={2}
            className="w-full bg-slate-950 border border-slate-800 rounded-lg p-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-emerald-500/60 focus:ring-1 focus:ring-emerald-500/40 resize-none font-sans"
          />
        </div>

        <div className="flex flex-wrap items-center justify-between gap-4 pt-1">
          <div className="flex items-center space-x-6 text-xs text-slate-400">
            <div className="flex items-center space-x-2">
              <Sliders className="w-3.5 h-3.5 text-slate-400" />
              <span>Similarity Threshold:</span>
              <span className="font-mono text-emerald-400 font-semibold">
                {threshold.toFixed(2)}
              </span>
              <input
                type="range"
                min="0.50"
                max="0.99"
                step="0.01"
                value={threshold}
                onChange={(e) =>
                  setThreshold(parseFloat(e.target.value))
                }
                className="w-24 accent-emerald-500 bg-slate-800 cursor-pointer"
              />
            </div>

            <label className="flex items-center space-x-2 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={forceRefresh}
                onChange={(e) => setForceRefresh(e.target.checked)}
                className="rounded border-slate-700 bg-slate-950 text-emerald-500 focus:ring-0 focus:ring-offset-0"
              />
              <span
                className={
                  forceRefresh ? 'text-amber-400' : 'text-slate-400'
                }
              >
                Bypass Cache (Force Refresh)
              </span>
            </label>
          </div>

          <button
            type="submit"
            disabled={!query.trim() || isLoading}
            className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-xs font-semibold rounded-lg flex items-center space-x-2 transition-all shadow-sm shadow-emerald-950"
          >
            {isLoading ? (
              <span>Processing...</span>
            ) : (
              <>
                <span>Run Query</span>
                <Send className="w-3 h-3" />
              </>
            )}
          </button>
        </div>
      </form>

      {error && (
        <div className="mt-4 p-3 bg-rose-950/40 border border-rose-900/60 rounded-lg text-rose-300 text-xs flex items-center space-x-2">
          <AlertTriangle className="w-4 h-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {lastResult && (
        <div className="mt-4 p-4 rounded-lg bg-slate-950 border border-slate-800 space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800/80 pb-3">
            <div className="flex items-center space-x-2">
              {lastResult.cache_hit ? (
                <span className="px-2.5 py-1 rounded bg-emerald-950/80 border border-emerald-800/80 text-emerald-400 text-xs font-mono font-semibold flex items-center space-x-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  <span>CACHE HIT</span>
                </span>
              ) : (
                <span className="px-2.5 py-1 rounded bg-amber-950/80 border border-amber-800/80 text-amber-400 text-xs font-mono font-semibold flex items-center space-x-1.5">
                  <Zap className="w-3.5 h-3.5" />
                  <span>CACHE MISS</span>
                </span>
              )}

              {lastResult.cached_similarity !== null &&
                lastResult.cached_similarity !== undefined && (
                  <span className="text-xs font-mono text-slate-400">
                    Sim:{' '}
                    <strong className="text-slate-200">
                      {(lastResult.cached_similarity * 100).toFixed(1)}%
                    </strong>
                  </span>
                )}

              <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-300">
                Model: <strong>{lastResult.selected_model}</strong>
              </span>
            </div>

            <div className="flex items-center space-x-4 text-xs font-mono text-slate-400">
              <span>
                Latency:{' '}
                <strong className="text-slate-200">
                  {lastResult.latency_ms} ms
                </strong>
              </span>
              <span>
                Est. Cost:{' '}
                <strong
                  className={
                    lastResult.cache_hit
                      ? 'text-emerald-400'
                      : 'text-amber-400'
                  }
                >
                  ${lastResult.estimated_cost.toFixed(6)}
                </strong>
              </span>
            </div>
          </div>

          <div className="text-xs">
            <div className="text-slate-400 font-mono mb-1">
              Routing Telemetry &amp; Reason:
            </div>
            <div className="p-2.5 bg-slate-900/90 rounded border border-slate-800 text-slate-300 font-mono text-[11px] leading-relaxed">
              {lastResult.routing_reason}
            </div>
          </div>

          <div className="text-xs">
            <div className="text-slate-400 font-mono mb-1">
              Model Response:
            </div>
            <pre className="p-3 bg-slate-900/60 rounded border border-slate-800 text-slate-200 text-xs font-mono whitespace-pre-wrap leading-relaxed">
              {lastResult.response}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
};
