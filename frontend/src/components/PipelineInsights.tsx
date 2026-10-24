import React from 'react';
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { MetricsSummaryExtended } from '../api/types';
import { fmtMs, fmtUsd, OUTCOME_LABELS } from '../lib/format';

const tooltipStyle = {
  backgroundColor: '#090d16',
  borderColor: '#1e293b',
  borderRadius: '0.375rem',
  fontSize: '12px',
  fontFamily: 'JetBrains Mono',
};

export const PipelineInsights: React.FC<{ metrics: MetricsSummaryExtended | null }> = ({ metrics }) => {
  if (!metrics || metrics.total_requests === 0) return null;

  const stages = [
    { stage: 'Embedding', ms: metrics.avg_embedding_ms },
    { stage: 'Cache lookup', ms: metrics.avg_cache_lookup_ms },
    { stage: 'Retrieval', ms: metrics.avg_retrieval_ms },
    { stage: 'Generation', ms: metrics.avg_generation_ms },
  ];
  const measured = stages.filter((s) => s.ms !== null) as { stage: string; ms: number }[];
  const outcomes = Object.entries(metrics.outcome_breakdown);
  const models = Object.entries(metrics.model_stats);

  return (
    <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">Average stage latency</h3>
        <p className="mt-0.5 font-mono text-[11px] text-slate-400">Measured per request; averaged where the stage ran</p>
        {measured.length === 0 ? (
          <p className="mt-6 text-center text-xs text-slate-500">No stage timings recorded yet.</p>
        ) : (
          <div className="mt-3 h-44">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={measured} layout="vertical" margin={{ left: 10, right: 16 }}>
                <XAxis type="number" stroke="#64748b" fontSize={10} unit=" ms" tickLine={false} />
                <YAxis type="category" dataKey="stage" stroke="#64748b" fontSize={10} width={80} tickLine={false} />
                <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => fmtMs(v)} />
                <Bar dataKey="ms" fill="#38bdf8" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
        <div className="mt-2 grid grid-cols-2 gap-2 font-mono text-[11px] text-slate-400">
          <span>Hit avg: <strong className="text-slate-200">{fmtMs(metrics.avg_latency_cache_hit_ms)}</strong></span>
          <span>Miss avg: <strong className="text-slate-200">{fmtMs(metrics.avg_latency_cache_miss_ms)}</strong></span>
        </div>
      </div>

      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">Answer outcomes</h3>
        <p className="mt-0.5 font-mono text-[11px] text-slate-400">Grounding result per request</p>
        <ul className="mt-3 space-y-2 text-xs">
          {outcomes.length === 0 && <li className="text-slate-500">No outcomes recorded.</li>}
          {outcomes.map(([k, v]) => (
            <li key={k} className="flex items-center gap-2">
              <span className="w-40 text-slate-300">{OUTCOME_LABELS[k] ?? k}</span>
              <div className="h-2 flex-1 overflow-hidden rounded bg-slate-800">
                <div
                  className={`h-full ${k === 'answered' ? 'bg-emerald-500' : k === 'error' ? 'bg-rose-500' : 'bg-amber-500'}`}
                  style={{ width: `${(v / metrics.total_requests) * 100}%` }}
                />
              </div>
              <span className="w-8 text-right font-mono text-slate-400">{v}</span>
            </li>
          ))}
        </ul>
        <div className="mt-4 space-y-1 font-mono text-[11px] text-slate-400">
          <div>Cache matches: exact {metrics.match_type_breakdown.exact ?? 0} · semantic {metrics.match_type_breakdown.semantic ?? 0}</div>
          <div>Model fallbacks: {metrics.fallback_count}</div>
          {Object.keys(metrics.error_category_breakdown).length > 0 && (
            <div>
              Errors: {Object.entries(metrics.error_category_breakdown).map(([k, v]) => `${k} ${v}`).join(' · ')}
            </div>
          )}
        </div>
      </div>

      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 overflow-x-auto">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">Fresh generations by model</h3>
        <p className="mt-0.5 font-mono text-[11px] text-slate-400">Provider-reported tokens · estimated cost</p>
        {models.length === 0 ? (
          <p className="mt-6 text-center text-xs text-slate-500">No model calls yet.</p>
        ) : (
          <table className="mt-3 w-full text-left font-mono text-[11px]">
            <thead className="text-slate-500">
              <tr>
                <th className="pb-1 pr-2">Model</th>
                <th className="pb-1 pr-2 text-right">Req</th>
                <th className="pb-1 pr-2 text-right">Avg</th>
                <th className="pb-1 pr-2 text-right">Tokens</th>
                <th className="pb-1 text-right">Est. $</th>
              </tr>
            </thead>
            <tbody className="text-slate-300">
              {models.map(([name, s]) => (
                <tr key={name}>
                  <td className="py-0.5 pr-2 max-w-[9rem] truncate" title={name}>{name}</td>
                  <td className="py-0.5 pr-2 text-right">{s.requests}</td>
                  <td className="py-0.5 pr-2 text-right">{fmtMs(s.avg_latency_ms)}</td>
                  <td className="py-0.5 pr-2 text-right">{s.input_tokens + s.output_tokens}</td>
                  <td className="py-0.5 text-right">{fmtUsd(s.total_estimated_cost)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <p className="mt-3 text-[11px] leading-relaxed text-slate-500">{metrics.pricing_note}</p>
      </div>
    </div>
  );
};
