import React, { useState } from 'react';
import { CheckCircle2, Zap, AlertCircle, ChevronDown, ChevronUp, Clock, DollarSign } from 'lucide-react';
import { RequestRecordExtended } from '../api/types';
import { fmtMs, OUTCOME_LABELS } from '../lib/format';

interface RequestHistoryTableProps {
  requests: RequestRecordExtended[];
  isLoading: boolean;
}

export const RequestHistoryTable: React.FC<RequestHistoryTableProps> = ({ requests, isLoading }) => {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const toggleExpand = (id: string) => {
    setExpandedId(expandedId === id ? null : id);
  };

  const formatTimestamp = (iso: string) => {
    try {
      const d = new Date(iso);
      return d.toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch {
      return iso;
    }
  };

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-xl overflow-hidden">
      <div className="p-5 border-b border-slate-800 flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-slate-200">Execution Telemetry Log</h2>
          <p className="text-xs text-slate-400 font-mono mt-0.5">
            Persisted in SQLite • Audit trail of cache lookups, routing decisions, & estimated costs
          </p>
        </div>
        <span className="text-xs font-mono px-2.5 py-1 rounded bg-slate-950 border border-slate-800 text-slate-400">
          Showing {requests.length} records
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-slate-800/80 bg-slate-950/40 text-[11px] font-mono uppercase tracking-wider text-slate-400">
              <th className="py-3 px-4">Time</th>
              <th className="py-3 px-4">Status</th>
              <th className="py-3 px-4">Mode</th>
              <th className="py-3 px-4">Query</th>
              <th className="py-3 px-4">Model</th>
              <th className="py-3 px-4 text-right">Latency</th>
              <th className="py-3 px-4 text-right">Tokens</th>
              <th className="py-3 px-4 text-right">Est. Cost</th>
              <th className="py-3 px-3 text-center">Details</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 text-xs font-mono">
            {requests.length === 0 ? (
              <tr>
                <td colSpan={9} className="py-12 text-center text-slate-400">
                  <div className="flex flex-col items-center justify-center space-y-2">
                    <Clock className="w-6 h-6 text-slate-400" />
                    <span>No requests logged yet. Execute a query to view real-time telemetry.</span>
                  </div>
                </td>
              </tr>
            ) : (
              requests.map((record) => {
                const isExpanded = expandedId === record.request_id;
                return (
                  <React.Fragment key={record.request_id}>
                    <tr
                      onClick={() => toggleExpand(record.request_id)}
                      className="hover:bg-slate-800/30 transition-colors cursor-pointer group"
                    >
                      {/* Time */}
                      <td className="py-3 px-4 text-slate-400 text-[11px] whitespace-nowrap">
                        {formatTimestamp(record.timestamp)}
                      </td>

                      {/* Cache Status Badge */}
                      <td className="py-3 px-4 whitespace-nowrap">
                        {record.has_error ? (
                          <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[10px] font-semibold bg-rose-950/70 border border-rose-800/80 text-rose-400">
                            <AlertCircle className="w-3 h-3" />
                            <span>ERROR</span>
                          </span>
                        ) : record.cache_hit ? (
                          <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-950/70 border border-emerald-800/80 text-emerald-400">
                            <CheckCircle2 className="w-3 h-3" />
                            <span>HIT{record.cache_match_type ? ` · ${record.cache_match_type}` : ''}</span>
                          </span>
                        ) : (
                          <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[10px] font-semibold bg-amber-950/70 border border-amber-800/80 text-amber-400">
                            <Zap className="w-3 h-3" />
                            <span>MISS</span>
                          </span>
                        )}
                      </td>

                      {/* Mode / outcome */}
                      <td className="py-3 px-4 whitespace-nowrap text-[11px]">
                        <span className={record.mode === 'rag' ? 'text-sky-300' : 'text-slate-400'}>
                          {record.mode === 'rag' ? 'RAG' : 'General'}
                        </span>
                        {record.outcome && record.outcome !== 'answered' && (
                          <span className={`ml-1 ${record.outcome === 'error' ? 'text-rose-400' : 'text-amber-400'}`}>
                            · {OUTCOME_LABELS[record.outcome] ?? record.outcome}
                          </span>
                        )}
                      </td>

                      {/* Query preview */}
                      <td className="py-3 px-4 text-slate-200 font-sans max-w-xs truncate" title={record.query}>
                        {record.query}
                      </td>

                      {/* Model */}
                      <td className="py-3 px-4 text-slate-300 whitespace-nowrap">
                        <span className="px-2 py-0.5 rounded bg-slate-800/80 border border-slate-700/60 text-[11px]">
                          {record.selected_model}
                        </span>
                      </td>

                      {/* Latency */}
                      <td className="py-3 px-4 text-right whitespace-nowrap text-slate-300">
                        {record.latency_ms.toFixed(1)} ms
                      </td>

                      {/* Tokens */}
                      <td className="py-3 px-4 text-right whitespace-nowrap text-slate-400 text-[11px]">
                        {record.estimated_input_tokens + record.estimated_output_tokens}
                      </td>

                      {/* Est Cost */}
                      <td className="py-3 px-4 text-right whitespace-nowrap font-semibold">
                        <span className={record.cache_hit ? 'text-emerald-400' : 'text-slate-300'}>
                          ${record.estimated_cost.toFixed(6)}
                        </span>
                      </td>

                      {/* Expand Toggle */}
                      <td className="py-3 px-3 text-center text-slate-400 group-hover:text-slate-300">
                        {isExpanded ? <ChevronUp className="w-4 h-4 mx-auto" /> : <ChevronDown className="w-4 h-4 mx-auto" />}
                      </td>
                    </tr>

                    {/* Expanded details row */}
                    {isExpanded && (
                      <tr className="bg-slate-950/80 border-b border-slate-800">
                        <td colSpan={9} className="p-4 space-y-3 font-sans text-xs">
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                              <span className="text-[11px] font-mono uppercase text-slate-400 block mb-1">
                                Full Request Query:
                              </span>
                              <div className="p-2.5 bg-slate-900 rounded border border-slate-800 text-slate-200 text-xs">
                                {record.query}
                              </div>
                            </div>

                            <div>
                              <span className="text-[11px] font-mono uppercase text-slate-400 block mb-1">
                                Routing Decision & Rationale:
                              </span>
                              <div className="p-2.5 bg-slate-900 rounded border border-slate-800 text-slate-300 font-mono text-[11px] leading-relaxed">
                                {record.routing_reason}
                              </div>
                            </div>
                          </div>

                          {record.mode === 'rag' && (
                            <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] font-mono text-slate-400">
                              <span>Collection: <span className="text-slate-300">{record.collection ?? 'n/a'}</span></span>
                              <span>Embed {fmtMs(record.embedding_ms)}</span>
                              <span>Cache {fmtMs(record.cache_lookup_ms)}</span>
                              <span>Retrieval {fmtMs(record.retrieval_ms)}</span>
                              <span>Generation {fmtMs(record.generation_ms)}</span>
                              <span>Chunks {record.retrieved_chunk_count ?? 'n/a'}</span>
                              <span>Top score {record.top_retrieval_score?.toFixed(3) ?? 'n/a'}</span>
                              <span>Citations {record.citation_count ?? 'n/a'}</span>
                              {record.fallback_model && <span className="text-purple-300">Fallback → {record.fallback_model}</span>}
                              {!!record.estimated_cost_avoided && <span className="text-emerald-300">Est. avoided ${record.estimated_cost_avoided.toFixed(6)}</span>}
                            </div>
                          )}
                          {record.has_error && (
                            <div className="text-[11px] font-mono text-rose-300">
                              Error: {record.error_category ?? 'unknown'} {record.error_message ? `(${record.error_message})` : ''}
                            </div>
                          )}
                          <div className="flex flex-wrap items-center justify-between text-[11px] font-mono text-slate-400 pt-2 border-t border-slate-800/80">
                            <div>Request ID: <span className="text-slate-300">{record.request_id}</span></div>
                            <div className="flex items-center space-x-4">
                              <span>Tokens: {record.estimated_input_tokens} In / {record.estimated_output_tokens} Out</span>
                              <span>Similarity: {record.cached_similarity !== null && record.cached_similarity !== undefined ? `${(record.cached_similarity * 100).toFixed(1)}%` : 'N/A'}</span>
                              <span>Actual billed cost: <strong className="text-slate-400">unavailable (estimates only)</strong></span>
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
