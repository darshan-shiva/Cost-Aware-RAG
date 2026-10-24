import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { Database, Zap, DollarSign, Timer, CircleAlert, PiggyBank } from 'lucide-react';
import { Header, TabId, TABS } from './components/Header';
import { MetricCard } from './components/MetricCard';
import { QueryRunner } from './components/QueryRunner';
import { ChartsSection } from './components/ChartsSection';
import { RequestHistoryTable } from './components/RequestHistoryTable';
import { PipelineInsights } from './components/PipelineInsights';
import { AskPanel } from './components/rag/AskPanel';
import { DocumentManager } from './components/documents/DocumentManager';

import {
  fetchCollections,
  fetchConfig,
  fetchDocuments,
  fetchHealth,
  fetchMetricsSummary,
  fetchRecentRequests,
  submitQuery,
  resetMetrics,
} from './api/client';
import {
  CollectionInfo,
  DocumentInfo,
  HealthDetails,
  MetricsSummaryExtended,
  PublicConfig,
  QueryResponse,
  RequestRecordExtended,
} from './api/types';

type ModeFilter = 'all' | 'rag' | 'general';

function tabFromHash(): TabId {
  const id = window.location.hash.replace('#', '');
  return (TABS.find((t) => t.id === id)?.id ?? 'ask') as TabId;
}

export function App() {
  const [tab, setTab] = useState<TabId>(tabFromHash);
  const [health, setHealth] = useState<HealthDetails | null>(null);
  const [config, setConfig] = useState<PublicConfig | null>(null);
  const [metrics, setMetrics] = useState<MetricsSummaryExtended | null>(null);
  const [requests, setRequests] = useState<RequestRecordExtended[]>([]);
  const [modeFilter, setModeFilter] = useState<ModeFilter>('all');
  const [isLoading, setIsLoading] = useState(false);
  const [isQueryRunning, setIsQueryRunning] = useState(false);
  const [collections, setCollections] = useState<CollectionInfo[]>([]);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [docsLoading, setDocsLoading] = useState(false);
  const [docsError, setDocsError] = useState<string | null>(null);
  const [collection, setCollection] = useState('default');

  useEffect(() => {
    const onHash = () => setTab(tabFromHash());
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  const changeTab = (next: TabId) => {
    window.location.hash = next;
    setTab(next);
  };

  const loadData = useCallback(async () => {
    setIsLoading(true);
    const mode = modeFilter === 'all' ? undefined : modeFilter;
    try {
      const [h, m, r] = await Promise.all([
        fetchHealth().catch(() => null),
        fetchMetricsSummary(mode).catch(() => null),
        fetchRecentRequests(50, mode).catch(() => []),
      ]);
      setHealth(h);
      setMetrics(m);
      setRequests(r);
    } finally {
      setIsLoading(false);
    }
  }, [modeFilter]);

  const loadDocuments = useCallback(async () => {
    setDocsLoading(true);
    try {
      const [cols, docs] = await Promise.all([fetchCollections(), fetchDocuments()]);
      setCollections(cols);
      setDocuments(docs);
      setDocsError(null);
    } catch (err) {
      setDocsError(err instanceof Error ? err.message : 'Failed to load documents.');
    } finally {
      setDocsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
    // Poll metrics every 15 seconds for real-time observability
    const interval = setInterval(loadData, 15000);
    return () => clearInterval(interval);
  }, [loadData]);

  useEffect(() => {
    loadDocuments();
    fetchConfig().then(setConfig).catch(() => setConfig(null));
  }, [loadDocuments]);

  // Poll while any document is still being processed.
  const anyProcessing = documents.some((d) => d.status === 'pending' || d.status === 'processing');
  useEffect(() => {
    if (!anyProcessing) return;
    const interval = setInterval(loadDocuments, 1500);
    return () => clearInterval(interval);
  }, [anyProcessing, loadDocuments]);

  const collectionNames = useMemo(() => {
    const names = new Set(collections.map((c) => c.name));
    names.add('default');
    names.add(collection);
    return Array.from(names).sort();
  }, [collections, collection]);

  const handleRunQuery = async (
    query: string,
    threshold: number,
    forceRefresh: boolean
  ): Promise<QueryResponse> => {
    setIsQueryRunning(true);
    try {
      const res = await submitQuery({
        query,
        similarity_threshold: threshold,
        force_refresh: forceRefresh,
      });
      // Refresh dashboard immediately
      await loadData();
      return res;
    } finally {
      setIsQueryRunning(false);
    }
  };

  const handleReset = async () => {
    if (window.confirm('Reset all telemetry and flush the semantic cache? Documents are kept.')) {
      try {
        await resetMetrics();
        await loadData();
      } catch (err) {
        alert(err instanceof Error ? err.message : 'Failed to reset metrics.');
      }
    }
  };

  const hasData = !!metrics && metrics.total_requests > 0;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      <Header
        health={health}
        isLoading={isLoading}
        onRefresh={() => {
          loadData();
          loadDocuments();
        }}
        onReset={handleReset}
        activeTab={tab}
        onTabChange={changeTab}
      />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
        {tab === 'ask' && (
          <AskPanel
            collection={collection}
            collections={collectionNames}
            onCollectionChange={setCollection}
            documents={documents}
            config={config}
            onAsked={loadData}
            onGoToDocuments={() => changeTab('documents')}
          />
        )}

        {tab === 'documents' && (
          <DocumentManager
            collection={collection}
            collections={collectionNames}
            onCollectionChange={setCollection}
            documents={documents}
            isLoading={docsLoading}
            loadError={docsError}
            config={config}
            onChanged={loadDocuments}
          />
        )}

        {tab === 'general' && <QueryRunner onRunQuery={handleRunQuery} isLoading={isQueryRunning} />}

        {tab === 'observability' && (
          <>
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 className="text-sm font-semibold text-slate-200">Observability</h2>
                <p className="text-xs text-slate-400">
                  Every figure comes from recorded requests. Costs are estimates from configured per-token prices.
                </p>
              </div>
              <div className="flex rounded-lg border border-slate-800 bg-slate-900/60 p-0.5 text-xs" role="group" aria-label="Request mode">
                {(['all', 'rag', 'general'] as ModeFilter[]).map((m) => (
                  <button
                    key={m}
                    type="button"
                    onClick={() => setModeFilter(m)}
                    aria-pressed={modeFilter === m}
                    className={`rounded-md px-3 py-1 ${modeFilter === m ? 'bg-slate-700 text-slate-100' : 'text-slate-400 hover:text-slate-200'}`}
                  >
                    {m === 'all' ? 'All' : m === 'rag' ? 'RAG' : 'General'}
                  </button>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
              <MetricCard
                title="Total Requests"
                value={metrics?.total_requests ?? 0}
                subValue={`${metrics?.cache_hits ?? 0} hits • ${metrics?.cache_misses ?? 0} misses`}
                icon={<Database className="w-5 h-5 text-sky-400" />}
              />

              <MetricCard
                title="Cache Hit Rate"
                value={hasData ? `${(metrics!.cache_hit_rate * 100).toFixed(1)}%` : 'n/a'}
                subValue={
                  metrics && metrics.cache_hits > 0
                    ? `${metrics.cache_hits} requests skipped the LLM`
                    : 'No cache hits yet'
                }
                icon={<Zap className="w-5 h-5 text-emerald-400" />}
              />

              <MetricCard
                title="Est. LLM Spend"
                value={metrics ? `$${metrics.total_estimated_spend.toFixed(5)}` : 'n/a'}
                subValue={
                  metrics
                    ? `${metrics.total_input_tokens} in / ${metrics.total_output_tokens} out tokens`
                    : 'Based on configured token pricing'
                }
                badge="estimate"
                badgeType="warning"
                icon={<DollarSign className="w-5 h-5 text-amber-400" />}
              />

              <MetricCard
                title="Est. Cost Avoided"
                value={metrics ? `$${metrics.total_estimated_cost_avoided.toFixed(5)}` : 'n/a'}
                subValue="Original cost of answers reused from cache"
                badge="estimate"
                badgeType="warning"
                icon={<PiggyBank className="w-5 h-5 text-emerald-400" />}
              />

              <MetricCard
                title="Latency (p50 / p95)"
                value={hasData ? `${metrics!.p50_latency_ms} / ${metrics!.p95_latency_ms} ms` : 'n/a'}
                subValue={hasData ? `Avg: ${metrics!.avg_latency_ms} ms · n=${metrics!.total_requests}` : 'No latency metrics yet'}
                icon={<Timer className="w-5 h-5 text-purple-400" />}
              />
              <MetricCard
                title="Error Rate"
                value={hasData ? `${(metrics!.error_rate * 100).toFixed(2)}%` : 'n/a'}
                subValue={`${metrics?.error_count ?? 0} failed requests`}
                icon={<CircleAlert className="w-5 h-5 text-rose-400" />}
              />
            </div>

            <PipelineInsights metrics={metrics} />

            {/* Telemetry Visualizations */}
            <ChartsSection metrics={metrics} />

            {/* Audit Log / Request History */}
            <RequestHistoryTable requests={requests} isLoading={isLoading} />
          </>
        )}
      </main>

      <footer className="border-t border-slate-900 bg-slate-950/80 py-4 text-center text-xs text-slate-400 font-mono">
        Cost-Aware RAG Engine • Grounded Retrieval, Semantic Caching & Model Routing
        {config && (
          <span className="block mt-1 text-[11px] text-slate-500">
            Embeddings: {config.embedding_provider} · Routing policy: {config.routing_policy}
          </span>
        )}
      </footer>
    </div>
  );
}
export default App;
