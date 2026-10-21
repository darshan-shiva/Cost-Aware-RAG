import React, { useState, useEffect, useCallback } from 'react';
import { Database, Zap, DollarSign, Timer, CircleAlert } from 'lucide-react';
import { Header } from './components/Header';
import { MetricCard } from './components/MetricCard';
import { QueryRunner } from './components/QueryRunner';
import { ChartsSection } from './components/ChartsSection';
import { RequestHistoryTable } from './components/RequestHistoryTable';

import {
  fetchHealth,
  fetchMetricsSummary,
  fetchRecentRequests,
  submitQuery,
  resetMetrics,
} from './api/client';
import { HealthResponse, MetricsSummary, RequestRecord, QueryResponse } from './api/types';

export function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [metrics, setMetrics] = useState<MetricsSummary | null>(null);
  const [requests, setRequests] = useState<RequestRecord[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isQueryRunning, setIsQueryRunning] = useState(false);

  const loadData = useCallback(async () => {
    setIsLoading(true);
    try {
      const [h, m, r] = await Promise.all([
        fetchHealth().catch(() => null),
        fetchMetricsSummary().catch(() => null),
        fetchRecentRequests(50).catch(() => []),
      ]);
      setHealth(h);
      setMetrics(m);
      setRequests(r);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
    // Poll metrics every 15 seconds for real-time observability
    const interval = setInterval(loadData, 15000);
    return () => clearInterval(interval);
  }, [loadData]);

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
    if (window.confirm('Reset all metrics telemetry and flush semantic cache?')) {
      try {
        await resetMetrics();
        await loadData();
      } catch (err) {
        alert('Failed to reset metrics.');
      }
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      <Header
        health={health}
        isLoading={isLoading}
        onRefresh={loadData}
        onReset={handleReset}
      />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
          <MetricCard
            title="Total Requests"
            value={metrics?.total_requests ?? 0}
            subValue={`${metrics?.cache_hits ?? 0} hits • ${metrics?.cache_misses ?? 0} misses`}
            icon={<Database className="w-5 h-5 text-sky-400" />}
          />

          <MetricCard
            title="Cache Hit Rate"
            value={
              metrics
                ? `${(metrics.cache_hit_rate * 100).toFixed(1)}%`
                : '0.0%'
            }
            subValue={
              metrics && metrics.cache_hits > 0
                ? `${metrics.cache_hits} queries bypassed LLM`
                : 'Awaiting repeated queries'
            }
            badge={metrics && metrics.cache_hits > 0 ? 'Cost Saved' : undefined}
            badgeType="success"
            icon={<Zap className="w-5 h-5 text-emerald-400" />}
          />

          <MetricCard
            title="Est. LLM Spend"
            value={
              metrics
                ? `$${metrics.total_estimated_spend.toFixed(5)}`
                : '$0.00000'
            }
            subValue="Based on model token pricing"
            icon={<DollarSign className="w-5 h-5 text-amber-400" />}
          />

          <MetricCard
            title="Latency (p50 / p95)"
            value={
              metrics && metrics.total_requests > 0
                ? `${metrics.p50_latency_ms} / ${metrics.p95_latency_ms} ms`
                : '0.0 / 0.0 ms'
            }
            subValue={
              metrics && metrics.total_requests > 0
                ? `Avg: ${metrics.avg_latency_ms} ms`
                : 'No latency metrics yet'
            }
            icon={<Timer className="w-5 h-5 text-purple-400" />}
          />
          <MetricCard
            title="Error Rate"
            value={`${((metrics?.error_rate ?? 0) * 100).toFixed(2)}%`}
            subValue={`${metrics?.error_count ?? 0} failed requests`}
            icon={<CircleAlert className="w-5 h-5 text-rose-400" />}
          />

        </div>

        {/* Interactive Query Runner */}
        <QueryRunner onRunQuery={handleRunQuery} isLoading={isQueryRunning} />

        {/* Telemetry Visualizations */}
        <ChartsSection metrics={metrics} />

        {/* Audit Log / Request History */}
        <RequestHistoryTable requests={requests} isLoading={isLoading} />
      </main>

      <footer className="border-t border-slate-900 bg-slate-950/80 py-4 text-center text-xs text-slate-400 font-mono">
        Cost-Aware RAG Engine • Phase 5 Observability • Semantic Caching & Model Routing
      </footer>
    </div>
  );
}
export default App;
