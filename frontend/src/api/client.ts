import { HealthResponse, MetricsSummary, RequestRecord, QueryRequest, QueryResponse } from './types';

const BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/+$/, '');

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BASE_URL}/health`);
  if (!res.ok) {
    throw new Error(`Health check failed with status: ${res.status}`);
  }
  return res.json();
}

export async function fetchMetricsSummary(): Promise<MetricsSummary> {
  const res = await fetch(`${BASE_URL}/api/metrics/summary`);
  if (!res.ok) {
    throw new Error(`Failed to fetch metrics summary: ${res.status}`);
  }
  return res.json();
}

export async function fetchRecentRequests(limit: number = 50): Promise<RequestRecord[]> {
  const res = await fetch(`${BASE_URL}/api/metrics/requests?limit=${limit}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch recent requests: ${res.status}`);
  }
  return res.json();
}

export async function submitQuery(payload: QueryRequest): Promise<QueryResponse> {
  const res = await fetch(`${BASE_URL}/api/query`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const errorBody = await res.text();
    throw new Error(`Query failed (${res.status}): ${errorBody}`);
  }
  return res.json();
}

export async function resetMetrics(): Promise<{ message: string }> {
  const res = await fetch(`${BASE_URL}/api/metrics/reset`, {
    method: 'POST',
  });
  if (!res.ok) {
    throw new Error(`Failed to reset metrics: ${res.status}`);
  }
  return res.json();
}
