import {
  CollectionInfo,
  DocumentInfo,
  HealthDetails,
  MetricsSummaryExtended,
  PublicConfig,
  QueryRequest,
  QueryResponse,
  RAGQueryRequest,
  RAGQueryResponse,
  RequestRecordExtended,
  UploadResult,
} from './types';

const BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/+$/, '');

/** Error carrying the HTTP status and the API's user-facing detail message. */
export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function readError(res: Response, fallback: string): Promise<ApiError> {
  let message = `${fallback} (${res.status})`;
  try {
    const body = await res.json();
    if (typeof body?.detail === 'string') {
      message = body.detail;
    } else if (Array.isArray(body?.detail) && body.detail[0]?.msg) {
      message = body.detail.map((d: { msg: string }) => d.msg).join('; ');
    } else if (typeof body?.error === 'string') {
      message = body.error;
    }
  } catch {
    // Non-JSON body: keep the generic message.
  }
  if (res.status === 429) message = 'Rate limit reached. Wait a minute and try again.';
  return new ApiError(res.status, message);
}

async function request<T>(path: string, init: RequestInit | undefined, fallback: string): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE_URL}${path}`, init);
  } catch {
    throw new ApiError(0, 'Could not reach the API. Check that the backend is running.');
  }
  if (!res.ok) throw await readError(res, fallback);
  return res.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
});

export function fetchHealth(): Promise<HealthDetails> {
  return request('/health', undefined, 'Health check failed');
}

export function fetchConfig(): Promise<PublicConfig> {
  return request('/api/config', undefined, 'Failed to load configuration');
}

export function fetchMetricsSummary(mode?: string): Promise<MetricsSummaryExtended> {
  const q = mode ? `?mode=${encodeURIComponent(mode)}` : '';
  return request(`/api/metrics/summary${q}`, undefined, 'Failed to fetch metrics summary');
}

export function fetchRecentRequests(limit: number = 50, mode?: string): Promise<RequestRecordExtended[]> {
  const q = mode ? `&mode=${encodeURIComponent(mode)}` : '';
  return request(`/api/metrics/requests?limit=${limit}${q}`, undefined, 'Failed to fetch recent requests');
}

export function submitQuery(payload: QueryRequest): Promise<QueryResponse> {
  return request('/api/query', json(payload), 'Query failed');
}

export function submitRagQuery(payload: RAGQueryRequest): Promise<RAGQueryResponse> {
  return request('/api/rag/query', json(payload), 'Question failed');
}

export function resetMetrics(): Promise<{ message: string }> {
  return request('/api/metrics/reset', { method: 'POST' }, 'Failed to reset metrics');
}

export function fetchCollections(): Promise<CollectionInfo[]> {
  return request('/api/collections', undefined, 'Failed to load collections');
}

export function fetchDocuments(collection?: string): Promise<DocumentInfo[]> {
  const q = collection ? `?collection=${encodeURIComponent(collection)}` : '';
  return request(`/api/documents${q}`, undefined, 'Failed to load documents');
}

export function uploadDocument(file: File, collection: string): Promise<UploadResult> {
  const form = new FormData();
  form.append('file', file);
  form.append('collection', collection);
  return request('/api/documents', { method: 'POST', body: form }, 'Upload failed');
}

export function deleteDocument(documentId: string): Promise<{ message: string }> {
  return request(`/api/documents/${encodeURIComponent(documentId)}`, { method: 'DELETE' }, 'Delete failed');
}

export function reindexDocument(documentId: string): Promise<DocumentInfo> {
  return request(
    `/api/documents/${encodeURIComponent(documentId)}/reindex`,
    { method: 'POST' },
    'Re-index failed'
  );
}
