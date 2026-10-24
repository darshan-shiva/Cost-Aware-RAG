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
  SessionInfo,
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

// ------------------------------------------------------------------ session
//
// Each page load gets its own private workspace on the server. The token
// lives only in this module's memory: it is never written to localStorage,
// sessionStorage or cookies, so a refresh or a new tab starts empty.

export const SESSION_HEADER = 'X-Session-Token';

let sessionToken: string | null = null;
let sessionStartedAt = 0;
let sessionPromise: Promise<string> | null = null;
const resetListeners = new Set<(reason: 'expired' | 'reset') => void>();

/** Called when the workspace is gone (expired, evicted, or reset by the user). */
export function onSessionReset(listener: (reason: 'expired' | 'reset') => void): () => void {
  resetListeners.add(listener);
  return () => resetListeners.delete(listener);
}

async function startSession(): Promise<string> {
  let res: Response;
  try {
    res = await fetch(`${BASE_URL}/api/session`, { method: 'POST' });
  } catch {
    throw new ApiError(0, 'Could not reach the API. Check that the backend is running.');
  }
  if (!res.ok) throw await readError(res, 'Could not start a session');
  const body = (await res.json()) as SessionInfo;
  if (!body.session_token) throw new ApiError(500, 'The server did not return a session token.');
  sessionToken = body.session_token;
  sessionStartedAt = Date.now();
  return sessionToken;
}

function getSessionToken(): Promise<string> {
  if (sessionToken) return Promise.resolve(sessionToken);
  if (!sessionPromise) {
    sessionPromise = startSession().finally(() => {
      sessionPromise = null;
    });
  }
  return sessionPromise;
}

function forgetSession(token: string, reason: 'expired' | 'reset') {
  // Several in-flight requests can fail with the same dead token; reset once.
  if (sessionToken !== token) return;
  sessionToken = null;
  // A session rejected moments after it was created points to a server
  // problem, not an expiry; resetting again would only loop.
  if (reason === 'expired' && Date.now() - sessionStartedAt < 5000) return;
  resetListeners.forEach((l) => l(reason));
}

/** Delete this workspace on the server; the next request starts a new one. */
export async function resetSession(): Promise<void> {
  const token = sessionToken;
  if (!token) return;
  try {
    await fetch(`${BASE_URL}/api/session`, { method: 'DELETE', headers: { [SESSION_HEADER]: token } });
  } catch {
    // The server-side session expires on its own if this request is lost.
  }
  forgetSession(token, 'reset');
}

/** Best-effort cleanup when the tab is closed or reloaded. Beacons cannot
 * set headers, so the token travels as the plain-text body. */
export function endSessionOnPageHide(): () => void {
  const handler = () => {
    if (sessionToken && typeof navigator !== 'undefined' && navigator.sendBeacon) {
      navigator.sendBeacon(`${BASE_URL}/api/session/close`, sessionToken);
      sessionToken = null;
    }
  };
  window.addEventListener('pagehide', handler);
  return () => window.removeEventListener('pagehide', handler);
}

/** Test helper: forget the in-memory session without calling the server. */
export function __clearSessionForTests() {
  sessionToken = null;
  sessionStartedAt = 0;
  sessionPromise = null;
  resetListeners.clear();
}

async function request<T>(
  path: string,
  init: RequestInit | undefined,
  fallback: string,
  withSession = true
): Promise<T> {
  let token: string | null = null;
  let finalInit = init;
  if (withSession) {
    token = await getSessionToken();
    finalInit = { ...init, headers: { ...(init?.headers as Record<string, string> | undefined), [SESSION_HEADER]: token } };
  }
  let res: Response;
  try {
    res = await fetch(`${BASE_URL}${path}`, finalInit);
  } catch {
    throw new ApiError(0, 'Could not reach the API. Check that the backend is running.');
  }
  if (res.status === 401 && token) {
    forgetSession(token, 'expired');
    throw new ApiError(401, 'Your session expired, so its documents and answers were deleted. Upload your files again.');
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
  return request('/health', undefined, 'Health check failed', false);
}

export function fetchConfig(): Promise<PublicConfig> {
  return request('/api/config', undefined, 'Failed to load configuration', false);
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
