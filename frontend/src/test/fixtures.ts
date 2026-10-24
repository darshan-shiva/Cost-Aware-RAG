import { DocumentInfo, RAGQueryResponse } from '../api/types';

export function makeDocument(overrides: Partial<DocumentInfo> = {}): DocumentInfo {
  return {
    document_id: 'doc_1',
    collection: 'default',
    filename: 'handbook.pdf',
    file_type: 'pdf',
    size_bytes: 2048,
    content_sha256: 'abcdef0123456789',
    version: 1,
    status: 'ready',
    error_message: null,
    page_count: 4,
    chunk_count: 4,
    char_count: 1200,
    embedding_model: 'sentence-transformers/all-MiniLM-L6-v2',
    needs_reindex: false,
    created_at: '2026-10-24T10:00:00+00:00',
    updated_at: '2026-10-24T10:00:00+00:00',
    ...overrides,
  };
}

export function makeAnswer(overrides: Partial<RAGQueryResponse> = {}): RAGQueryResponse {
  return {
    request_id: 'req-1',
    mode: 'rag',
    query: 'How many vacation days?',
    collection: 'default',
    answer: 'Full-time employees receive 26 days of paid vacation [S1].',
    outcome: 'answered',
    citations: [
      {
        label: 'S1',
        document_id: 'doc_1',
        filename: 'handbook.pdf',
        page_number: 2,
        chunk_id: 'doc_1:v1:c1',
        chunk_index: 1,
        document_version: 1,
        relevance_score: 0.61,
        excerpt: 'Full-time employees receive 26 days of paid vacation per calendar year.',
      },
    ],
    retrieval: {
      top_k: 5,
      min_relevance: 0.3,
      retrieved_count: 2,
      relevant_count: 1,
      top_score: 0.61,
      sources: [
        { label: 'S1', document_id: 'doc_1', filename: 'handbook.pdf', page_number: 2, chunk_id: 'doc_1:v1:c1', relevance_score: 0.61, in_prompt: true, cited: true },
        { label: null, document_id: 'doc_1', filename: 'handbook.pdf', page_number: 1, chunk_id: 'doc_1:v1:c0', relevance_score: 0.12, in_prompt: false, cited: false },
      ],
      suspicious_passages: 0,
      performed: true,
    },
    cache: { hit: false, match_type: null, similarity: null, lookup_ms: 0.4, rejection_reason: null, bypassed: false },
    routing: {
      selected_model: 'openai/gpt-oss-20b',
      tier: 'lightweight',
      reason: 'Assessed as LOW complexity.',
      policy: 'complexity',
      factors: [],
      fallback_used: false,
      retry_count: 0,
    },
    usage: { input_tokens: 320, output_tokens: 24, token_source: 'provider', estimated_cost: 0.000031, estimated_cost_avoided: 0 },
    latency: { total_ms: 812.4, embedding_ms: 9.1, cache_lookup_ms: 0.4, retrieval_ms: 0.8, generation_ms: 798.2 },
    timestamp: '2026-10-24T10:01:00+00:00',
    ...overrides,
  };
}

type Handler = (url: string, init?: RequestInit) => unknown;

/** Install a fetch mock that routes by "METHOD path" prefix. */
export const TEST_SESSION_TOKEN = 'T'.repeat(43);

export function mockApi(customRoutes: Record<string, Handler | unknown>) {
  const routes: Record<string, Handler | unknown> = {
    'POST /api/session': {
      session_token: TEST_SESSION_TOKEN, expires_in_seconds: 1800, idle_ttl_seconds: 1800,
      max_documents: 10, max_upload_mb: 25,
    },
    ...customRoutes,
  };
  const calls: { method: string; url: string; init?: RequestInit }[] = [];
  const fetchMock = async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const method = (init?.method ?? 'GET').toUpperCase();
    calls.push({ method, url, init });
    const key = Object.keys(routes)
      .filter((k) => {
        const [m, p] = k.split(' ');
        return m === method && url.startsWith(p);
      })
      .sort((a, b) => b.length - a.length)[0];
    if (!key) return new Response(JSON.stringify({ detail: `unmocked ${method} ${url}` }), { status: 404 });
    const route = routes[key];
    const value = await (typeof route === 'function' ? (route as Handler)(url, init) : route);
    if (value instanceof Response) return value;
    return new Response(JSON.stringify(value), { status: 200, headers: { 'Content-Type': 'application/json' } });
  };
  globalThis.fetch = fetchMock as typeof fetch;
  return calls;
}
