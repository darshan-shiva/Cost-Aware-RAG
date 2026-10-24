import React from 'react';
import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { App } from '../App';
import { makeAnswer, makeDocument, mockApi } from './fixtures';

const baseRoutes = {
  'GET /health': { status: 'healthy', version: '0.2.0', simulated_mode: false, timestamp: '', llm_configured: true },
  'GET /api/config': {
    embedding_provider: 'sentence-transformers/all-MiniLM-L6-v2', embedding_simulated: false, llm_configured: true,
    routing_policy: 'complexity', models: [], pricing_is_estimate: true, cache_similarity_threshold: 0.82,
    retrieval_top_k: 5, retrieval_max_top_k: 12, retrieval_min_relevance: 0.3, rag_cache_min_evidence_overlap: 0.5,
    max_upload_mb: 10, allowed_file_types: ['.pdf', '.txt'], chunk_size_chars: 900, chunk_overlap_chars: 150,
    telemetry_stores_query_text: true,
  },
  'GET /api/metrics/summary': {
    total_requests: 0, cache_hits: 0, cache_misses: 0, cache_hit_rate: 0, total_estimated_spend: 0,
    p50_latency_ms: 0, p95_latency_ms: 0, avg_latency_ms: 0, routing_breakdown: {}, cache_breakdown: {},
    is_simulated: false, error_count: 0, error_rate: 0, total_estimated_cost_avoided: 0, total_input_tokens: 0,
    total_output_tokens: 0, mode_breakdown: {}, outcome_breakdown: {}, match_type_breakdown: {},
    error_category_breakdown: {}, model_stats: {}, avg_cache_lookup_ms: null, avg_embedding_ms: null,
    avg_retrieval_ms: null, avg_generation_ms: null, avg_latency_cache_hit_ms: null, avg_latency_cache_miss_ms: null,
    fallback_count: 0, pricing_note: 'Costs are estimates.',
  },
  'GET /api/metrics/requests': [],
};

describe('upload-to-query workflow', () => {
  it('uploads a document, shows processing status, then answers with citations', async () => {
    let docs: ReturnType<typeof makeDocument>[] = [];
    let ragBody: Record<string, unknown> | null = null;
    let releaseAnswer: () => void = () => {};
    const calls = mockApi({
      ...baseRoutes,
      'GET /api/collections': () => (docs.length ? [{ name: 'default', document_count: 1, ready_count: 1, chunk_count: 4 }] : []),
      'GET /api/documents': () => docs,
      'POST /api/documents': () => {
        docs = [makeDocument({ status: 'processing', page_count: 0, chunk_count: 0 })];
        return { document: docs[0], outcome: 'created', message: "'handbook.pdf' was accepted and is being processed." };
      },
      'POST /api/rag/query': (_url: string, init?: RequestInit) => {
        ragBody = JSON.parse(String(init?.body));
        // Hold the response so the loading state can be observed.
        return new Promise((resolve) => {
          releaseAnswer = () => resolve(makeAnswer());
        });
      },
    });

    window.location.hash = 'documents';
    render(<App />);
    expect(await screen.findByText('No documents in this collection yet.')).toBeInTheDocument();

    const file = new File(['%PDF-1.4 test'], 'handbook.pdf', { type: 'application/pdf' });
    fireEvent.change(screen.getByTestId('file-input'), { target: { files: [file] } });

    expect(await screen.findByText(/was accepted and is being processed/)).toBeInTheDocument();
    expect(await screen.findByText('processing')).toBeInTheDocument();
    const upload = calls.find((c) => c.method === 'POST' && c.url.endsWith('/api/documents'));
    const form = upload?.init?.body as FormData;
    expect(form.get('collection')).toBe('default');
    expect((form.get('file') as File).name).toBe('handbook.pdf');

    // The backend finishes processing; the page polls and picks it up.
    docs = [makeDocument()];
    expect(await screen.findByText('ready', {}, { timeout: 4000 })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Ask documents' }));
    const box = await screen.findByLabelText('Question');
    fireEvent.change(box, { target: { value: 'How many vacation days?' } });
    fireEvent.click(screen.getByRole('button', { name: 'Ask' }));

    expect(await screen.findByText(/Retrieving passages/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Answering…' })).toBeDisabled();
    releaseAnswer();
    const card = await screen.findByTestId('answer-card');
    expect(within(card).getByText('Grounded answer')).toBeInTheDocument();
    expect(within(card).getByText('handbook.pdf')).toBeInTheDocument();
    expect(ragBody).toMatchObject({ query: 'How many vacation days?', collection: 'default', force_refresh: false });
  });

  it('rejects unsupported files before uploading', async () => {
    const calls = mockApi({ ...baseRoutes, 'GET /api/collections': [], 'GET /api/documents': [] });
    window.location.hash = 'documents';
    render(<App />);
    const file = new File(['x'], 'notes.docx');
    fireEvent.change(await screen.findByTestId('file-input'), { target: { files: [file] } });
    expect(await screen.findByRole('alert')).toHaveTextContent('only .pdf and .txt files are supported');
    expect(calls.some((c) => c.method === 'POST' && c.url.includes('/api/documents'))).toBe(false);
  });

  it('shows the API error message when a question fails', async () => {
    mockApi({
      ...baseRoutes,
      'GET /api/collections': [{ name: 'default', document_count: 1, ready_count: 1, chunk_count: 4 }],
      'GET /api/documents': [makeDocument()],
      'POST /api/rag/query': () =>
        new Response(JSON.stringify({ detail: 'No language model provider is configured on the server (RAG_GROQ_API_KEY).' }), { status: 503 }),
    });
    render(<App />);
    fireEvent.change(await screen.findByLabelText('Question'), { target: { value: 'Anything?' } });
    fireEvent.click(screen.getByRole('button', { name: 'Ask' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('No language model provider is configured');
  });

  it('shows an empty state on the observability tab without fabricated data', async () => {
    mockApi({ ...baseRoutes, 'GET /api/collections': [], 'GET /api/documents': [] });
    window.location.hash = 'observability';
    render(<App />);
    expect(await screen.findByText(/No requests logged yet/)).toBeInTheDocument();
    expect(screen.getAllByText('n/a').length).toBeGreaterThan(0);
    expect(screen.getByText('No cache telemetry data yet')).toBeInTheDocument();
  });

  it('confirms before deleting a document', async () => {
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(false);
    const calls = mockApi({
      ...baseRoutes,
      'GET /api/collections': [{ name: 'default', document_count: 1, ready_count: 1, chunk_count: 4 }],
      'GET /api/documents': [makeDocument()],
    });
    window.location.hash = 'documents';
    render(<App />);
    fireEvent.click(await screen.findByRole('button', { name: 'Delete handbook.pdf' }));
    expect(confirm).toHaveBeenCalled();
    await waitFor(() => expect(calls.some((c) => c.method === 'DELETE')).toBe(false));
  });
});
