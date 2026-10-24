import React from 'react';
import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { App } from '../App';
import { fetchDocuments, SESSION_HEADER } from '../api/client';
import { makeDocument, mockApi, TEST_SESSION_TOKEN } from './fixtures';

const quietRoutes = {
  'GET /health': { status: 'healthy', version: '0.3.0', simulated_mode: false, timestamp: '' },
  'GET /api/config': () => new Response('{}', { status: 500 }),
  'GET /api/metrics/summary': () => new Response('{}', { status: 500 }),
  'GET /api/metrics/requests': [],
};

function headerOf(init?: RequestInit): string | undefined {
  return (init?.headers as Record<string, string> | undefined)?.[SESSION_HEADER];
}

describe('per-visitor sessions', () => {
  it('starts one session and sends its token on every workspace request', async () => {
    const calls = mockApi({ 'GET /api/documents': [] });
    await Promise.all([fetchDocuments(), fetchDocuments(), fetchDocuments()]);
    expect(calls.filter((c) => c.method === 'POST' && c.url.endsWith('/api/session'))).toHaveLength(1);
    const docCalls = calls.filter((c) => c.url.includes('/api/documents'));
    expect(docCalls).toHaveLength(3);
    docCalls.forEach((c) => expect(headerOf(c.init)).toBe(TEST_SESSION_TOKEN));
  });

  it('never writes the token or any data to browser storage', async () => {
    const setLocal = vi.spyOn(Storage.prototype, 'setItem');
    mockApi({ ...quietRoutes, 'GET /api/collections': [], 'GET /api/documents': [makeDocument()] });
    render(<App />);
    await screen.findByRole('heading', { name: 'Ask your documents' });
    expect(setLocal).not.toHaveBeenCalled();
    expect(window.localStorage.length).toBe(0);
    expect(window.sessionStorage.length).toBe(0);
    expect(document.cookie).not.toContain(TEST_SESSION_TOKEN);
  });

  it('a new page load starts with an empty workspace', async () => {
    // Simulates a refresh: no token survives, so the app asks for a new session.
    const calls = mockApi({ ...quietRoutes, 'GET /api/collections': [], 'GET /api/documents': [] });
    render(<App />);
    window.location.hash = 'documents';
    await waitFor(() => expect(calls.some((c) => c.url.endsWith('/api/session'))).toBe(true));
  });

  it('reset deletes the server session and clears the UI', async () => {
    let docs = [makeDocument()];
    vi.spyOn(window, 'confirm').mockReturnValue(true);
    const calls = mockApi({
      ...quietRoutes,
      'GET /api/collections': () => (docs.length ? [{ name: 'default', document_count: 1, ready_count: 1, chunk_count: 4 }] : []),
      'GET /api/documents': () => docs,
      'DELETE /api/session': () => {
        docs = [];
        return { message: 'ended' };
      },
    });
    window.location.hash = 'documents';
    render(<App />);
    await screen.findByText('handbook.pdf');
    fireEvent.click(screen.getByRole('button', { name: 'Start a fresh session' }));
    await screen.findByText(/Started a fresh session/);
    await waitFor(() => expect(screen.queryByText('handbook.pdf')).not.toBeInTheDocument());
    const del = calls.find((c) => c.method === 'DELETE' && c.url.endsWith('/api/session'));
    expect(headerOf(del?.init)).toBe(TEST_SESSION_TOKEN);
    // A new session was requested after the reset.
    expect(calls.filter((c) => c.method === 'POST' && c.url.endsWith('/api/session')).length).toBeGreaterThanOrEqual(2);
  });

  it('an expired session resets the workspace and tells the visitor', async () => {
    let expired = false;
    mockApi({
      ...quietRoutes,
      'GET /api/collections': [],
      'GET /api/documents': () => (expired ? new Response('{"detail":"expired"}', { status: 401 }) : [makeDocument()]),
    });
    window.location.hash = 'documents';
    render(<App />);
    await screen.findByText('handbook.pdf');
    expired = true;
    vi.spyOn(Date, 'now').mockReturnValue(Date.now() + 60_000);
    fireEvent.click(screen.getByTitle('Refresh Metrics'));
    await screen.findByText(/Your session expired/);
  });

  it('closes the session with a beacon when the page is hidden', async () => {
    const beacon = vi.fn().mockReturnValue(true);
    Object.defineProperty(navigator, 'sendBeacon', { value: beacon, configurable: true });
    mockApi({ ...quietRoutes, 'GET /api/collections': [], 'GET /api/documents': [] });
    render(<App />);
    await waitFor(() => expect(screen.getByText(/No documents|Upload/i)).toBeTruthy());
    window.dispatchEvent(new Event('pagehide'));
    expect(beacon).toHaveBeenCalledWith(expect.stringMatching(/\/api\/session\/close$/), TEST_SESSION_TOKEN);
  });
});
