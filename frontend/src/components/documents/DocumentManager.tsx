import React, { useRef, useState } from 'react';
import { AlertTriangle, CheckCircle2, FileText, FolderPlus, Loader2, RotateCw, Trash2, UploadCloud } from 'lucide-react';
import { deleteDocument, reindexDocument, uploadDocument } from '../../api/client';
import { DocumentInfo, DocumentStatus, PublicConfig } from '../../api/types';
import { fmtBytes, fmtTime } from '../../lib/format';

interface DocumentManagerProps {
  collection: string;
  collections: string[];
  onCollectionChange: (name: string) => void;
  documents: DocumentInfo[];
  isLoading: boolean;
  loadError: string | null;
  config: PublicConfig | null;
  onChanged: () => void;
}

const STATUS_STYLES: Record<DocumentStatus, string> = {
  pending: 'border-slate-700 bg-slate-900 text-slate-300',
  processing: 'border-sky-800/70 bg-sky-950/50 text-sky-300',
  ready: 'border-emerald-800/70 bg-emerald-950/50 text-emerald-300',
  failed: 'border-rose-800/70 bg-rose-950/50 text-rose-300',
};

const COLLECTION_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/;

export const StatusBadge: React.FC<{ status: DocumentStatus }> = ({ status }) => (
  <span
    className={`inline-flex items-center gap-1 rounded border px-2 py-0.5 font-mono text-[10px] font-semibold uppercase ${STATUS_STYLES[status]}`}
  >
    {(status === 'pending' || status === 'processing') && <Loader2 className="h-3 w-3 animate-spin" />}
    {status === 'ready' && <CheckCircle2 className="h-3 w-3" />}
    {status === 'failed' && <AlertTriangle className="h-3 w-3" />}
    {status}
  </span>
);

export const DocumentManager: React.FC<DocumentManagerProps> = ({
  collection,
  collections,
  onCollectionChange,
  documents,
  isLoading,
  loadError,
  config,
  onChanged,
}) => {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [notice, setNotice] = useState<{ kind: 'ok' | 'error'; text: string } | null>(null);
  const [newCollection, setNewCollection] = useState('');
  const [busyId, setBusyId] = useState<string | null>(null);

  const maxMb = config?.max_upload_mb ?? 10;
  const docs = documents.filter((d) => d.collection === collection);

  const handleFiles = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    setNotice(null);
    setUploading(true);
    const messages: string[] = [];
    let failed = false;
    for (const file of Array.from(files)) {
      const ext = file.name.toLowerCase().slice(file.name.lastIndexOf('.'));
      if (!['.pdf', '.txt'].includes(ext)) {
        messages.push(`${file.name}: only .pdf and .txt files are supported.`);
        failed = true;
        continue;
      }
      if (file.size > maxMb * 1024 * 1024) {
        messages.push(`${file.name}: larger than the ${maxMb} MB limit.`);
        failed = true;
        continue;
      }
      try {
        const res = await uploadDocument(file, collection);
        messages.push(res.message);
      } catch (err) {
        failed = true;
        messages.push(`${file.name}: ${err instanceof Error ? err.message : 'upload failed'}`);
      }
    }
    setUploading(false);
    setNotice({ kind: failed ? 'error' : 'ok', text: messages.join(' ') });
    if (inputRef.current) inputRef.current.value = '';
    onChanged();
  };

  const act = async (doc: DocumentInfo, action: 'delete' | 'reindex') => {
    if (action === 'delete' && !window.confirm(`Delete "${doc.filename}" with its chunks, embeddings and cached answers?`)) {
      return;
    }
    setBusyId(doc.document_id);
    setNotice(null);
    try {
      if (action === 'delete') {
        const res = await deleteDocument(doc.document_id);
        setNotice({ kind: 'ok', text: res.message });
      } else {
        await reindexDocument(doc.document_id);
        setNotice({ kind: 'ok', text: `Re-indexing "${doc.filename}".` });
      }
    } catch (err) {
      setNotice({ kind: 'error', text: err instanceof Error ? err.message : `${action} failed` });
    } finally {
      setBusyId(null);
      onChanged();
    }
  };

  const createCollection = (e: React.FormEvent) => {
    e.preventDefault();
    const name = newCollection.trim();
    if (!COLLECTION_RE.test(name)) {
      setNotice({ kind: 'error', text: 'Collection names use 1-64 letters, digits, "-" or "_".' });
      return;
    }
    onCollectionChange(name);
    setNewCollection('');
    setNotice({ kind: 'ok', text: `Collection "${name}" selected. It is created when you upload its first document.` });
  };

  return (
    <div className="space-y-5">
      <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-4">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h2 className="text-sm font-semibold text-slate-200">Documents</h2>
            <p className="mt-0.5 text-xs text-slate-400">
              PDF or TXT, up to {maxMb} MB. Text is extracted page by page, split into
              {config ? ` ${config.chunk_size_chars}-character` : ''} overlapping chunks and embedded
              {config ? ` with ${config.embedding_provider}` : ''}. Files are processed on the API server and kept only in this browser session's private, in-memory workspace: they are deleted when you start a fresh session, reload the page, stay idle for {config?.session_idle_ttl_minutes ?? 30} minutes, or the server restarts.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <select
              aria-label="Collection"
              value={collection}
              onChange={(e) => onCollectionChange(e.target.value)}
              className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 font-mono text-xs text-slate-200"
            >
              {collections.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
            <form onSubmit={createCollection} className="flex items-center gap-1">
              <input
                value={newCollection}
                onChange={(e) => setNewCollection(e.target.value)}
                placeholder="new-collection"
                aria-label="New collection name"
                className="w-36 rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 font-mono text-xs text-slate-200 placeholder-slate-600"
              />
              <button
                type="submit"
                title="Use new collection"
                className="rounded-md border border-slate-700 bg-slate-800/80 p-1.5 text-slate-300 hover:bg-slate-700/80"
              >
                <FolderPlus className="h-4 w-4" />
              </button>
            </form>
          </div>
        </div>

        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            handleFiles(e.dataTransfer.files);
          }}
          className={`flex flex-col items-center justify-center rounded-lg border border-dashed px-4 py-8 text-center transition-colors ${dragging ? 'border-emerald-500 bg-emerald-950/20' : 'border-slate-700 bg-slate-950/40'}`}
        >
          {uploading ? (
            <Loader2 className="h-6 w-6 animate-spin text-emerald-400" />
          ) : (
            <UploadCloud className="h-6 w-6 text-slate-400" />
          )}
          <p className="mt-2 text-sm text-slate-300">
            {uploading ? 'Uploading…' : 'Drop files here or'}{' '}
            {!uploading && (
              <button type="button" onClick={() => inputRef.current?.click()} className="text-emerald-400 underline-offset-2 hover:underline">
                choose files
              </button>
            )}
          </p>
          <p className="mt-1 font-mono text-[11px] text-slate-500">into collection “{collection}”</p>
          <input
            ref={inputRef}
            type="file"
            accept=".pdf,.txt,application/pdf,text/plain"
            multiple
            className="hidden"
            data-testid="file-input"
            onChange={(e) => handleFiles(e.target.files)}
          />
        </div>

        {notice && (
          <div
            role={notice.kind === 'error' ? 'alert' : 'status'}
            className={`rounded-lg border p-3 text-xs ${notice.kind === 'error' ? 'border-rose-900/60 bg-rose-950/40 text-rose-300' : 'border-emerald-900/60 bg-emerald-950/30 text-emerald-300'}`}
          >
            {notice.text}
          </div>
        )}
      </section>

      <section className="overflow-hidden rounded-xl border border-slate-800 bg-slate-900/60">
        <div className="flex items-center justify-between border-b border-slate-800 p-4">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">“{collection}” documents</h3>
          <span className="font-mono text-[11px] text-slate-400">
            {docs.filter((d) => d.status === 'ready').length}/{docs.length} ready
          </span>
        </div>
        {loadError ? (
          <div role="alert" className="p-6 text-center text-xs text-rose-300">{loadError}</div>
        ) : docs.length === 0 ? (
          <div className="flex flex-col items-center gap-2 p-10 text-center text-xs text-slate-400">
            <FileText className="h-6 w-6" />
            {isLoading ? 'Loading documents…' : 'No documents in this collection yet.'}
          </div>
        ) : (
          <ul className="divide-y divide-slate-800/70">
            {docs.map((d) => (
              <li key={d.document_id} className="flex flex-col gap-2 p-4 sm:flex-row sm:items-center">
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <FileText className="h-4 w-4 flex-shrink-0 text-slate-400" />
                    <span className="truncate text-sm font-medium text-slate-200">{d.filename}</span>
                    <StatusBadge status={d.status} />
                    {d.needs_reindex && (
                      <span className="rounded border border-amber-800/70 bg-amber-950/40 px-1.5 py-0.5 font-mono text-[10px] text-amber-300" title="Indexed with a different embedding model than the server now uses.">
                        NEEDS RE-INDEX
                      </span>
                    )}
                  </div>
                  <div className="mt-1 flex flex-wrap gap-x-4 gap-y-0.5 font-mono text-[11px] text-slate-500">
                    <span>{d.file_type.toUpperCase()} · {fmtBytes(d.size_bytes)}</span>
                    <span>v{d.version}</span>
                    {d.status === 'ready' && (
                      <span>
                        {d.page_count} page{d.page_count === 1 ? '' : 's'} · {d.chunk_count} chunks
                      </span>
                    )}
                    <span>updated {fmtTime(d.updated_at)}</span>
                    <span title={d.content_sha256}>sha256 {d.content_sha256.slice(0, 10)}</span>
                  </div>
                  {d.status === 'failed' && d.error_message && (
                    <p className="mt-1 text-xs text-rose-300">{d.error_message}</p>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    disabled={busyId === d.document_id || d.status === 'processing' || d.status === 'pending'}
                    onClick={() => act(d, 'reindex')}
                    className="flex items-center gap-1 rounded-lg border border-slate-700 bg-slate-800/70 px-2.5 py-1.5 text-xs text-slate-300 hover:bg-slate-700/70 disabled:opacity-40"
                  >
                    <RotateCw className="h-3.5 w-3.5" /> Re-index
                  </button>
                  <button
                    type="button"
                    disabled={busyId === d.document_id}
                    onClick={() => act(d, 'delete')}
                    aria-label={`Delete ${d.filename}`}
                    className="flex items-center gap-1 rounded-lg border border-rose-900/50 bg-rose-950/20 px-2.5 py-1.5 text-xs text-rose-300 hover:bg-rose-900/30 disabled:opacity-40"
                  >
                    <Trash2 className="h-3.5 w-3.5" /> Delete
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
};
