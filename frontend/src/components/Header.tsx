import React from 'react';
import { RefreshCw, Trash2, Cpu, AlertCircle } from 'lucide-react';
import { HealthDetails } from '../api/types';

export type TabId = 'ask' | 'documents' | 'observability' | 'general';

export const TABS: { id: TabId; label: string }[] = [
  { id: 'ask', label: 'Ask documents' },
  { id: 'documents', label: 'Documents' },
  { id: 'observability', label: 'Observability' },
  { id: 'general', label: 'General mode' },
];

interface HeaderProps {
  health: HealthDetails | null;
  isLoading: boolean;
  onRefresh: () => void;
  onReset: () => void;
  activeTab: TabId;
  onTabChange: (tab: TabId) => void;
}

export const Header: React.FC<HeaderProps> = ({ health, isLoading, onRefresh, onReset, activeTab, onTabChange }) => {
  return (
    <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 min-h-16 py-2 flex flex-wrap gap-3 items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-emerald-500/10 border border-emerald-500/20 rounded-lg text-emerald-400">
            <Cpu className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="font-semibold text-slate-100 text-lg tracking-tight">Cost-Aware RAG</h1>
            </div>

            <p className="text-xs text-slate-400">Document-grounded answers with semantic caching & model routing</p>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          {/* Health indicator */}
          <div className="flex items-center space-x-2 text-xs font-mono px-3 py-1.5 rounded-md bg-slate-950 border border-slate-800">
            {health ? (
              <>
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                <span className="text-slate-300">API: {health.status} (v{health.version})</span>
                {health.llm_configured === false && (
                  <span className="text-amber-400" title="RAG_GROQ_API_KEY is not set on the server">· no LLM key</span>
                )}
                {health.simulated_mode && (
                  <span className="text-amber-400" title="The server uses a lexical hashing embedder, not a semantic model">· hashing embeddings</span>
                )}
              </>
            ) : (
              <>
                <AlertCircle className="w-3.5 h-3.5 text-amber-500" />
                <span className="text-slate-400">Connecting to API...</span>
              </>
            )}
          </div>

          {/* Action buttons (telemetry controls) */}
          <div className="flex items-center space-x-2">
            <button
              onClick={onRefresh}
              disabled={isLoading}
              title="Refresh Metrics"
              className="p-2 text-slate-400 hover:text-slate-200 bg-slate-800/80 hover:bg-slate-700/80 border border-slate-700/60 rounded-lg transition-colors disabled:opacity-50"
            >
              <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
            </button>
            <button
              onClick={onReset}
              title="Start a fresh session (deletes your documents, answers and telemetry)"
              aria-label="Start a fresh session"
              className="p-2 text-rose-400 hover:text-rose-300 bg-rose-950/20 hover:bg-rose-900/30 border border-rose-900/40 rounded-lg transition-colors"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
      <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex gap-1 overflow-x-auto" aria-label="Sections">
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => onTabChange(t.id)}
            aria-current={activeTab === t.id ? 'page' : undefined}
            className={`whitespace-nowrap border-b-2 px-3 py-2 text-xs font-medium transition-colors ${activeTab === t.id ? 'border-emerald-500 text-emerald-300' : 'border-transparent text-slate-400 hover:text-slate-200'}`}
          >
            {t.label}
          </button>
        ))}
      </nav>
    </header>
  );
};
