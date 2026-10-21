import React from 'react';
import { Activity, RefreshCw, Trash2, Cpu, ShieldCheck, AlertCircle } from 'lucide-react';
import { HealthResponse } from '../api/types';

interface HeaderProps {
  health: HealthResponse | null;
  isLoading: boolean;
  onRefresh: () => void;
  onReset: () => void;
}

export const Header: React.FC<HeaderProps> = ({ health, isLoading, onRefresh, onReset }) => {
  return (
    <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-emerald-500/10 border border-emerald-500/20 rounded-lg text-emerald-400">
            <Cpu className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="font-semibold text-slate-100 text-lg tracking-tight">Cost-Aware RAG</h1>
              <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-emerald-950/80 text-emerald-400 border border-emerald-800/80">
                Phase 5 Observability
              </span>
            </div>

            <p className="text-xs text-slate-400">Semantic Caching & Intelligent Model Routing Observability</p>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          {/* Health indicator */}
          <div className="flex items-center space-x-2 text-xs font-mono px-3 py-1.5 rounded-md bg-slate-950 border border-slate-800">
            {health ? (
              <>
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                <span className="text-slate-300">API: {health.status} (v{health.version})</span>
              </>
            ) : (
              <>
                <AlertCircle className="w-3.5 h-3.5 text-amber-500" />
                <span className="text-slate-400">Connecting to API...</span>
              </>
            )}
          </div>

          {/* Action buttons */}
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
              title="Reset Cache & Metrics"
              className="p-2 text-rose-400 hover:text-rose-300 bg-rose-950/20 hover:bg-rose-900/30 border border-rose-900/40 rounded-lg transition-colors"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};
