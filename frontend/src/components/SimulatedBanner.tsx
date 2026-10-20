import React from 'react';
import { Info, ShieldAlert } from 'lucide-react';

export const SimulatedBanner: React.FC = () => {
  return (
    <div className="bg-slate-900/40 border border-slate-800 rounded-lg p-3.5 flex items-start space-x-3 text-xs">
      <Info className="w-4 h-4 text-sky-400 mt-0.5 flex-shrink-0" />
      <div className="space-y-0.5 text-slate-300">
        <span className="font-semibold text-slate-200">Phase 2 Real Semantic Caching Active:</span>
        <p className="text-slate-400 leading-relaxed">
          Powered by local dense semantic embeddings (<strong className="text-slate-300">sentence-transformers/all-MiniLM-L6-v2</strong>, 384-dimensional unit vectors) with SQLite persistence across server restarts. Cosine similarity evaluates true semantic meaning. Inference spend reflects configurable model token pricing.
        </p>

      </div>
    </div>
  );
};
