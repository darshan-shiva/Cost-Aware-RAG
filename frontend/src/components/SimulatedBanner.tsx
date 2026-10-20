import React from 'react';
import { Info, ShieldAlert } from 'lucide-react';

export const SimulatedBanner: React.FC = () => {
  return (
    <div className="bg-slate-900/40 border border-slate-800 rounded-lg p-3.5 flex items-start space-x-3 text-xs">
      <Info className="w-4 h-4 text-sky-400 mt-0.5 flex-shrink-0" />
      <div className="space-y-0.5 text-slate-300">
        <span className="font-semibold text-slate-200">Phase 1 Foundation Architecture Notice:</span>
        <p className="text-slate-400 leading-relaxed">
          Operating in local simulation mode. Embeddings use deterministic feature-hashing vectorization (64-dim unit hypersphere) for real cosine similarity verification without external GPU/API costs. LLM spend reflects configurable token pricing formulas. Provider-reported billing APIs remain an extension point for subsequent phases.
        </p>
      </div>
    </div>
  );
};
