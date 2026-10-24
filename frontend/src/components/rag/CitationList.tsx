import React from 'react';
import { ChevronDown, ChevronRight, FileText } from 'lucide-react';
import { Citation } from '../../api/types';

interface CitationListProps {
  citations: Citation[];
  openLabel: string | null;
  onToggle: (label: string) => void;
}

export const CitationList: React.FC<CitationListProps> = ({ citations, openLabel, onToggle }) => {
  if (citations.length === 0) return null;
  return (
    <div className="space-y-1.5" aria-label="Sources">
      <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400">Sources</div>
      {citations.map((c) => {
        const open = openLabel === c.label;
        return (
          <div key={c.label} className="rounded-lg border border-slate-800 bg-slate-950/60">
            <button
              type="button"
              onClick={() => onToggle(c.label)}
              aria-expanded={open}
              className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs hover:bg-slate-900/60"
            >
              {open ? (
                <ChevronDown className="h-3.5 w-3.5 flex-shrink-0 text-slate-400" />
              ) : (
                <ChevronRight className="h-3.5 w-3.5 flex-shrink-0 text-slate-400" />
              )}
              <span className="rounded border border-sky-800/70 bg-sky-950/60 px-1 font-mono text-[10px] font-semibold text-sky-300">
                {c.label}
              </span>
              <FileText className="h-3.5 w-3.5 flex-shrink-0 text-slate-400" />
              <span className="min-w-0 truncate font-medium text-slate-200">{c.filename}</span>
              <span className="whitespace-nowrap font-mono text-slate-400">p. {c.page_number}</span>
              <span className="ml-auto whitespace-nowrap font-mono text-[11px] text-slate-500">
                relevance {c.relevance_score.toFixed(3)}
              </span>
            </button>
            {open && (
              <div className="border-t border-slate-800 px-3 py-2.5">
                <blockquote className="whitespace-pre-wrap border-l-2 border-sky-800 pl-3 text-xs leading-relaxed text-slate-300">
                  {c.excerpt}
                </blockquote>
                <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 font-mono text-[10px] text-slate-500">
                  <span>chunk #{c.chunk_index}</span>
                  <span>document v{c.document_version}</span>
                  <span className="break-all">{c.chunk_id}</span>
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};
