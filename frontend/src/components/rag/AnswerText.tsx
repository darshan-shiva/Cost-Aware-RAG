import React from 'react';
import { Citation } from '../../api/types';

interface AnswerTextProps {
  text: string;
  citations: Citation[];
  onCite: (label: string) => void;
}

const MARKER = /\[(S\d{1,3})\]/g;

/** Renders an answer with [S1]-style markers as buttons that open the
 *  matching citation. Markers without a citation (should not happen, the
 *  backend strips them) are shown as plain text. */
export const AnswerText: React.FC<AnswerTextProps> = ({ text, citations, onCite }) => {
  const known = new Map(citations.map((c) => [c.label, c]));
  const parts: React.ReactNode[] = [];
  let last = 0;
  for (const match of text.matchAll(MARKER)) {
    const index = match.index ?? 0;
    if (index > last) parts.push(text.slice(last, index));
    const label = match[1];
    const citation = known.get(label);
    parts.push(
      citation ? (
        <button
          key={`${label}-${index}`}
          type="button"
          onClick={() => onCite(label)}
          title={`${citation.filename}, page ${citation.page_number}`}
          aria-label={`Source ${label}: ${citation.filename}, page ${citation.page_number}`}
          className="mx-0.5 inline-flex items-center rounded border border-sky-800/70 bg-sky-950/60 px-1 font-mono text-[10px] font-semibold text-sky-300 hover:bg-sky-900/60 align-text-top"
        >
          {label}
        </button>
      ) : (
        match[0]
      )
    );
    last = index + match[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return <p className="whitespace-pre-wrap text-sm leading-relaxed text-slate-200">{parts}</p>;
};
