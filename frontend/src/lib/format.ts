/** Display helpers. Missing measurements render as "n/a", never as zero. */

export function fmtMs(value: number | null | undefined): string {
  if (value === null || value === undefined) return 'n/a';
  if (value >= 1000) return `${(value / 1000).toFixed(2)} s`;
  return `${value.toFixed(value < 10 ? 2 : 1)} ms`;
}

export function fmtUsd(value: number | null | undefined, digits = 6): string {
  if (value === null || value === undefined) return 'n/a';
  return `$${value.toFixed(digits)}`;
}

export function fmtBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function fmtPct(value: number | null | undefined): string {
  if (value === null || value === undefined) return 'n/a';
  return `${(value * 100).toFixed(1)}%`;
}

export function fmtTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString([], { hour12: false, month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

export const OUTCOME_LABELS: Record<string, string> = {
  answered: 'Grounded answer',
  insufficient_evidence: 'Insufficient evidence',
  uncited: 'Unverified (no citations)',
  no_documents: 'No documents',
  error: 'Error',
};
