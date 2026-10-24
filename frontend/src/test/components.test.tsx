import React from 'react';
import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { AnswerText } from '../components/rag/AnswerText';
import { AnswerCard } from '../components/rag/AnswerCard';
import { makeAnswer } from './fixtures';

describe('AnswerText', () => {
  it('turns known citation markers into buttons and leaves unknown ones as text', () => {
    const onCite = vi.fn();
    const answer = makeAnswer();
    render(<AnswerText text="Fact one [S1]. Fact two [S4]." citations={answer.citations} onCite={onCite} />);
    const chip = screen.getByRole('button', { name: /Source S1: handbook.pdf, page 2/ });
    fireEvent.click(chip);
    expect(onCite).toHaveBeenCalledWith('S1');
    expect(screen.queryByRole('button', { name: /S4/ })).toBeNull();
    expect(screen.getByText(/\[S4\]/)).toBeInTheDocument();
  });
});

describe('AnswerCard', () => {
  it('renders a fresh grounded answer with expandable citation details', () => {
    render(<AnswerCard result={makeAnswer()} />);
    expect(screen.getByText('Grounded answer')).toBeInTheDocument();
    expect(screen.getByText(/FRESH · openai\/gpt-oss-20b/)).toBeInTheDocument();
    expect(screen.getByText('handbook.pdf')).toBeInTheDocument();
    expect(screen.getByText('p. 2')).toBeInTheDocument();

    expect(screen.queryByText(/26 days of paid vacation per calendar year/)).toBeNull();
    const sourceToggle = screen.getAllByRole('button', { expanded: false })[0];
    fireEvent.click(sourceToggle);
    expect(screen.getByText(/26 days of paid vacation per calendar year/)).toBeInTheDocument();
    expect(screen.getByText('document v1')).toBeInTheDocument();
  });

  it('opens a citation when its inline marker is clicked', () => {
    render(<AnswerCard result={makeAnswer()} />);
    fireEvent.click(screen.getByRole('button', { name: /Source S1/ }));
    expect(screen.getByText(/26 days of paid vacation per calendar year/)).toBeInTheDocument();
  });

  it('labels cache hits and shows avoided cost, not tokens', () => {
    const cached = makeAnswer({
      cache: { hit: true, match_type: 'semantic', similarity: 0.874, lookup_ms: 2.1, rejection_reason: null, bypassed: false },
      usage: { input_tokens: 0, output_tokens: 0, token_source: 'none', estimated_cost: 0, estimated_cost_avoided: 0.000031 },
      latency: { total_ms: 12.5, embedding_ms: 9, cache_lookup_ms: 2.1, retrieval_ms: 0.7, generation_ms: null },
    });
    render(<AnswerCard result={cached} />);
    expect(screen.getByText(/CACHED · semantic · sim 0.874/)).toBeInTheDocument();
    expect(screen.getByText(/est. avoided \$0.000031/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Pipeline details/ }));
    expect(screen.getAllByText('none used')).toHaveLength(2);
    // Generation did not run, so it is reported as unavailable, not as 0 ms.
    expect(screen.getByText('Generation').nextSibling).toHaveTextContent('n/a');
    expect(screen.getByText('Actual billed cost').nextSibling).toHaveTextContent('unavailable');
  });

  it('shows insufficient evidence distinctly with no sources', () => {
    render(
      <AnswerCard
        result={makeAnswer({
          outcome: 'insufficient_evidence',
          answer: 'I could not find enough support for this question in the selected documents.',
          citations: [],
          routing: { selected_model: null, tier: null, reason: 'No model call', policy: 'complexity', factors: [], fallback_used: false, retry_count: 0 },
        })}
      />
    );
    expect(screen.getByText('Insufficient evidence')).toBeInTheDocument();
    expect(screen.getByText('NO MODEL CALL')).toBeInTheDocument();
    expect(screen.queryByLabelText('Sources')).toBeNull();
  });

  it('flags suspicious passages and cache rejections in details', () => {
    const r = makeAnswer({
      cache: { hit: false, similarity: 0.83, lookup_ms: 1, rejection_reason: 'different named entities', bypassed: false },
    });
    r.retrieval.suspicious_passages = 1;
    render(<AnswerCard result={r} />);
    expect(screen.getByText(/1 suspicious passage/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Pipeline details/ }));
    expect(screen.getByText(/Cache candidate rejected \(similarity 0.830\): different named entities/)).toBeInTheDocument();
    expect(screen.getByText('below threshold')).toBeInTheDocument();
  });
});
