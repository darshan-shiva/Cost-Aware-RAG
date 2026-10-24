"""Cheap lexical checks that stop a semantic cache from conflating questions.

Sentence embeddings place "What is the capital of France?" and "What is the
capital of Germany?" very close together, so similarity alone is not a safe
reuse signal. These checks reject a semantic match when the two questions
mention different numbers or different named entities. They are a heuristic:
lowercase entity names and paraphrased entities are not detected, which is
why RAG answers also require matching retrieved evidence (see rag/service.py).
"""

import re
import unicodedata
from typing import Optional, Set

_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)*%?")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")
_SENTENCE_START = re.compile(r"(?:^|[.?!]\s+)([A-Za-z][A-Za-z0-9'\-]*)")
_QUOTED = re.compile(r"[\"“”]([^\"“”]{2,60})[\"“”]")

# Capitalised words that are usually not entities when they start a question.
_COMMON_CAPITALISED = {
    "i", "a", "an", "the", "what", "which", "who", "whom", "whose", "when",
    "where", "why", "how", "is", "are", "was", "were", "do", "does", "did",
    "can", "could", "should", "would", "will", "please", "explain", "describe",
    "list", "tell", "give", "summarize", "summarise", "compare", "define",
}


def normalize_query(text: Optional[str]) -> str:
    """Canonical form used for exact-match cache keys."""
    text = unicodedata.normalize("NFKC", text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text.rstrip(" ?!.")


def _numbers(text: str) -> Set[str]:
    return {n.replace(",", "") for n in _NUMBER.findall(text)}


def _entities(text: str) -> Set[str]:
    starts = {m.start(1) for m in _SENTENCE_START.finditer(text)}
    found: Set[str] = set()
    for match in _WORD.finditer(text):
        word = match.group(0)
        is_acronym = len(word) >= 2 and word.isupper()
        is_capitalised = word[0].isupper() and match.start() not in starts
        if (is_acronym or is_capitalised) and word.lower() not in _COMMON_CAPITALISED:
            found.add(word.lower().rstrip("s") if is_acronym else word.lower())
    found.update(q.lower().strip() for q in _QUOTED.findall(text))
    return found


def salient_terms_conflict(query_a: str, query_b: str) -> Optional[str]:
    """Return a reason string if the queries name different specifics."""
    nums_a, nums_b = _numbers(query_a), _numbers(query_b)
    if nums_a != nums_b:
        return "different numbers"
    ents_a, ents_b = _entities(query_a), _entities(query_b)
    if ents_a and ents_b and ents_a != ents_b:
        return "different named entities"
    return None


_TERM = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "do", "does",
    "did", "of", "in", "on", "at", "to", "for", "from", "by", "with", "and",
    "or", "what", "which", "who", "whom", "when", "where", "why", "how",
    "many", "much", "can", "could", "should", "would", "will", "i", "we",
    "you", "they", "it", "its", "this", "that", "these", "those", "there",
    "my", "our", "your", "their", "me", "us", "please", "tell", "about",
    "any", "some", "get", "gets", "s",
}


def _terms(text: str) -> Set[str]:
    out = set()
    for token in _TERM.findall((text or "").lower()):
        if token in _STOPWORDS:
            continue
        out.add(token[:-1] if len(token) > 3 and token.endswith("s") else token)
    return out


def contrastive_terms_conflict(query_a: str, query_b: str, evidence_text: str) -> Optional[str]:
    """Detect two questions that ask about different parts of the same passage.

    If each question has a term the other lacks, and the evidence passage
    mentions both of those terms (e.g. "full-time" vs "part-time", both in the
    vacation paragraph), the questions are probably contrasting two things the
    passage distinguishes, so one answer must not serve both. A paraphrase
    usually swaps in words the passage does not use ("get" vs "receive"), so
    it passes.
    """
    only_a = _terms(query_a) - _terms(query_b)
    only_b = _terms(query_b) - _terms(query_a)
    if not only_a or not only_b:
        return None
    evidence = _terms(evidence_text)
    in_a = sorted(only_a & evidence)
    in_b = sorted(only_b & evidence)
    if in_a and in_b:
        return f"contrasting terms in the same evidence ({in_a[0]} vs {in_b[0]})"
    return None


def uncovered_terms_conflict(
    new_query: str, cached_query: str, evidence_text: str, cached_answer: str
) -> Optional[str]:
    """Reject reuse when the cached answer ignores what the new question asks.

    "What is the capacity of the X3?" and "What is the warranty of the X3?"
    can be close in embedding space and retrieve the same passages. If the
    new question has a term the cached one lacks, the passages use that term
    (so it is a real topic, not filler), and the cached answer never mentions
    it, the answer was written for a different question.
    """
    only_new = _terms(new_query) - _terms(cached_query)
    uncovered = sorted((only_new & _terms(evidence_text)) - _terms(cached_answer))
    if uncovered:
        return f"cached answer does not address '{uncovered[0]}'"
    return None
