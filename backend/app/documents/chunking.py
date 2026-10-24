"""Page-aware, overlapping character-window chunking."""

from dataclasses import dataclass
from typing import List

from .extraction import PageText


@dataclass
class TextChunk:
    chunk_index: int  # position within the document, 0-based
    page_number: int
    text: str
    start_char: int  # offsets within the page's normalized text
    end_char: int


# Preferred break points, strongest first.
_BREAKS = ("\n\n", ". ", "? ", "! ", "\n", "; ", ", ", " ")


def _find_break(text: str, start: int, hard_end: int, min_end: int) -> int:
    """Return an end offset in [min_end, hard_end] at the strongest boundary."""
    window = text[min_end:hard_end]
    for marker in _BREAKS:
        pos = window.rfind(marker)
        if pos != -1:
            return min_end + pos + len(marker)
    return hard_end


def chunk_pages(
    pages: List[PageText],
    chunk_size: int,
    overlap: int,
    min_chunk_chars: int = 1,
) -> List[TextChunk]:
    """Split pages into overlapping chunks that never cross a page boundary.

    Keeping chunks within a page means every chunk has exactly one page
    number, so citations can name the page without guessing. Windows end at
    the strongest nearby boundary (paragraph, sentence, clause, word) in the
    last 30% of the window, and the next window starts ``overlap`` characters
    before the previous end, snapped forward to a word boundary.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks: List[TextChunk] = []
    for page in pages:
        text = page.text
        n = len(text)
        start = 0
        while start < n:
            hard_end = min(start + chunk_size, n)
            if hard_end < n:
                min_end = start + int(chunk_size * 0.7)
                end = _find_break(text, start, hard_end, min_end)
            else:
                end = n
            piece = text[start:end].strip()
            # Short trailing fragments are mostly covered by the previous
            # window's overlap; a short page on its own is still kept.
            if piece and (len(piece) >= min_chunk_chars or start == 0):
                lead = len(text[start:end]) - len(text[start:end].lstrip())
                chunks.append(
                    TextChunk(
                        chunk_index=len(chunks),
                        page_number=page.page_number,
                        text=piece,
                        start_char=start + lead,
                        end_char=start + lead + len(piece),
                    )
                )
            if end >= n:
                break
            next_start = max(end - overlap, start + 1)
            # Snap forward so the overlap does not begin mid-word.
            space = text.find(" ", next_start, end)
            if space != -1 and overlap > 0:
                next_start = space + 1
            start = next_start
    return chunks
