"""Text extraction and normalization for uploaded documents."""

import io
import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


class DocumentValidationError(ValueError):
    """Raised for a file that is rejected before or during extraction.

    The message is safe to show to the user.
    """


@dataclass
class PageText:
    page_number: int  # 1-based
    text: str


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")
_INLINE_WS = re.compile(r"[ \t ]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_SINGLE_NEWLINE = re.compile(r"(?<!\n)\n(?!\n)")


def normalize_text(text: str) -> str:
    """Clean extracted text while keeping paragraph boundaries.

    NFKC folds ligatures and full-width forms that PDF extractors emit,
    words hyphenated across line breaks are re-joined, control characters
    are dropped, and runs of whitespace are collapsed.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub("", text)
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)
    text = _INLINE_WS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()


def extract_txt(data: bytes) -> List[PageText]:
    """Decode a plain-text upload. TXT files are treated as a single page."""
    if b"\x00" in data[:8192]:
        raise DocumentValidationError(
            "The file appears to be binary, not plain text."
        )
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise DocumentValidationError(
            "The text file is not valid UTF-8 or UTF-16."
        )
    return [PageText(page_number=1, text=normalize_text(text))]


def extract_pdf(data: bytes, max_pages: int) -> List[PageText]:
    """Extract per-page text from a PDF with page numbers preserved."""
    if not data.startswith(b"%PDF-"):
        raise DocumentValidationError(
            "The file does not have a valid PDF header."
        )
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as exc:  # pragma: no cover - dependency is required
        raise RuntimeError("pypdf is required for PDF extraction.") from exc

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise DocumentValidationError(
                "Encrypted or password-protected PDFs are not supported."
            )
        page_count = len(reader.pages)
    except DocumentValidationError:
        raise
    except (PdfReadError, ValueError, KeyError, TypeError) as exc:
        logger.warning("pdf_parse_failed", extra={"error_type": type(exc).__name__})
        raise DocumentValidationError("The PDF could not be parsed.") from exc

    if page_count > max_pages:
        raise DocumentValidationError(
            f"The PDF has {page_count} pages; the limit is {max_pages}."
        )

    pages: List[PageText] = []
    for index, page in enumerate(reader.pages):
        try:
            raw = page.extract_text() or ""
        except Exception as exc:  # pypdf raises many types on malformed pages
            logger.warning(
                "pdf_page_extract_failed",
                extra={"error_type": type(exc).__name__},
            )
            raw = ""
        # PDF line breaks are layout, not structure: join wrapped lines and
        # keep blank-line paragraph breaks.
        text = _SINGLE_NEWLINE.sub(" ", normalize_text(raw))
        pages.append(PageText(page_number=index + 1, text=text))
    return pages
