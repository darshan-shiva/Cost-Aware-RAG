"""Write the evaluation corpus files from corpus.json (run from backend/)."""

import json
from pathlib import Path

from tests.pdf_utils import make_pdf

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"


def build() -> list:
    spec = json.loads((CORPUS_DIR / "corpus.json").read_text(encoding="utf-8"))
    written = []
    for name, content in spec.items():
        path = CORPUS_DIR / name
        if name.endswith(".pdf"):
            path.write_bytes(make_pdf(content))
        else:
            path.write_text(content, encoding="utf-8")
        written.append(path)
    return written


if __name__ == "__main__":
    for p in build():
        print(f"wrote {p.relative_to(CORPUS_DIR.parent.parent)} ({p.stat().st_size} bytes)")
