"""Document ingestion: validation, extraction, chunking, duplicates, versions,
deletion and re-indexing."""

import sqlite3

import pytest

from app.documents import (
    DocumentStore,
    DocumentValidationError,
    PageText,
    chunk_pages,
    extract_txt,
    normalize_text,
    sanitize_filename,
    validate_collection,
)
from tests.conftest import upload
from tests.pdf_utils import make_pdf

WARRANTY_TXT = (
    b"Aurora X2 Battery Warranty\n\n"
    b"The Aurora X2 battery is covered for 24 months from the date of purchase. "
    b"Claims require the original receipt and the serial number printed under the cover.\n\n"
    b"Water damage and physical drops are not covered by the warranty."
)


def _chunk_rows(db_path, document_id):
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            "SELECT chunk_id, page_number, document_version, text FROM chunks WHERE document_id = ?"
            " ORDER BY chunk_index",
            (document_id,),
        ).fetchall()
    finally:
        conn.close()


# ------------------------------------------------------------- happy paths


def test_txt_upload_is_extracted_chunked_and_indexed(api, tmp_db_path):
    res = upload(api, "warranty.txt", WARRANTY_TXT)
    assert res.status_code == 202
    body = res.json()
    assert body["outcome"] == "created"
    doc_id = body["document"]["document_id"]

    doc = api.get(f"/api/documents/{doc_id}").json()
    assert doc["status"] == "ready"
    assert doc["file_type"] == "txt"
    assert doc["page_count"] == 1
    assert doc["chunk_count"] >= 1
    assert doc["version"] == 1
    assert doc["embedding_model"].startswith("Simulated-Hashing-Embedder")

    rows = _chunk_rows(tmp_db_path, doc_id)
    assert len(rows) == doc["chunk_count"]
    assert all(r[1] == 1 and r[2] == 1 for r in rows)
    assert "24 months" in " ".join(r[3] for r in rows)


def test_pdf_upload_preserves_page_numbers(api, tmp_db_path):
    pdf = make_pdf([
        "Page one covers onboarding. New staff receive a laptop on the first day.",
        "Page two covers travel. Flights must be booked through the travel portal.",
        "Page three covers expenses. Receipts are required for every claim over 25 dollars.",
    ])
    res = upload(api, "Employee Handbook.pdf", pdf)
    assert res.status_code == 202
    doc_id = res.json()["document"]["document_id"]
    doc = api.get(f"/api/documents/{doc_id}").json()
    assert doc["status"] == "ready"
    assert doc["page_count"] == 3

    rows = _chunk_rows(tmp_db_path, doc_id)
    by_page = {r[1]: r[3] for r in rows}
    assert set(by_page) == {1, 2, 3}
    assert "laptop" in by_page[1]
    assert "travel portal" in by_page[2]
    assert "25 dollars" in by_page[3]


def test_documents_are_listed_by_collection(api):
    upload(api, "a.txt", b"Alpha document about apples and orchards.", collection="fruit")
    upload(api, "b.txt", b"Beta document about bridges and steel.", collection="civil")
    fruit = api.get("/api/documents", params={"collection": "fruit"}).json()
    assert [d["filename"] for d in fruit] == ["a.txt"]
    collections = {c["name"]: c for c in api.get("/api/collections").json()}
    assert collections["fruit"]["document_count"] == 1
    assert collections["civil"]["ready_count"] == 1


# --------------------------------------------------------------- rejections


@pytest.mark.parametrize(
    "filename,data,fragment",
    [
        ("notes.docx", b"PK\x03\x04 not supported", "Only .pdf and .txt"),
        ("empty.txt", b"", "empty"),
        ("fake.pdf", b"This is plain text pretending to be a PDF", "not a PDF"),
        ("sneaky.txt", b"%PDF-1.4 a pdf with a txt name", "contains a PDF"),
        ("noext", b"some text", "Only .pdf and .txt"),
    ],
)
def test_invalid_files_are_rejected_with_clear_messages(api, filename, data, fragment):
    res = upload(api, filename, data)
    assert res.status_code == 400
    assert fragment.lower() in res.json()["detail"].lower()
    assert api.get("/api/documents").json() == []


def test_oversized_upload_returns_413(api):
    api.rag.ingestion.max_upload_bytes = 100
    res = upload(api, "big.txt", b"x" * 500)
    assert res.status_code == 413


def test_binary_txt_fails_processing_with_message(api):
    res = upload(api, "binary.txt", b"\x00\x01\x02\x03binary")
    doc = api.get(f"/api/documents/{res.json()['document']['document_id']}").json()
    assert doc["status"] == "failed"
    assert "binary" in doc["error_message"].lower()


def test_pdf_without_text_fails_with_ocr_message(api):
    res = upload(api, "scan.pdf", make_pdf([""]))
    doc = api.get(f"/api/documents/{res.json()['document']['document_id']}").json()
    assert doc["status"] == "failed"
    assert "ocr" in doc["error_message"].lower()


def test_corrupt_pdf_fails_safely(api):
    res = upload(api, "broken.pdf", b"%PDF-1.4\nthis is not really a pdf body")
    doc = api.get(f"/api/documents/{res.json()['document']['document_id']}").json()
    assert doc["status"] == "failed"
    assert doc["error_message"]


def test_invalid_collection_name_rejected(api):
    res = upload(api, "a.txt", b"hello world", collection="../../etc")
    assert res.status_code == 400


# ------------------------------------------------- duplicates and versions


def test_duplicate_upload_is_not_reprocessed(api):
    first = upload(api, "warranty.txt", WARRANTY_TXT).json()
    second = upload(api, "warranty-copy.txt", WARRANTY_TXT).json()
    assert second["outcome"] == "duplicate"
    assert second["document"]["document_id"] == first["document"]["document_id"]
    assert len(api.get("/api/documents").json()) == 1


def test_same_filename_new_content_replaces_with_new_version(api, tmp_db_path):
    first = upload(api, "policy.txt", b"Version one says refunds take 14 days.").json()
    doc_id = first["document"]["document_id"]
    old_chunks = {r[0] for r in _chunk_rows(tmp_db_path, doc_id)}

    second = upload(api, "policy.txt", b"Version two says refunds take 30 days.").json()
    assert second["outcome"] == "replaced"
    assert second["document"]["document_id"] == doc_id  # stable identifier
    doc = api.get(f"/api/documents/{doc_id}").json()
    assert doc["version"] == 2 and doc["status"] == "ready"

    rows = _chunk_rows(tmp_db_path, doc_id)
    assert {r[0] for r in rows}.isdisjoint(old_chunks)
    assert all(r[2] == 2 for r in rows)
    assert "30 days" in rows[0][3]


def test_delete_removes_document_pages_chunks_and_embeddings(api, tmp_db_path):
    doc_id = upload(api, "warranty.txt", WARRANTY_TXT).json()["document"]["document_id"]
    assert _chunk_rows(tmp_db_path, doc_id)

    res = api.delete(f"/api/documents/{doc_id}")
    assert res.status_code == 200
    assert api.get(f"/api/documents/{doc_id}").status_code == 404
    assert _chunk_rows(tmp_db_path, doc_id) == []
    conn = sqlite3.connect(tmp_db_path)
    try:
        pages = conn.execute(
            "SELECT COUNT(*) FROM document_pages WHERE document_id = ?", (doc_id,)
        ).fetchone()[0]
    finally:
        conn.close()
    assert pages == 0
    assert api.delete(f"/api/documents/{doc_id}").status_code == 404


def test_reindex_rebuilds_chunks_under_new_version(api, tmp_db_path):
    doc_id = upload(api, "warranty.txt", WARRANTY_TXT).json()["document"]["document_id"]
    before = _chunk_rows(tmp_db_path, doc_id)

    api.rag.ingestion.chunk_size = 80
    api.rag.ingestion.chunk_overlap = 20
    res = api.post(f"/api/documents/{doc_id}/reindex")
    assert res.status_code == 202
    doc = api.get(f"/api/documents/{doc_id}").json()
    assert doc["status"] == "ready" and doc["version"] == 2

    after = _chunk_rows(tmp_db_path, doc_id)
    assert len(after) > len(before)
    assert all(r[2] == 2 for r in after)
    assert api.post("/api/documents/doc_missing/reindex").status_code == 404


def test_interrupted_processing_is_marked_failed_on_restart(tmp_db_path):
    store = DocumentStore(tmp_db_path)
    from app.documents import DocumentRecord

    store.create(DocumentRecord(
        document_id="doc_x", collection="default", filename="x.txt", file_type="txt",
        size_bytes=1, content_sha256="0", version=1, status="processing",
        error_message=None, page_count=0, chunk_count=0, char_count=0,
        embedding_model=None, created_at=0.0, updated_at=0.0,
    ))
    assert DocumentStore(tmp_db_path).mark_interrupted() == 1
    doc = store.get("doc_x")
    assert doc.status == "failed" and "restart" in doc.error_message


# ------------------------------------------------------------ unit helpers


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("../../etc/passwd.txt", "passwd.txt"),
        ("C:\\Users\\me\\report.PDF", "report.pdf"),
        ("<script>alert(1)</script>.txt", "script_.txt"),
        ("bad\x00name?.txt", "bad_name_.txt"),
        ("  spaced name .txt", "spaced name .txt"),
    ],
)
def test_sanitize_filename_strips_paths_and_unsafe_characters(raw, expected):
    assert sanitize_filename(raw) == expected


def test_sanitize_filename_rejects_empty():
    with pytest.raises(DocumentValidationError):
        sanitize_filename("../..")


def test_collection_validation():
    assert validate_collection("team-docs_1") == "team-docs_1"
    for bad in ("", "a/b", "x" * 65, "-leading"):
        with pytest.raises(DocumentValidationError):
            validate_collection(bad)


def test_normalize_text_cleans_pdf_artifacts():
    raw = "The ﬁnal re-\nport\x00 is   due.\r\n\r\n\r\n\r\nNext   para."
    assert normalize_text(raw) == "The final report is due.\n\nNext para."


def test_extract_txt_handles_bom_and_rejects_invalid_encoding():
    assert extract_txt("\ufeffHello".encode("utf-8"))[0].text == "Hello"
    with pytest.raises(DocumentValidationError):
        extract_txt(b"\xff\xfe\xfa\xfb\xfc" * 3 + b"\xd8")


def test_chunks_overlap_respect_size_and_never_cross_pages():
    sentence = "Sentence number {i} talks about topic {i}. "
    page1 = "".join(sentence.format(i=i) for i in range(40))
    page2 = "Second page has one short line."
    pages = [PageText(1, page1), PageText(2, page2)]
    chunks = chunk_pages(pages, chunk_size=200, overlap=50)

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    p1 = [c for c in chunks if c.page_number == 1]
    assert len(p1) > 1
    for c in p1:
        assert len(c.text) <= 200
        # Offsets point at exactly this text in the page.
        assert page1[c.start_char:c.end_char] == c.text
    for a, b in zip(p1, p1[1:]):
        assert b.start_char < a.end_char, "consecutive chunks should overlap"
    # Chunks end on sentence boundaries when one is available.
    assert all(c.text.endswith(".") for c in p1[:-1])
    p2 = [c for c in chunks if c.page_number == 2]
    assert len(p2) == 1 and p2[0].text == page2


def test_chunking_rejects_bad_parameters():
    with pytest.raises(ValueError):
        chunk_pages([PageText(1, "abc")], chunk_size=0, overlap=0)
    with pytest.raises(ValueError):
        chunk_pages([PageText(1, "abc")], chunk_size=10, overlap=10)
