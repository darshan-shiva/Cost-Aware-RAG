import hashlib

from ..config import settings


def telemetry_text(text: str, limit: int = 2000) -> str:
    """Query or response text as it should be persisted in telemetry.

    With RAG_TELEMETRY_STORE_QUERY_TEXT=false only a short hash is kept, which
    still lets repeated queries be correlated without retaining their content.
    """
    if settings.TELEMETRY_STORE_QUERY_TEXT:
        return (text or "")[:limit]
    digest = hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:12]
    return f"[redacted sha256:{digest}]"
