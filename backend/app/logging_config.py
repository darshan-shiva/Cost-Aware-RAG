
import json
import logging
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """Format log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include explicitly supplied structured fields.
        for field in (
            "request_id",
            "method",
            "path",
            "status_code",
            "latency_ms",
            "cache_hit",
            "selected_model",
            "error_type",
        ):
            value = getattr(record, field, None)
            if value is not None:
                log_entry[field] = value

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


def configure_logging() -> None:
    """Configure consistent JSON logging to stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    # Keep access logs available without excessive verbosity.
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
