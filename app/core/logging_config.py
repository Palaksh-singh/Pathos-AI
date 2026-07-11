"""
Pathos AI — Logging Configuration
====================================
Structured JSON logs, safe for shipping to an external log aggregator.
Two safety properties are enforced here rather than left to caller
discipline:

1. `PIIRedactionFilter` runs the privacy engine's regex layer over every
   log message before emission, so even an accidental `logger.info(raw_text)`
   call somewhere in the codebase doesn't leak PII into log storage.
2. Log records never include the raw `pii_map` even if it's passed in
   `extra=` — it's stripped explicitly in the formatter.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

_SENSITIVE_EXTRA_KEYS = {"pii_map", "raw_message", "password", "authorization"}


class PIIRedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Deferred import avoids a circular import at logging setup time
        # (privacy_engine imports schemas, not logging_config).
        from app.services.privacy_engine import privacy_engine

        if isinstance(record.msg, str):
            record.msg = privacy_engine.redact_for_logging(record.msg)

        for key in list(record.__dict__.keys()):
            if key in _SENSITIVE_EXTRA_KEYS:
                setattr(record, key, "[REDACTED]")
        return True


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        reserved = set(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime"}
        for key, value in record.__dict__.items():
            if key not in reserved and key not in ("msg", "args"):
                if key in _SENSITIVE_EXTRA_KEYS:
                    payload[key] = "[REDACTED]"
                else:
                    payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JSONFormatter())
    handler.addFilter(PIIRedactionFilter())
    root_logger.addHandler(handler)

    # Quiet down noisy third-party loggers by default.
    for noisy_logger in ("httpx", "httpcore", "uvicorn.access"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
