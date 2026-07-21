"""Structured (JSON) logging setup.

One JSON object per line: fixed fields (timestamp/level/logger/message) plus
whatever the caller passed via ``extra=`` — e.g. ``trigger_id``, ``episode_id``,
``gym``, ``action`` (see ``app.engine``). That's what makes a trigger's
correlation ids actually filterable/queryable downstream (Cloud Logging, jq,
...) instead of living inside a free-text message that has to be parsed.
"""

from __future__ import annotations

import json
import logging
import time

# Fields present on every stdlib LogRecord; anything else on the record came
# from the caller's ``extra=`` and should be surfaced as its own JSON key.
_RESERVED_FIELDS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
}


class JsonFormatter(logging.Formatter):
    """Renders each ``LogRecord`` as a single-line JSON object, in UTC."""

    converter = time.gmtime  # type: ignore[assignment]  # UTC, matching the old format

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S") + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(
            (k, v) for k, v in record.__dict__.items() if k not in _RESERVED_FIELDS
        )
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Install the JSON formatter as the sole handler on the root logger."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=level, handlers=[handler], force=True)
