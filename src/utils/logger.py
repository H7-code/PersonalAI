"""
ARIA V3 — Mandatory Rotating JSONL Logger
Implements Section 28 & Phase 12 Logging Specifications:
- Rotating JSON Lines (JSONL) file logging: session_*.jsonl
- Mandatory rotation constraints: maxBytes = 10 MB (10 * 1024 * 1024), backupCount = 5
- Thread-safe, non-blocking formatted logging
- Disk-error fault tolerance: logging failure never crashes the application
- Zero external dependencies (uses standard library logging & json)
"""

import json
import logging
import logging.handlers
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Default constraints per V3 Specification
DEFAULT_LOG_DIR = Path("logs")
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_BACKUP_COUNT = 5


class JSONLFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "epoch": record.created,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include contextual attributes if attached to record
        if hasattr(record, "request_id"):
            log_entry["request_id"] = getattr(record, "request_id")
        if hasattr(record, "latency_ms"):
            log_entry["latency_ms"] = getattr(record, "latency_ms")
        if hasattr(record, "lang_mode"):
            log_entry["lang_mode"] = getattr(record, "lang_mode")
        if hasattr(record, "extra_data") and isinstance(record.extra_data, dict):
            log_entry["extra"] = record.extra_data

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


class FaultTolerantRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """
    RotatingFileHandler with fault tolerance to ensure disk/permission errors
    do not crash voice assistant worker threads.
    """

    def handleError(self, record: logging.LogRecord):
        # Fail silently to console rather than propagating exception to worker
        try:
            sys.stderr.write(f"[ARIA Log Error] Failed to write log: {record.getMessage()}\n")
        except Exception:
            pass


_logger_lock = threading.Lock()
_initialized = False


def setup_logger(
    log_dir: Path = DEFAULT_LOG_DIR,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    log_level: int = logging.INFO,
    console_output: bool = True,
) -> logging.Logger:
    """
    Initializes root ARIA logger with rotating JSONL handler and console handler.
    """
    global _initialized
    with _logger_lock:
        root_logger = logging.getLogger("aria")
        root_logger.setLevel(log_level)

        if _initialized:
            return root_logger

        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        session_filename = f"session_{datetime.now().strftime('%Y%m%d')}.jsonl"
        log_file = log_dir / session_filename

        # 1. Rotating JSONL File Handler (10 MB x 5 backups)
        file_handler = FaultTolerantRotatingFileHandler(
            str(log_file),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(JSONLFormatter())
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)

        # 2. Console Handler (clean human-readable formatting)
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_fmt = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
                datefmt="%H:%M:%S",
            )
            console_handler.setFormatter(console_fmt)
            console_handler.setLevel(log_level)
            root_logger.addHandler(console_handler)

        _initialized = True
        root_logger.info(
            f"ARIA rotating logger initialized: {log_file} (max {max_bytes / 1024 / 1024:.1f} MB, {backup_count} backups)"
        )
        return root_logger


def get_logger(name: str = "aria") -> logging.Logger:
    """Returns a namespaced logger."""
    if not _initialized:
        setup_logger()
    return logging.getLogger(f"aria.{name}" if not name.startswith("aria") else name)
