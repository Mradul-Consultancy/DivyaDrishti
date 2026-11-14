"""
Audit trail and zero-retention handling for the detection API.

Two compliance properties this module is responsible for:

1. **No biometric content is persisted.** Uploaded images and video frames
   are processed in memory only. The audit record keeps a SHA-256 digest of
   the input bytes, which is enough to prove that two records refer to the
   same media without retaining the media itself.

2. **Every decision is attributable.** Each inference appends a record with
   the model, the threshold that was applied, the digest of the input and the
   outcome, so a human reviewer can reconstruct what the system decided and
   why.
"""

import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Audit records are metadata only - never image bytes.
AUDIT_LOG = Path(os.environ.get("DF_AUDIT_LOG", "logs/audit.jsonl"))
MAX_RECORDS_IN_MEMORY = 10_000


@dataclass
class AuditRecord:
    """One inference decision."""

    timestamp: str
    request_id: str
    model: str
    domain: Optional[str]
    threshold: float
    threshold_source: str
    verdict: str
    fake_probability: float
    face_count: int
    input_sha256: str
    processing_time_ms: float
    media_type: str
    notes: List[str] = field(default_factory=list)


_lock = threading.Lock()
_memory_records: List[Dict[str, Any]] = []


def sha256_bytes(data: bytes) -> str:
    """Digest of the raw media bytes."""
    return hashlib.sha256(data).hexdigest()


def record(
    *,
    request_id: str,
    model: str,
    domain: Optional[str],
    threshold: float,
    threshold_source: str,
    verdict: str,
    fake_probability: float,
    face_count: int,
    input_sha256: str,
    processing_time_ms: float,
    media_type: str,
    notes: Optional[List[str]] = None,
) -> AuditRecord:
    """
    Appends an audit record. Only metadata is stored - never the media.
    """
    rec = AuditRecord(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        request_id=request_id,
        model=model,
        domain=domain,
        threshold=threshold,
        threshold_source=threshold_source,
        verdict=verdict,
        fake_probability=round(fake_probability, 6),
        face_count=face_count,
        input_sha256=input_sha256,
        processing_time_ms=round(processing_time_ms, 3),
        media_type=media_type,
        notes=notes or [],
    )
    payload = asdict(rec)

    with _lock:
        _memory_records.append(payload)
        if len(_memory_records) > MAX_RECORDS_IN_MEMORY:
            del _memory_records[: len(_memory_records) - MAX_RECORDS_IN_MEMORY]

    # Append to disk when a log path is configured. The record still contains
    # no image content.
    try:
        AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass  # auditing must never break inference

    return rec


def recent(limit: int = 50) -> List[Dict[str, Any]]:
    """Returns the most recent audit records (metadata only)."""
    with _lock:
        return list(_memory_records[-limit:])


def wipe_in_memory(limit: int = 50) -> int:
    """
    Clears the in-memory audit buffer.

    Provided for deployments that require the decision trail to be held only
    for the duration of an investigation. The on-disk log is unaffected.
    """
    with _lock:
        n = len(_memory_records)
        _memory_records.clear()
        return n


def zero_buffer(buf: bytearray) -> None:
    """
    Overwrites a byte buffer in place.

    Python cannot guarantee that an immutable ``bytes`` object is released
    promptly, so callers that need to wipe sensitive data should hold it in a
    mutable bytearray and call this before dropping the reference.
    """
    for i in range(len(buf)):
        buf[i] = 0
