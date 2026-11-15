"""
Temp-upload hygiene for the Streamlit UI (and any other frontend that has to
spool biometric media to disk before OpenCV can read it).

Retention policy
----------------
* Uploads are spooled into a dedicated subdirectory of the system temp folder
  (``<temp>/divyadrishhti_video``) - never persisted anywhere else.
* The caller is expected to unlink the file as soon as it is processed; the
  session registry plus :func:`cleanup_session_files` removes whatever is
  left when the interpreter exits (registered by the frontend at import).
* If the process dies before that hook runs, :func:`sweep_stale_files`
  removes anything older than ``DEFAULT_TTL_SECONDS`` (6 h) the next time a
  process starts. That is the only way an upload can outlive its session.
"""

import os
import tempfile
import time
from typing import List

DEFAULT_TTL_SECONDS = 6 * 3600  # crash-orphan sweep horizon

TEMP_DIR = os.path.join(tempfile.gettempdir(), "divyadrishhti_video")

_SESSION_FILES: List[str] = []
_SWEEP_DONE = False


def ensure_temp_dir() -> str:
    os.makedirs(TEMP_DIR, exist_ok=True)
    return TEMP_DIR


def register_session_file(path: str) -> None:
    """Track a spooled upload so it can be removed when the session ends."""
    _SESSION_FILES.append(path)


def cleanup_session_files() -> None:
    """Delete every upload this process spooled; safe to call repeatedly."""
    for path in list(_SESSION_FILES):
        try:
            os.unlink(path)
        except OSError:
            pass
    _SESSION_FILES.clear()


def sweep_stale_files(ttl_seconds: int = DEFAULT_TTL_SECONDS, *, _force: bool = False) -> int:
    """
    Best-effort removal of uploads orphaned by a previous crashed run.

    Runs at most once per process (unless ``_force=True``, which exists for
    tests). Returns the number of files removed. Files younger than the TTL
    are left alone - another live process may still be using them.
    """
    global _SWEEP_DONE
    if _SWEEP_DONE and not _force:
        return 0
    _SWEEP_DONE = True
    if not os.path.isdir(TEMP_DIR):
        return 0
    removed = 0
    now = time.time()
    for name in os.listdir(TEMP_DIR):
        path = os.path.join(TEMP_DIR, name)
        try:
            if now - os.path.getmtime(path) > ttl_seconds:
                os.unlink(path)
                removed += 1
        except OSError:
            # The file may be locked by a concurrent process; leave it for
            # the next sweep rather than crashing the frontend.
            continue
    return removed


def register_exit_cleanup() -> None:
    import atexit
    atexit.register(cleanup_session_files)
