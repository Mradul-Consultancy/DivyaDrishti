"""
Tests for utils/temp_uploads.py - the Streamlit UI's upload retention policy.

The API spools uploads to an unlinked temp file that dies with the request,
but the Streamlit frontend must keep the file alive for the duration of a
page render, so it relies on tracked cleanup + a stale sweep. These tests
verify that nothing can outlive its session except the documented crash
window (files older than the TTL when the process died).
"""

import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import temp_uploads


@pytest.fixture()
def fresh_registry(monkeypatch):
    """Isolate the module-level session registry between tests."""
    monkeypatch.setattr(temp_uploads, "_SESSION_FILES", [])
    monkeypatch.setattr(temp_uploads, "_SWEEP_DONE", False)
    yield


def test_sweep_removes_only_files_older_than_ttl(tmp_path, monkeypatch, fresh_registry):
    old = tmp_path / "old.mp4"
    new = tmp_path / "new.mp4"
    old.write_bytes(b"x")
    new.write_bytes(b"y")
    # Backdate one file beyond the TTL, leave the other fresh.
    backdated = time.time() - (temp_uploads.DEFAULT_TTL_SECONDS + 60)
    os.utime(old, (backdated, backdated))
    monkeypatch.setattr(temp_uploads, "TEMP_DIR", str(tmp_path))

    removed = temp_uploads.sweep_stale_files()
    assert removed == 1
    assert not old.exists()
    assert new.exists()  # a live session may still be using it


def test_sweep_runs_once_per_process_unless_forced(tmp_path, monkeypatch, fresh_registry):
    monkeypatch.setattr(temp_uploads, "TEMP_DIR", str(tmp_path))
    victim = tmp_path / "victim.mp4"
    victim.write_bytes(b"x")
    backdated = time.time() - (temp_uploads.DEFAULT_TTL_SECONDS + 60)
    os.utime(victim, (backdated, backdated))

    assert temp_uploads.sweep_stale_files() == 1
    # Second run is a no-op: the sweep already happened this process.
    assert temp_uploads.sweep_stale_files() == 0
    # ... but _force reruns it (the tests' escape hatch / new start).
    other = tmp_path / "other.mp4"
    other.write_bytes(b"x")
    os.utime(other, (backdated, backdated))
    assert temp_uploads.sweep_stale_files(_force=True) == 1


def test_cleanup_session_files_removes_tracked_uploads(tmp_path, fresh_registry):
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    a.write_bytes(b"1")
    b.write_bytes(b"2")
    temp_uploads.register_session_file(str(a))
    temp_uploads.register_session_file(str(b))

    temp_uploads.cleanup_session_files()
    assert not a.exists() and not b.exists()
    # Repeated calls are safe.
    temp_uploads.cleanup_session_files()


def test_cleanup_survives_already_deleted_files(tmp_path, fresh_registry):
    ghost = tmp_path / "ghost.mp4"
    temp_uploads.register_session_file(str(ghost))
    temp_uploads.cleanup_session_files()  # must not raise


def test_uploads_land_in_the_dedicated_directory(monkeypatch, tmp_path, fresh_registry):
    monkeypatch.setattr(temp_uploads, "TEMP_DIR", str(tmp_path / "divyadrishhti_video"))
    d = temp_uploads.ensure_temp_dir()
    assert d.endswith("divyadrishhti_video")
    assert Path(d).is_dir()
