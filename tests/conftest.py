"""
Shared pytest configuration.

Temp-root workaround: a once-killed run left ``%TEMP%/pytest-of-<user>`` with
a corrupt ACL on this machine, and every subsequent run failed to create a
numbered directory inside it. Point pytest's numbered-temp root at a clean
subdirectory instead; harmless everywhere else and self-healing.
"""

import os
import tempfile

_PYTEST_TMP = os.path.join(tempfile.gettempdir(), "divyadrishhti-pytest-tmp")
os.makedirs(_PYTEST_TMP, exist_ok=True)
tempfile.tempdir = _PYTEST_TMP
