"""
C2PA / Content Credentials presence detection.

C2PA (Coalition for Content Provenance and Authenticity) signs media with a
cryptographic manifest carried in the file itself. When a manifest is
present and valid, a detector's verdict is only one input: the provenance
chain is authoritative about how the media was produced.

This module *detects* the presence of a manifest and extracts the basic
claims. It does not verify signatures - that requires the C2PA SDK (the
`c2pa-python` package), which is not installable in an offline sandbox.
Callers are told which of the two happened so a verdict is never presented
as if a signature had been checked.
"""

import json
import re
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# JUMBF box type used by C2PA for the manifest store ("c2pa").
C2PA_BOX_TYPE = b"c2pa"
# C2PA claim / manifest generator identifiers seen in the wild.
C2PA_MARKERS = (
    "c2pa", "contentcredentials", "content credentials",
    "claim_generator", "c2pa.assertions", "jumbf",
)


@dataclass
class C2PAInfo:
    """What could be determined about a media file's provenance."""

    present: bool = False
    container: Optional[str] = None      # "jpeg" / "png" / "unknown"
    signatures: int = 0                  # number of signature boxes found
    asset_type: Optional[str] = None
    generator: Optional[str] = None
    raw_manifest_bytes: int = 0
    claims: List[Dict[str, Any]] = field(default_factory=list)
    signature_verified: bool = False
    note: str = (
        "Presence of a C2PA manifest was detected, but the signature was NOT "
        "cryptographically verified: that requires the C2PA SDK, which is not "
        "available here. Treat the provenance chain as unconfirmed."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "present": self.present,
            "container": self.container,
            "signatures": self.signatures,
            "asset_type": self.asset_type,
            "generator": self.generator,
            "raw_manifest_bytes": self.raw_manifest_bytes,
            "claims": self.claims,
            "signature_verified": self.signature_verified,
            "note": self.note,
        }



def _jpeg_app_segments(data: bytes):
    """Returns the payload of every JPEG APP11 segment (where JUMBF lives)."""
    segments = []
    i = 2
    n = len(data)
    while i + 4 <= n:
        if data[i] != 0xFF:
            break
        marker = data[i + 1]
        if marker == 0xDA:          # start of scan - no more metadata
            break
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        seg_len = int.from_bytes(data[i + 2:i + 4], "big")
        if seg_len < 2 or i + 2 + seg_len > n:
            break
        if marker == 0xEB:           # APP11
            segments.append(data[i + 4:i + 2 + seg_len])
        i += 2 + seg_len
    return segments


def _png_itxt_chunks(data: bytes):
    """Returns the payload of every PNG iTXt chunk (where JUMBF may live)."""
    chunks = []
    i = 8
    n = len(data)
    while i + 8 <= n:
        length = int.from_bytes(data[i:i + 4], "big")
        ctype = data[i + 4:i + 8]
        payload = data[i + 8:i + 8 + length]
        if ctype == b"iTXt":
            chunks.append(payload)
        i += 12 + length   # length + type + data + crc
        if ctype == b"IEND":
            break
    return chunks


def _scan_bytes(data: bytes) -> C2PAInfo:
    """Scans raw bytes for C2PA/JUMBF markers."""
    info = C2PAInfo()

    # JPEG APP11 segments carry a JUMBF "jumb" superbox; the C2PA manifest
    # lives inside as a "c2pa" box.
    # A C2PA manifest lives in a JUMBF box whose type is literally "c2pa",
    # inside a JPEG APP11 segment or a PNG iTXt chunk. Searching the whole
    # file for the substring "c2pa" would fire on any JPEG whose EXIF or
    # comment happens to mention it, so the search is restricted to the
    # structures that can actually carry a manifest.
    if data[:2] == b"\xff\xd8":
        info.container = "jpeg"
        segments = _jpeg_app_segments(data)
        info.signatures = sum(seg.count(b"c2pa") + seg.count(b"C2PA") for seg in segments)
    elif data[:8] == b"\x89PNG\r\n\x1a\n":
        info.container = "png"
        chunks = _png_itxt_chunks(data)
        info.signatures = sum(c.count(b"c2pa") + c.count(b"C2PA") for c in chunks)
    else:
        info.container = "unknown"

    if info.signatures:
        info.present = True
        info.raw_manifest_bytes = _extract_jumbf_payload_size(data)

    # Try to pull the generator name out of the manifest JSON, if present.
    m = re.search(rb'"generator"\s*:\s*"([^"]{1,120})"', data)
    if m:
        info.generator = m.group(1).decode("utf-8", "replace")
    m = re.search(rb'"alg"\s*:\s*"([^"]{1,60})"', data)
    if m:
        info.claims.append({"alg": m.group(1).decode("utf-8", "replace")})
    m = re.search(rb'"claim_generator_info"\s*:\s*\{[^}]*?"name"\s*:\s*"([^"]{1,120})"', data, re.S)
    if m:
        info.generator = info.generator or m.group(1).decode("utf-8", "replace")

    return info


def _extract_jumbf_payload_size(data: bytes) -> int:
    """
    Best-effort size of the JUMBF manifest payload.

    The JUMBF box header is: 4-byte type, 4-byte length (big endian).
    """
    total = 0
    idx = 0
    while True:
        idx = data.find(C2PA_BOX_TYPE, idx)
        if idx < 0:
            break
        if idx + 8 <= len(data):
            (length,) = struct.unpack(">I", data[idx + 4: idx + 8])
            if 0 < length < len(data):
                total += length
        idx += 4
    return total


def inspect_bytes(data: bytes) -> C2PAInfo:
    """
    Inspects raw media bytes for a C2PA / Content Credentials manifest.
    """
    if not data:
        return C2PAInfo(note="empty input")
    return _scan_bytes(data)


def inspect_file(path) -> C2PAInfo:
    """Inspects a file on disk for a C2PA manifest."""
    try:
        with open(path, "rb") as f:
            data = f.read()
    except Exception:
        return C2PAInfo(note="file could not be read")
    return inspect_bytes(data)
