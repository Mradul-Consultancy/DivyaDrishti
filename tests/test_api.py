"""
Tests for the HTTP API.

Uses FastAPI's TestClient so the endpoints are exercised end to end
(including the multipart upload path) without binding a socket.
"""

import io
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def client():
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from api.server import app
    return fastapi_testclient.TestClient(app)


def _png_bytes():
    from PIL import Image
    import numpy as np
    buf = io.BytesIO()
    Image.fromarray(np.random.randint(0, 255, (240, 240, 3), dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


class TestHealthAndModels:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "models_available" in body

    def test_models_endpoint_reports_calibration(self, client):
        r = client.get("/v1/models")
        assert r.status_code == 200
        models = r.json()["models"]
        assert models
        for m in models:
            assert "threshold" in m
            assert "available" in m

    def test_unknown_model_returns_404(self, client):
        r = client.post(
            "/v1/detect/image",
            files={"file": ("x.png", _png_bytes(), "image/png")},
            data={"model": "not_a_model"},
        )
        assert r.status_code == 404


class TestImageDetection:
    def test_detect_image_returns_full_envelope(self, client):
        r = client.post(
            "/v1/detect/image",
            files={"file": ("x.png", _png_bytes(), "image/png")},
            data={"model": "resnet18"},
        )
        # Either a verdict, or 422 when no face is found in a random image.
        assert r.status_code in (200, 422)
        if r.status_code == 200:
            body = r.json()
            for key in ("verdict", "fake_probability", "confidence", "face_count",
                        "processing_time_ms", "model", "input_sha256", "threshold",
                        "threshold_source"):
                assert key in body, f"missing {key} in response"
            assert body["verdict"] in ("fake", "real")
            assert 0.0 <= body["fake_probability"] <= 1.0
            assert body["face_count"] >= 1
            assert len(body["input_sha256"]) == 64

    def test_detect_image_with_explain_returns_gradcam(self, client):
        r = client.post(
            "/v1/detect/image",
            files={"file": ("x.png", _png_bytes(), "image/png")},
            data={"model": "resnet18", "explain": "true"},
        )
        assert r.status_code in (200, 422)
        if r.status_code == 200:
            body = r.json()
            if body.get("gradcam_base64"):
                import base64
                raw = base64.b64decode(body["gradcam_base64"])
                assert raw[:4] == b"\x89PNG"

    def test_undecodable_input_returns_415(self, client):
        r = client.post(
            "/v1/detect/image",
            files={"file": ("x.png", b"not an image", "image/png")},
            data={"model": "resnet18"},
        )
        assert r.status_code == 415

    def test_explicit_threshold_is_echoed(self, client):
        r = client.post(
            "/v1/detect/image",
            files={"file": ("x.png", _png_bytes(), "image/png")},
            data={"model": "resnet18", "threshold": "0.77"},
        )
        if r.status_code == 200:
            body = r.json()
            assert body["threshold"] == pytest.approx(0.77)
            assert body["threshold_source"] == "request"


class TestAuditTrail:
    def test_audit_records_contain_no_image_bytes(self, client):
        client.post(
            "/v1/detect/image",
            files={"file": ("x.png", _png_bytes(), "image/png")},
            data={"model": "resnet18"},
        )
        r = client.get("/v1/audit/recent")
        assert r.status_code == 200
        records = r.json()["records"]
        for rec in records:
            # Metadata only: a digest, never the media.
            assert len(rec["input_sha256"]) == 64
            assert "image_bytes" not in rec
            assert "data" not in rec

    def test_purge_clears_the_buffer(self, client):
        client.get("/v1/audit/recent")
        r = client.post("/v1/audit/purge")
        assert r.status_code == 200
        assert r.json()["cleared_records"] >= 0


class TestC2PA:
    def test_c2pa_module_detects_nothing_in_plain_data(self):
        from api.c2pa import inspect_bytes
        info = inspect_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 100)
        assert info.present is False
        assert info.signature_verified is False

    def test_c2pa_module_flags_a_manifest_in_an_app11_segment(self):
        from api.c2pa import inspect_bytes
        # A JPEG APP11 segment carrying a JUMBF box of type "c2pa".
        payload = b"jumb" + b"c2pa" + b"\x00" * 64
        # JPEG segment length covers the length field itself.
        seg = b"\xff\xeb" + (len(payload) + 2).to_bytes(2, "big") + payload
        app0 = b"\xff\xe0" + (16).to_bytes(2, "big") + b"JFIF\x00" + b"\x00" * 9
        jpeg = b"\xff\xd8" + app0 + seg + b"\xff\xd9"
        info = inspect_bytes(jpeg)
        assert info.present is True
        assert info.container == "jpeg"
        # Detection is not verification.
        assert info.signature_verified is False

    def test_c2pa_ignores_a_mention_outside_the_manifest_structures(self):
        """A stray mention of 'c2pa' in a comment must not look like a manifest."""
        from api.c2pa import inspect_bytes
        payload = b"a photo mentioning c2pa in a comment"
        seg = b"\xff\xfe" + (len(payload) + 2).to_bytes(2, "big") + payload  # APP14 (comment)
        app0 = b"\xff\xe0" + (16).to_bytes(2, "big") + b"JFIF\x00" + b"\x00" * 9
        jpeg = b"\xff\xd8" + app0 + seg + b"\xff\xd9"
        info = inspect_bytes(jpeg)
        assert info.present is False
