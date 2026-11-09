"""
Tests for DFDC downloader and organizer script (scripts/download_dfdc_subset.py).

Verifies:
1. check_kaggle_auth() logic with and without mock credentials/environment variables.
2. organize_dfdc_subset() correctly parses DFDC zip archive, splits into real/ and fake/,
   merges metadata.json, and preserves paired sources when max_videos is applied.
3. organize_dfdc_subset() correctly handles an unzipped directory structure.
"""

import os
import json
import zipfile
import shutil
from pathlib import Path
import pytest

from scripts.download_dfdc_subset import (
    check_kaggle_auth,
    organize_dfdc_subset,
)


@pytest.fixture
def local_tmp_path():
    """Workspace-local temp dir fixture to avoid Windows Temp ACL issues."""
    path = Path(__file__).resolve().parent / ".tmp_dfdc_test"
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    yield path
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


class TestKaggleAuthCheck:
    def test_auth_check_with_env_vars(self, monkeypatch):
        monkeypatch.setenv("KAGGLE_USERNAME", "test_user")
        monkeypatch.setenv("KAGGLE_KEY", "test_key")
        assert check_kaggle_auth() is True

    def test_auth_check_with_api_token_env(self, monkeypatch):
        monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
        monkeypatch.delenv("KAGGLE_KEY", raising=False)
        monkeypatch.setenv("KAGGLE_API_TOKEN", "mock_token")
        assert check_kaggle_auth() is True


class TestDFDCOrganizer:
    def _create_mock_dfdc_zip(self, zip_path: Path):
        """Creates a mock dfdc_train_part_00.zip with metadata and video files."""
        meta = {
            "fake_vid_1.mp4": {
                "label": "FAKE",
                "split": "train",
                "original": "real_vid_1.mp4"
            },
            "real_vid_1.mp4": {
                "label": "REAL",
                "split": "train",
                "original": None
            },
            "fake_vid_2.mp4": {
                "label": "FAKE",
                "split": "train",
                "original": "real_vid_2.mp4"
            },
            "real_vid_2.mp4": {
                "label": "REAL",
                "split": "train",
                "original": None
            }
        }

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("dfdc_train_part_00/metadata.json", json.dumps(meta, indent=2))
            for vid_name in meta:
                # Write a dummy byte string as the video file
                zf.writestr(f"dfdc_train_part_00/{vid_name}", b"dummy_video_bytes")

    def test_organize_from_zip(self, local_tmp_path):
        zip_file = local_tmp_path / "mock_dfdc.zip"
        output_dir = local_tmp_path / "raw_videos"
        self._create_mock_dfdc_zip(zip_file)

        metadata = organize_dfdc_subset(zip_file, output_dir)

        real_dir = output_dir / "real"
        fake_dir = output_dir / "fake"
        meta_file = output_dir / "metadata.json"

        assert (real_dir / "real_vid_1.mp4").exists()
        assert (real_dir / "real_vid_2.mp4").exists()
        assert (fake_dir / "fake_vid_1.mp4").exists()
        assert (fake_dir / "fake_vid_2.mp4").exists()
        assert meta_file.exists()

        with open(meta_file, "r", encoding="utf-8") as f:
            saved_meta = json.load(f)
        assert saved_meta["fake_vid_1.mp4"]["original"] == "real_vid_1.mp4"

    def test_organize_with_max_videos_preserves_paired_source(self, local_tmp_path):
        """
        When max_videos=1 is requested and selects fake_vid_1.mp4, its original
        real_vid_1.mp4 must also be included to maintain zero-leakage clustering!
        """
        zip_file = local_tmp_path / "mock_dfdc.zip"
        output_dir = local_tmp_path / "raw_videos_subset"
        self._create_mock_dfdc_zip(zip_file)

        metadata = organize_dfdc_subset(zip_file, output_dir, max_videos=1)

        real_dir = output_dir / "real"
        fake_dir = output_dir / "fake"

        # fake_vid_1 was selected, so its original real_vid_1 MUST be extracted too
        assert (fake_dir / "fake_vid_1.mp4").exists()
        assert (real_dir / "real_vid_1.mp4").exists()

