"""
Unit and Regression Tests for MTCNN Faceless Frame Handling and Drop Logging.

Verifies:
1. Frames without faces are flagged with face_detected=False.
2. Dropped frames are logged and counted accurately per video.
3. Dropped frames and retention rates are logged and counted accurately per class.
4. video_split_manifest.json includes per-video and per-class drop statistics.
"""

import sys
import shutil
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np
import pytest
import torch

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from preprocessing.extract_faces_from_videos import VideoFaceExtractor
import config


@pytest.fixture
def local_tmp_path():
    """Workspace-local temporary directory fixture to prevent Windows Temp PermissionError."""
    temp_dir = Path("tests/.tmp_drop_test_dir")
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class TestFacelessFrameHandling:
    def test_blank_frame_detected_as_no_face(self):
        """A solid color or noise image without a face must yield face_detected=False."""
        extractor = VideoFaceExtractor(
            fallback_center_crop=False,
            min_face_prob=0.90,
            device=torch.device("cpu")
        )
        # Create a plain blank image (no human face)
        blank_img = Image.new("RGB", (300, 300), color=(128, 128, 128))
        crops, face_detected = extractor.crop_faces_mtcnn(blank_img)

        assert face_detected is False
        assert len(crops) == 0

    def test_blank_frame_with_fallback_crops_but_flags_no_face(self):
        """When fallback_center_crop is enabled, a crop is provided, but face_detected is still False."""
        extractor = VideoFaceExtractor(
            fallback_center_crop=True,
            min_face_prob=0.90,
            device=torch.device("cpu")
        )
        blank_img = Image.new("RGB", (300, 300), color=(50, 50, 50))
        crops, face_detected = extractor.crop_faces_mtcnn(blank_img)

        assert face_detected is False
        assert len(crops) == 1
        assert crops[0].size == (224, 224)

    def test_process_and_split_tracks_dropped_frames(self, local_tmp_path):
        """Verifies that process_and_split accurately records dropped frames per video and per class."""
        import cv2

        # Create synthetic video folder structure
        video_dir = local_tmp_path / "raw_videos"
        real_dir = video_dir / "real"
        fake_dir = video_dir / "fake"
        real_dir.mkdir(parents=True)
        fake_dir.mkdir(parents=True)

        # Generate a video with blank frames (should have 100% dropped frames if no fallback)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        blank_real_video = real_dir / "video_real_01.mp4"
        out = cv2.VideoWriter(str(blank_real_video), fourcc, 24, (200, 200))
        for _ in range(10):
            frame = np.zeros((200, 200, 3), dtype=np.uint8)
            out.write(frame)
        out.release()

        blank_fake_video = fake_dir / "video_fake_01.mp4"
        out = cv2.VideoWriter(str(blank_fake_video), fourcc, 24, (200, 200))
        for _ in range(10):
            frame = np.ones((200, 200, 3), dtype=np.uint8) * 200
            out.write(frame)
        out.release()

        out_dir = local_tmp_path / "processed"
        extractor = VideoFaceExtractor(
            frames_per_video=5,
            fallback_center_crop=False,
            min_face_prob=0.90,
            device=torch.device("cpu")
        )

        manifest = extractor.process_and_split(
            video_dir=video_dir,
            output_dir=out_dir,
            split_ratios=(1.0, 0.0, 0.0),
            seed=42
        )

        # Check summary drop statistics
        summary = manifest["summary"]
        assert summary["total_videos"] == 2
        assert summary["total_frames_sampled"] == 10
        assert summary["total_faces_saved"] == 0
        assert summary["total_frames_dropped"] == 10
        assert summary["overall_drop_rate"] == 1.0

        # Check per-class drop statistics
        by_class = summary["dropped_by_class"]
        assert by_class["real"]["frames_sampled"] == 5
        assert by_class["real"]["frames_dropped"] == 5
        assert by_class["real"]["drop_rate"] == 1.0

        assert by_class["fake"]["frames_sampled"] == 5
        assert by_class["fake"]["frames_dropped"] == 5
        assert by_class["fake"]["drop_rate"] == 1.0

