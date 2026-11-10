"""
Unit and Regression Tests for Identity- and Source-Aware Dataset Splitting.

Verifies:
1. Disjoint-Set union-find clustering logic.
2. Dataset naming convention key extraction (FF++, Celeb-DF, DFDC, custom).
3. Benchmark metadata parsers (DFDC, Celeb-DF, FF++).
4. Connected component clustering of real source videos and derivative fakes.
5. Zero identity/source leakage across train, val, and test splits.
6. Rejection of data leakage: DataLeakageError is raised on intentional split contamination.
"""

import json
import shutil
import tempfile
import pytest
from pathlib import Path
from preprocessing.split_by_identity import (
    IdentitySourceSplitter,
    DisjointSet,
    DataLeakageError,
)


@pytest.fixture
def local_tmp_path():
    """Workspace-local temporary directory fixture to prevent Windows Temp PermissionError."""
    temp_dir = Path("tests/.tmp_test_dir")
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class TestDisjointSet:
    def test_union_and_find(self):
        ds = DisjointSet()
        ds.union("a", "b")
        ds.union("b", "c")
        ds.union("d", "e")

        assert ds.find("a") == ds.find("c")
        assert ds.find("a") != ds.find("d")
        assert ds.find("d") == ds.find("e")


class TestKeyExtraction:
    def test_ffpp_patterns(self):
        splitter = IdentitySourceSplitter()
        # FF++ real
        k1, k2 = splitter.extract_identity_or_source_key("042.mp4")
        assert k1 == "042"
        assert k2 is None

        # FF++ swap
        k1, k2 = splitter.extract_identity_or_source_key("042_183.mp4")
        assert k1 == "042"
        assert k2 == "183"

    def test_celeb_df_patterns(self):
        splitter = IdentitySourceSplitter()
        # Celeb-DF real
        k1, k2 = splitter.extract_identity_or_source_key("id3_0005.mp4")
        assert k1 == "id3"
        assert k2 is None

        # Celeb-DF swap
        k1, k2 = splitter.extract_identity_or_source_key("id3_id7_0002.mp4")
        assert k1 == "id3"
        assert k2 == "id7"

    def test_sample_and_actor_patterns(self):
        splitter = IdentitySourceSplitter()
        k1, k2 = splitter.extract_identity_or_source_key("video_real_01.mp4")
        assert k1 == "01"
        assert k2 is None

        k1, k2 = splitter.extract_identity_or_source_key("video_fake_01.mp4")
        assert k1 == "01"
        assert k2 is None

        k1, k2 = splitter.extract_identity_or_source_key("actor_05_interview.mp4")
        assert k1 == "actor_05"
        assert k2 is None


class TestBenchmarkParsers:
    def test_dfdc_metadata_parser(self, local_tmp_path):
        meta_file = local_tmp_path / "metadata.json"
        meta_data = {
            "fake_a.mp4": {"label": "FAKE", "original": "real_a.mp4", "actor_id": "actor_1"},
            "real_a.mp4": {"label": "REAL", "actor_id": "actor_1"}
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta_data, f)

        parsed = IdentitySourceSplitter.parse_dfdc_metadata(meta_file)
        assert len(parsed) == 2
        assert parsed["fake_a.mp4"]["original"] == "real_a.mp4"

    def test_celeb_df_parser(self, local_tmp_path):
        list_file = local_tmp_path / "List_of_testing_videos.txt"
        with open(list_file, "w", encoding="utf-8") as f:
            f.write("1 Celeb-synthesis/id0_id1_0000.mp4\n")
            f.write("0 Celeb-real/id0_0000.mp4\n")

        parsed = IdentitySourceSplitter.parse_celeb_df_test_list(list_file)
        assert "id0_id1_0000.mp4" in parsed
        assert "id0_0000.mp4" in parsed

    def test_ffpp_parser(self, local_tmp_path):
        splits_dir = local_tmp_path / "splits"
        splits_dir.mkdir()
        with open(splits_dir / "train.json", "w", encoding="utf-8") as f:
            json.dump([["001", "002"]], f)
        with open(splits_dir / "val.json", "w", encoding="utf-8") as f:
            json.dump([["003"]], f)
        with open(splits_dir / "test.json", "w", encoding="utf-8") as f:
            json.dump([["004"]], f)

        parsed = IdentitySourceSplitter.parse_ffpp_official_splits(splits_dir)
        assert "001" in parsed["train"]
        assert "002" in parsed["train"]
        assert "003" in parsed["val"]
        assert "004" in parsed["test"]


class TestClusteringAndSplitting:
    def test_source_video_clustering(self):
        splitter = IdentitySourceSplitter()
        videos = [
            Path("video_real_01.mp4"),
            Path("video_fake_01.mp4"),
            Path("video_real_02.mp4"),
            Path("video_fake_02.mp4"),
            Path("video_real_03.mp4"),
            Path("video_fake_03.mp4"),
        ]
        metadata = {
            "video_fake_01.mp4": {"original": "video_real_01.mp4", "actor_id": "01"},
            "video_real_01.mp4": {"actor_id": "01"},
            "video_fake_02.mp4": {"original": "video_real_02.mp4", "actor_id": "02"},
            "video_real_02.mp4": {"actor_id": "02"},
            "video_fake_03.mp4": {"original": "video_real_03.mp4", "actor_id": "03"},
            "video_real_03.mp4": {"actor_id": "03"},
        }

        clusters = splitter.cluster_videos(videos, metadata=metadata)
        assert len(clusters) == 3

        splits = splitter.split_clusters(clusters, split_ratios=(0.60, 0.20, 0.20), seed=42)
        assert len(splits["train"]) > 0
        assert len(splits["val"]) > 0
        assert len(splits["test"]) > 0

        # Verify zero leakage mathematically
        audit = splitter.verify_zero_leakage(splits, metadata=metadata)
        assert audit["zero_leakage_guaranteed"] is True
        assert len(audit["train_test_source_overlap"]) == 0
        assert len(audit["train_test_identity_overlap"]) == 0

    def test_multi_actor_face_swap_clustering(self):
        """Celeb-DF / FF++ case: Fake video swaps actor 1 onto actor 2."""
        splitter = IdentitySourceSplitter()
        videos = [
            Path("id1_0001.mp4"),
            Path("id2_0001.mp4"),
            Path("id1_id2_0001.mp4"),
            Path("id3_0001.mp4"),
            Path("id3_0002.mp4"),
            Path("id4_0001.mp4"),
        ]

        clusters = splitter.cluster_videos(videos)
        for c_key, c_vids in clusters.items():
            names = [v.name for v in c_vids]
            if "id1_id2_0001.mp4" in names:
                assert "id1_0001.mp4" in names
                assert "id2_0001.mp4" in names

    def test_leakage_detection_raises_error(self):
        """Intentionally inject a fake video into test whose source is in train."""
        splitter = IdentitySourceSplitter()
        splits = {
            "train": [Path("video_real_01.mp4"), Path("video_real_02.mp4")],
            "val": [Path("video_real_03.mp4"), Path("video_fake_03.mp4")],
            # Intentional leak: video_fake_01 in test while video_real_01 is in train!
            "test": [Path("video_fake_01.mp4"), Path("video_fake_02.mp4")],
        }
        metadata = {
            "video_fake_01.mp4": {"original": "video_real_01.mp4", "actor_id": "01"},
            "video_real_01.mp4": {"actor_id": "01"},
            "video_fake_02.mp4": {"original": "video_real_02.mp4", "actor_id": "02"},
            "video_real_02.mp4": {"actor_id": "02"},
        }

        with pytest.raises(DataLeakageError) as exc_info:
            splitter.verify_zero_leakage(splits, metadata=metadata)
        assert "CRITICAL DATA LEAKAGE DETECTED" in str(exc_info.value)


class TestPrimaryIdentityMode:
    """
    Regression tests for the 'identity_primary' split mode.

    Celeb-DF v2's swap graph chains identities together (id0_id16,
    id16_id17, ...). Under strict mode the union-find merges those chains
    into a handful of giant clusters, so a single official test-listed video
    drags every video of that identity into test and the dataset collapses
    into a single split.

    'identity_primary' groups videos by their PRIMARY identity only. The
    donor identity of a swap is recorded and reported as an advisory
    residual instead of silently merging the groups.
    """

    def _celeb_chain(self):
        """A minimal Celeb-DF-style swap chain: id1 donates to id0, id2 to id1."""
        return [
            Path("id1_0001.mp4"),   # real, identity id1
            Path("id2_0001.mp4"),   # real, identity id2
            Path("id0_0001.mp4"),   # real, identity id0
            Path("id0_id1_0001.mp4"),  # swap: id0 is base, id1 is donor
            Path("id1_id2_0001.mp4"),  # swap: id1 is base, id2 is donor
        ]

    def test_strict_mode_merges_swap_partners(self):
        """Default (strict) mode must keep the existing chaining behaviour."""
        splitter = IdentitySourceSplitter()
        clusters = splitter.cluster_videos(self._celeb_chain())
        # id0, id1 and id2 are all connected through the swap chain.
        assert len(clusters) == 1

    def test_primary_mode_keeps_identities_separate(self):
        """Primary mode must produce one cluster per primary identity."""
        splitter = IdentitySourceSplitter()
        clusters = splitter.cluster_videos(self._celeb_chain(), link_swap_partners=False)
        assert len(clusters) == 3

        for c_vids in clusters.values():
            names = {v.name for v in c_vids}
            if "id0_id1_0001.mp4" in names:
                # The swap stays with its PRIMARY identity (id0) only.
                assert "id0_0001.mp4" in names
                assert "id1_0001.mp4" not in names

    def test_primary_mode_audit_passes_and_reports_residual(self):
        """The audit must pass and report donor overlap as advisory."""
        splitter = IdentitySourceSplitter()
        videos = self._celeb_chain()
        clusters = splitter.cluster_videos(videos, link_swap_partners=False)
        splits = splitter.split_clusters(clusters, split_ratios=(0.34, 0.33, 0.33), seed=1)

        audit = splitter.verify_zero_leakage(
            splits, allow_swap_partner_overlap=True
        )
        assert audit["zero_leakage_guaranteed"] is True
        assert audit["swap_partner_overlap_is_advisory"] is True
        # Primary identities never overlap.
        assert audit["train_test_identity_overlap"] == []
        assert audit["train_val_identity_overlap"] == []
        assert audit["val_test_identity_overlap"] == []

    def test_primary_mode_still_raises_on_real_primary_leak(self):
        """
        Primary-identity overlap must still be fatal even in advisory mode -
        only DONOR overlap is downgraded to a warning.
        """
        splitter = IdentitySourceSplitter()
        splits = {
            "train": [Path("id0_0001.mp4")],
            "val":   [Path("id3_0001.mp4")],
            "test":  [Path("id0_0002.mp4")],   # same primary identity as train
        }
        with pytest.raises(DataLeakageError) as exc_info:
            splitter.verify_zero_leakage(splits, allow_swap_partner_overlap=True)
        assert "LEAKAGE DETECTED" in str(exc_info.value)
