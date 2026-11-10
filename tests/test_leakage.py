"""
Zero-Leakage Assertion Script & Test Suite.

Asserts that no video ID, source video, or actor/celebrity identity appears in more than one split
(train, val, test). Fails with an explicit AssertionError if any leakage is detected across splits.

Usage as a standalone script:
    python tests/test_leakage.py [--data-dir data/processed] [--metadata-file data/raw_videos/metadata.json]

Usage via pytest:
    pytest -v tests/test_leakage.py
"""
import sys

# The Windows console uses cp1252, which cannot encode emoji/arrows. Force
# UTF-8 output so this script never crashes after doing its work.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass


import os
import re
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, Set, List, Tuple, Optional, Any
from collections import defaultdict
import pytest

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from preprocessing.split_by_identity import IdentitySourceSplitter


@pytest.fixture
def local_tmp_path():
    """Workspace-local temporary directory fixture to prevent Windows Temp PermissionError."""
    temp_dir = Path("tests/.tmp_leakage_test_dir")
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def extract_video_stem_from_crop(filename: str | Path) -> str:
    """
    Extracts the source video stem from a cropped face image filename.
    Format: '{video_stem}_f{frame_idx:05d}_face{face_idx}.jpg' -> '{video_stem}'
    Falls back to stem if not a video frame crop.
    """
    stem = Path(filename).stem
    # Match pattern: <video_stem>_f<digits>_face<digits>
    clean_stem = re.sub(r"_f\d{4,6}_face\d+$", "", stem, flags=re.IGNORECASE)
    return clean_stem


def scan_splits_and_assert_no_leakage(
    data_dir: Path | str = config.PROCESSED_DATA_DIR,
    metadata_file: Optional[Path | str] = None,
    manifest_file: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    Inspects train, val, and test splits on disk and asserts that NO video ID,
    source video, or actor identity appears in multiple splits.

    Raises:
        AssertionError: If any video ID, source video, or identity crosses split boundaries.
    """
    data_dir = Path(data_dir)
    splits = ["train", "val", "test"]

    # Verify split directories exist
    existing_splits = [s for s in splits if (data_dir / s).exists()]
    if not existing_splits:
        raise FileNotFoundError(f"No split directories found in '{data_dir}'. Expected train/, val/, test/.")

    # Load metadata if available (for original source video and actor_id mapping)
    metadata: Dict[str, Dict[str, Any]] = {}
    meta_candidates = []
    if metadata_file:
        meta_candidates.append(Path(metadata_file))
    meta_candidates.extend([
        Path("data/raw_videos/metadata.json"),
        Path("data/raw_videos/dfdc_metadata.json"),
        data_dir / "metadata.json"
    ])
    for cand in meta_candidates:
        if cand.exists():
            try:
                metadata = IdentitySourceSplitter.parse_dfdc_metadata(cand)
                break
            except Exception:
                pass

    splitter = IdentitySourceSplitter()

    # Track entities per split
    split_video_ids: Dict[str, Set[str]] = defaultdict(set)
    split_sources: Dict[str, Set[str]] = defaultdict(set)
    split_identities: Dict[str, Set[str]] = defaultdict(set)
    split_file_counts: Dict[str, int] = defaultdict(int)

    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    for s_name in existing_splits:
        split_path = data_dir / s_name
        for img_path in split_path.rglob("*"):
            if img_path.suffix.lower() not in image_extensions:
                continue

            split_file_counts[s_name] += 1
            vid_stem = extract_video_stem_from_crop(img_path.name)
            split_video_ids[s_name].add(vid_stem)

            # 1. Source & identity via metadata if mapped
            vid_meta = None
            for key in [vid_stem, f"{vid_stem}.mp4", f"{vid_stem}.avi", img_path.name]:
                if key in metadata:
                    vid_meta = metadata[key]
                    break

            if vid_meta:
                orig = vid_meta.get("original")
                if orig:
                    split_sources[s_name].add(Path(orig).stem.lower())
                else:
                    split_sources[s_name].add(vid_stem.lower())

                actor = vid_meta.get("actor_id") or vid_meta.get("identity")
                if actor:
                    split_identities[s_name].add(str(actor).lower())
            else:
                # 2. Source & identity via heuristic filename key extraction
                k1, k2 = splitter.extract_identity_or_source_key(vid_stem)
                split_sources[s_name].add(k1.lower())
                split_identities[s_name].add(k1.lower())
                if k2:
                    split_sources[s_name].add(k2.lower())
                    split_identities[s_name].add(k2.lower())

    # Check manifest if available
    manifest_audit = None
    manifest_path = Path(manifest_file) if manifest_file else (data_dir / "video_split_manifest.json")
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
            manifest_audit = manifest_data.get("zero_leakage_audit")

    # Audit pairwise overlaps
    split_pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    leakage_failures: List[str] = []

    for s1, s2 in split_pairs:
        if s1 not in existing_splits or s2 not in existing_splits:
            continue

        # Check exact video ID overlap
        video_overlap = split_video_ids[s1] & split_video_ids[s2]
        if video_overlap:
            leakage_failures.append(
                f"[VIDEO ID LEAKAGE] {len(video_overlap)} video ID(s) appear in BOTH '{s1}' and '{s2}':\n"
                f"    {sorted(list(video_overlap))}"
            )

        # Check source video overlap (e.g. fake in test derived from real in train)
        source_overlap = split_sources[s1] & split_sources[s2]
        if source_overlap:
            leakage_failures.append(
                f"[SOURCE VIDEO LEAKAGE] {len(source_overlap)} source key(s) appear in BOTH '{s1}' and '{s2}':\n"
                f"    {sorted(list(source_overlap))}"
            )

        # Check identity/actor overlap
        identity_overlap = split_identities[s1] & split_identities[s2]
        if identity_overlap:
            leakage_failures.append(
                f"[IDENTITY LEAKAGE] {len(identity_overlap)} actor identity(ies) appear in BOTH '{s1}' and '{s2}':\n"
                f"    {sorted(list(identity_overlap))}"
            )

    # Compile audit summary
    summary = {
        "status": "PASSED" if not leakage_failures else "FAILED",
        "splits_evaluated": existing_splits,
        "files_per_split": dict(split_file_counts),
        "unique_video_ids_per_split": {s: len(split_video_ids[s]) for s in existing_splits},
        "unique_identities_per_split": {s: len(split_identities[s]) for s in existing_splits},
        "unique_sources_per_split": {s: len(split_sources[s]) for s in existing_splits},
        "leakage_violations_count": len(leakage_failures),
        "manifest_audit": manifest_audit
    }

    # Print human-readable audit table
    print("\n" + "=" * 70)
    print("DATASET LEAKAGE AUDIT REPORT")
    print("=" * 70)
    print(f"Data Directory: {data_dir}")
    print(f"Splits Evaluated: {existing_splits}")
    print("-" * 70)
    for s in existing_splits:
        print(f"  Split: {s.upper():<5} | Files: {split_file_counts[s]:<4} | "
              f"Videos: {len(split_video_ids[s]):<3} | "
              f"Identities: {len(split_identities[s]):<3} | "
              f"Sources: {len(split_sources[s]):<3}")
    print("-" * 70)

    # Assert no leakage
    if leakage_failures:
        print("❌ LEAKAGE DETECTED! Violations found:")
        for failure_msg in leakage_failures:
            print(f"  * {failure_msg}")
        print("=" * 70 + "\n")
        raise AssertionError(
            f"DATA LEAKAGE ASSERTION FAILED: {len(leakage_failures)} boundary violation(s) detected!\n"
            + "\n".join(leakage_failures)
        )

    print("✅ ZERO-LEAKAGE VERIFIED: No video ID, source, or identity appears in two splits.")
    print("=" * 70 + "\n")
    return summary


# Pytest Test Cases

def test_no_video_id_or_identity_leakage():
    """Asserts that no video ID or identity appears in more than one split on disk."""
    summary = scan_splits_and_assert_no_leakage(data_dir=config.PROCESSED_DATA_DIR)
    assert summary["status"] == "PASSED"
    assert summary["leakage_violations_count"] == 0


def test_manifest_zero_leakage():
    """Asserts that video_split_manifest.json exists and confirms zero leakage."""
    manifest_path = config.PROCESSED_DATA_DIR / "video_split_manifest.json"
    if not manifest_path.exists():
        pytest.skip("video_split_manifest.json not found on disk")

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    audit = data.get("zero_leakage_audit", {})
    assert audit.get("zero_leakage_guaranteed") is True
    assert len(audit.get("train_test_source_overlap", [])) == 0
    assert len(audit.get("train_test_identity_overlap", [])) == 0
    assert len(audit.get("train_val_source_overlap", [])) == 0
    assert len(audit.get("val_test_source_overlap", [])) == 0


def test_intentional_leakage_fails_assertion(local_tmp_path):
    """Simulates dataset split contamination and asserts that an AssertionError is raised."""
    train_dir = local_tmp_path / "train" / "real"
    test_dir = local_tmp_path / "test" / "fake"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    # Intentionally plant the same video ID into both train and test
    (train_dir / "video_001_f00000_face0.jpg").write_text("fake_image_content")
    (test_dir / "video_001_f00005_face0.jpg").write_text("fake_image_content")

    with pytest.raises(AssertionError) as exc_info:
        scan_splits_and_assert_no_leakage(data_dir=local_tmp_path)

    assert "DATA LEAKAGE ASSERTION FAILED" in str(exc_info.value)
    assert "video_001" in str(exc_info.value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assert zero video ID or identity leakage across dataset splits.")
    parser.add_argument("--data-dir", type=str, default=str(config.PROCESSED_DATA_DIR), help="Path to processed dataset root")
    parser.add_argument("--metadata-file", type=str, default=None, help="Path to DFDC metadata.json or dataset metadata")
    parser.add_argument("--manifest-file", type=str, default=None, help="Path to video_split_manifest.json")
    args = parser.parse_args()

    try:
        scan_splits_and_assert_no_leakage(
            data_dir=args.data_dir,
            metadata_file=args.metadata_file,
            manifest_file=args.manifest_file
        )
        sys.exit(0)
    except AssertionError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Execution Error: {e}", file=sys.stderr)
        sys.exit(2)
