"""
DFDC (Deepfake Detection Challenge) Kaggle Downloader and Organizer.

Automates:
1. Kaggle API credential checking (supports both legacy username/key and new KGAT_ tokens).
2. Downloading DFDC dataset videos directly via Kaggle API.
3. Unpacking and organizing into the project structure:
   data/raw_videos/
     ├── real/
     ├── fake/
     └── metadata.json
4. Ensuring that for any extracted fake video, its original real source video
   is also preserved to maintain zero-leakage identity clustering.
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
import sys
import json
import zipfile
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Set, List

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from utils.helpers import setup_logger, ensure_dirs


logger = setup_logger("DFDCDownloader")

DATASET_REF = "pranay22077/dfdc-10"
DATASET_SUBDIR = "dfdc_train_part_00/dfdc_train_part_0"
COMPETITION_NAME = "deepfake-detection-challenge"
DEFAULT_PART = "dfdc_train_part_00.zip"


def check_kaggle_auth() -> bool:
    """
    Checks whether Kaggle credentials or environment variables are present.
    """
    user_home = Path.home()
    kaggle_dir = user_home / ".kaggle"
    token_file = kaggle_dir / "kaggle.json"
    access_token = kaggle_dir / "access_token"

    has_files = token_file.exists() or access_token.exists()
    has_env = (
        bool(os.environ.get("KAGGLE_API_TOKEN"))
        or (bool(os.environ.get("KAGGLE_USERNAME")) and bool(os.environ.get("KAGGLE_KEY")))
    )

    return has_files or has_env


def print_auth_instructions():
    """Prints user-friendly instructions for setting up Kaggle credentials."""
    print(
        "\n" + "=" * 65 + "\n"
        "  Kaggle credentials not found.\n\n"
        "  Option A — environment variables (recommended for CI/Kaggle kernels):\n"
        "    export KAGGLE_USERNAME=<your_username>\n"
        "    export KAGGLE_KEY=<your_api_key>\n\n"
        "  Option B — kaggle.json token file:\n"
        "    1. Go to https://www.kaggle.com/settings → 'Create New Token'\n"
        "    2. Place the downloaded kaggle.json in  ~/.kaggle/kaggle.json\n"
        "    3. Run:  chmod 600 ~/.kaggle/kaggle.json\n"
        + "=" * 65 + "\n"
    )


def download_dfdc_part(
    part: str,
    download_dir: Path,
    competition: str = COMPETITION_NAME,
) -> Path:
    """
    Downloads a single DFDC competition archive part via the Kaggle CLI.

    Args:
        part:         Filename of the archive part (e.g. ``dfdc_train_part_00.zip``).
        download_dir: Directory to save the downloaded file.
        competition:  Kaggle competition slug (default: deepfake-detection-challenge).

    Returns:
        Path to the downloaded archive file.

    Raises:
        RuntimeError: If the kaggle CLI returns a non-zero exit code.
    """
    ensure_dirs(download_dir)
    archive_path = download_dir / part
    if archive_path.exists():
        logger.info(f"Archive already cached at {archive_path}. Skipping download.")
        return archive_path

    logger.info(f"Downloading '{part}' from competition '{competition}' …")
    cmd = [
        sys.executable, "-m", "kaggle", "competitions", "download",
        "-c", competition,
        "-f", part,
        "-p", str(download_dir),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"kaggle CLI failed (exit {res.returncode}):\n"
            f"  stdout: {res.stdout.strip()}\n"
            f"  stderr: {res.stderr.strip()}"
        )
    if not archive_path.exists():
        raise FileNotFoundError(
            f"Download reported success but archive not found at {archive_path}"
        )
    logger.info(f"Downloaded {archive_path.stat().st_size / 1e6:.1f} MB → {archive_path}")
    return archive_path


def select_paired_subset(meta: Dict[str, Dict[str, Any]], target_count: int) -> Set[str]:
    """
    Selects target_count videos while strictly ensuring that for every selected
    FAKE video, its pristine REAL original video is also included.
    """
    selected: Set[str] = set()

    # 1. Add fake-real pairs first to maintain identity clusters
    for vid_name, info in meta.items():
        if len(selected) >= target_count:
            break
        if str(info.get("label", "")).upper() == "FAKE" and info.get("original"):
            orig = info["original"]
            if orig in meta:
                selected.add(vid_name)
                selected.add(orig)

    # 2. If additional videos are needed to reach target_count, add more real/fake videos
    for vid_name, info in meta.items():
        if len(selected) >= target_count:
            break
        selected.add(vid_name)
        orig = info.get("original")
        if orig and orig in meta:
            selected.add(orig)

    return selected


def download_dfdc_from_dataset(
    output_dir: Path,
    download_dir: Path,
    max_videos: Optional[int] = 50
) -> Dict[str, Any]:
    """
    Downloads videos from the public DFDC dataset (pranay22077/dfdc-10) using Kaggle API.
    If max_videos is set, only downloads the required paired videos to conserve bandwidth.
    """
    ensure_dirs(download_dir, output_dir)
    real_dir = output_dir / "real"
    fake_dir = output_dir / "fake"
    ensure_dirs(real_dir, fake_dir)

    # Step 1: Download metadata.json
    meta_rel_path = f"{DATASET_SUBDIR}/metadata.json"
    local_meta_dir = download_dir / Path(DATASET_SUBDIR)
    local_meta_path = local_meta_dir / "metadata.json"

    if not local_meta_path.exists():
        logger.info(f"Fetching DFDC metadata from Kaggle dataset '{DATASET_REF}'...")
        cmd = [
            sys.executable, "-m", "kaggle", "datasets", "download",
            "-d", DATASET_REF,
            "-f", meta_rel_path,
            "-p", str(download_dir)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            logger.error(f"Failed to fetch metadata: {res.stderr or res.stdout}")
            raise RuntimeError(f"Kaggle download failed: {res.stderr or res.stdout}")

    with open(local_meta_path, "r", encoding="utf-8") as f:
        full_metadata = json.load(f)

    logger.info(f"Loaded DFDC part_00 metadata containing {len(full_metadata)} videos.")

    # Step 2: Select videos to download
    if max_videos:
        selected_vids = select_paired_subset(full_metadata, max_videos)
        logger.info(
            f"Selected balanced subset of {len(selected_vids)} videos (guaranteeing all original-fake pairs)."
        )
    else:
        selected_vids = set(full_metadata.keys())
        logger.info(f"Targeting all {len(selected_vids)} videos in part_00.")

    # Step 3: Download and sort each video
    metadata_path = output_dir / "metadata.json"
    existing_metadata: Dict[str, Any] = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                existing_metadata = json.load(f)
        except Exception:
            existing_metadata = {}

    downloaded_counts = {"real": 0, "fake": 0}
    total_to_fetch = len(selected_vids)

    logger.info(f"Downloading and organizing {total_to_fetch} videos into '{output_dir}'...")

    for i, vid_name in enumerate(selected_vids, 1):
        info = full_metadata[vid_name]
        label = str(info.get("label", "")).upper()
        target_folder = real_dir if label == "REAL" else fake_dir
        dest_video = target_folder / vid_name

        if dest_video.exists():
            downloaded_counts[label.lower()] += 1
            existing_metadata[vid_name] = info
            continue

        cached_video = local_meta_dir / vid_name
        if not cached_video.exists():
            file_rel_path = f"{DATASET_SUBDIR}/{vid_name}"
            cmd = [
                sys.executable, "-m", "kaggle", "datasets", "download",
                "-d", DATASET_REF,
                "-f", file_rel_path,
                "-p", str(download_dir)
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                logger.warning(f"Could not download {vid_name}: {res.stderr.strip() or res.stdout.strip()}")
                continue

        if cached_video.exists():
            shutil.copy2(cached_video, dest_video)
            downloaded_counts[label.lower()] += 1
            existing_metadata[vid_name] = info
            if i % 5 == 0 or i == total_to_fetch:
                logger.info(f"Progress: [{i}/{total_to_fetch}] videos acquired.")

    # Write merged metadata.json
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(existing_metadata, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"DFDC Acquisition Complete!")
    logger.info(f"Total REAL videos in raw_videos/real: {len(list(real_dir.glob('*.mp4')))}")
    logger.info(f"Total FAKE videos in raw_videos/fake: {len(list(fake_dir.glob('*.mp4')))}")
    logger.info(f"Metadata saved to: {metadata_path}")
    logger.info("=" * 60)

    return existing_metadata


def organize_dfdc_subset(
    archive_or_dir: Path,
    output_dir: Path,
    max_videos: Optional[int] = None
) -> Dict[str, Any]:
    """
    Extracts videos and organizes them into output_dir/real and output_dir/fake,
    writing an updated metadata.json.
    """
    output_dir = Path(output_dir)
    real_dir = output_dir / "real"
    fake_dir = output_dir / "fake"
    ensure_dirs(real_dir, fake_dir)

    metadata_path = output_dir / "metadata.json"
    existing_metadata: Dict[str, Any] = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                existing_metadata = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read existing metadata.json: {e}")

    temp_extract_dir = output_dir / "_temp_extract"

    try:
        # Step 1: Extract archive if zip
        if archive_or_dir.is_file() and archive_or_dir.suffix.lower() == ".zip":
            logger.info(f"Extracting '{archive_or_dir.name}' to temporary directory...")
            with zipfile.ZipFile(archive_or_dir, "r") as zf:
                meta_names = [n for n in zf.namelist() if n.endswith("metadata.json")]
                if not meta_names:
                    raise FileNotFoundError("No metadata.json found inside DFDC archive.")

                zf.extract(meta_names[0], temp_extract_dir)
                source_meta_file = temp_extract_dir / meta_names[0]

                with open(source_meta_file, "r", encoding="utf-8") as f:
                    part_metadata = json.load(f)

                if max_videos:
                    selected_vids = select_paired_subset(part_metadata, max_videos)
                else:
                    selected_vids = set(part_metadata.keys())

                logger.info(f"Extracting {len(selected_vids)} videos from archive...")

                for item in zf.namelist():
                    base_name = Path(item).name
                    if base_name in selected_vids:
                        zf.extract(item, temp_extract_dir)

            search_dir = temp_extract_dir
        else:
            search_dir = archive_or_dir
            meta_candidates = list(search_dir.glob("**/metadata.json"))
            if not meta_candidates:
                raise FileNotFoundError(f"No metadata.json found in {search_dir}")
            source_meta_file = meta_candidates[0]
            with open(source_meta_file, "r", encoding="utf-8") as f:
                part_metadata = json.load(f)

        # Step 2: Organize videos into real and fake folders
        video_files = list(search_dir.glob("**/*.mp4"))
        logger.info(f"Found {len(video_files)} video files to process.")

        extracted_count = {"real": 0, "fake": 0}
        for vid_file in video_files:
            vid_name = vid_file.name
            meta = part_metadata.get(vid_name)
            if not meta:
                continue

            label = str(meta.get("label", "")).upper()
            if label == "REAL":
                target_folder = real_dir
                extracted_count["real"] += 1
            elif label == "FAKE":
                target_folder = fake_dir
                extracted_count["fake"] += 1
            else:
                continue

            target_path = target_folder / vid_name
            shutil.copy2(vid_file, target_path)
            existing_metadata[vid_name] = meta

        # Step 3: Write merged metadata.json
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(existing_metadata, f, indent=2)

        logger.info(
            f"Organization complete! Placed {extracted_count['real']} REAL and "
            f"{extracted_count['fake']} FAKE videos in '{output_dir}'.\n"
            f"Metadata updated at '{metadata_path}'."
        )

        return existing_metadata

    finally:
        if temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Download and organize a DFDC subset from Kaggle into data/raw_videos"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="dataset",
        choices=["dataset", "competition"],
        help="Source on Kaggle: 'dataset' (instant, no competition rules required) or 'competition'"
    )
    parser.add_argument(
        "--part",
        type=str,
        default=DEFAULT_PART,
        help=f"DFDC archive part to download if using competition (default: '{DEFAULT_PART}')"
    )
    parser.add_argument(
        "--archive-path",
        type=str,
        default=None,
        help="Optional local path to an already downloaded DFDC zip file"
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default=str(config.RAW_DATA_DIR / "downloads"),
        help="Directory to store downloaded Kaggle files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(config.BASE_DIR / "data" / "raw_videos"),
        help="Target raw_videos directory for extracted real/ and fake/ files"
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=50,
        help="Number of videos to download and organize (preserves pairs, default: 50)"
    )

    args = parser.parse_args()

    # If local archive is provided, unpack and organize
    if args.archive_path:
        local_archive = Path(args.archive_path)
        if not local_archive.exists():
            logger.error(f"Provided archive does not exist: {local_archive}")
            sys.exit(1)
        organize_dfdc_subset(local_archive, Path(args.output_dir), max_videos=args.max_videos)
        return

    # Check authentication
    if not check_kaggle_auth():
        logger.error("Kaggle credentials not found in ~/.kaggle or environment variables.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    download_dir = Path(args.download_dir)

    if args.source == "dataset":
        download_dfdc_from_dataset(output_dir, download_dir, max_videos=args.max_videos)
    else:
        # Fallback to competition download
        cmd = [
            sys.executable, "-m", "kaggle", "competitions", "download",
            "-c", COMPETITION_NAME,
            "-f", args.part,
            "-p", str(download_dir)
        ]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            logger.warning("Competition download failed. Falling back to Kaggle dataset source...")
            download_dfdc_from_dataset(output_dir, download_dir, max_videos=args.max_videos)
        else:
            archive_file = download_dir / args.part
            organize_dfdc_subset(archive_file, output_dir, max_videos=args.max_videos)


if __name__ == "__main__":
    main()
