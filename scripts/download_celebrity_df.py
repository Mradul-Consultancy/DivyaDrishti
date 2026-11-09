"""
Download and extract Celeb-DF v2 dataset from Kaggle.
Dataset: jessicali9530/celeb-df-v2
"""
import sys
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config
from utils.helpers import setup_logger, ensure_dirs

logger = setup_logger("DownloadCelebDF")


def download_celebdf(output_dir: Path, unzip: bool = True):
    ensure_dirs(output_dir)
    logger.info(f"Downloading Celeb-DF v2 from Kaggle to: {output_dir}")
    cmd = [
        "kaggle", "datasets", "download",
        "-d", "jessicali9530/celeb-df-v2",
        "-p", str(output_dir),
    ]
    if unzip:
        cmd.append("--unzip")

    try:
        subprocess.check_call(cmd)
        logger.info("Celeb-DF v2 downloaded successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download Celeb-DF v2: {e}")
        logger.info("Ensure Kaggle credentials (KAGGLE_USERNAME, KAGGLE_KEY) are configured.")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Celeb-DF v2 from Kaggle")
    parser.add_argument("--output-dir", type=str, default=str(config.RAW_DATA_DIR / "celebdf"))
    parser.add_argument("--no-unzip", action="store_true", help="Do not unzip after download")
    args = parser.parse_args()
    download_celebdf(Path(args.output_dir), unzip=not args.no_unzip)
