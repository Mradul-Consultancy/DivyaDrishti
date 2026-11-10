"""
Generate dedicated Kaggle notebook for Celeb-DF full training.
"""
import json
from pathlib import Path

nb = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"tags": ["parameters"]},
            "outputs": [],
            "source": [
                "# Parameters\n",
                "EPOCHS = 25\n",
                "BATCH_SIZE = 32\n",
                "SKIP_YOLO = False\n",
                "FRAMES_PER_VIDEO = 10\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Celeb-DF v2 Training Pipeline (Kaggle GPU)\n",
                "\n",
                "Trains **EfficientNet-B3**, **ConvNeXt-Tiny**, and **YOLO26n-cls** on the official Celeb-DF v2 benchmark.\n",
                "\n",
                "**Inputs**:\n",
                "- Dataset: `reubensuju/celeb-df-v2` (mounted at `/kaggle/input/celeb-df-v2`)\n",
                "- Code: `mradulmanimishra/deepfake-code` (mounted at `/kaggle/input/deepfake-code`)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ── Install dependencies ──────────────────────────────────────\n",
                "!pip install -q timm==1.0.12 facenet-pytorch==2.6.0 ultralytics==8.3.40\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ── Mount & Extract Codebase ──────────────────────────────────\n",
                "import os, sys, zipfile\n",
                "from pathlib import Path\n",
                "\n",
                "REPO_DIR = Path('/kaggle/working')\n",
                "code_zip = Path('/kaggle/input/deepfake-code/deepfake_code.zip')\n",
                "if code_zip.exists():\n",
                "    with zipfile.ZipFile(code_zip, 'r') as zf:\n",
                "        zf.extractall(str(REPO_DIR))\n",
                "    print('✅ Code unpacked from deepfake-code dataset.')\n",
                "\n",
                "sys.path.insert(0, str(REPO_DIR))\n",
                "os.chdir(str(REPO_DIR))\n",
                "print(f'Working directory: {Path.cwd()}')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ── Verify GPU ────────────────────────────────────────────────\n",
                "import torch\n",
                "print(f'CUDA available: {torch.cuda.is_available()}')\n",
                "if torch.cuda.is_available():\n",
                "    print(f'GPU Device: {torch.cuda.get_device_name(0)}')\n",
                "else:\n",
                "    print('⚠️ Running on CPU')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ── Extract Celeb-DF Faces ─────────────────────────────────────\n",
                "import config\n",
                "from preprocessing.extract_faces_from_videos import VideoFaceExtractor\n",
                "from utils.helpers import setup_logger, ensure_dirs\n",
                "\n",
                "celeb_input = Path('/kaggle/input/celeb-df-v2')\n",
                "if not celeb_input.exists():\n",
                "    # Fallback to subdirectory if nested\n",
                "    candidates = list(Path('/kaggle/input').glob('*celeb*'))\n",
                "    if candidates:\n",
                "        celeb_input = candidates[0]\n",
                "\n",
                "print(f'Celeb-DF source: {celeb_input}')\n",
                "out_dir = REPO_DIR / 'data' / 'processed' / 'celebdf'\n",
                "ensure_dirs(out_dir)\n",
                "\n",
                "logger = setup_logger('CelebDF_Extract')\n",
                "extractor = VideoFaceExtractor(\n",
                "    frames_per_video     = int(FRAMES_PER_VIDEO),\n",
                "    image_size           = 224,\n",
                "    margin_ratio         = 0.20,\n",
                "    min_face_prob        = 0.90,\n",
                "    fallback_center_crop = True,\n",
                "    device               = config.DEVICE,\n",
                "    logger               = logger\n",
                ")\n",
                "\n",
                "extractor.process_and_split(\n",
                "    video_dir          = celeb_input,\n",
                "    output_dir         = out_dir,\n",
                "    split_mode         = 'celeb_df',\n",
                "    clean_output       = False,\n",
                "    seed               = 42\n",
                ")\n",
                "print('✅ Face extraction and official benchmark split complete.')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ── Train All 3 Models on Celeb-DF ─────────────────────────────\n",
                "import types\n",
                "from scripts.train_all_datasets import run_all\n",
                "\n",
                "train_args = types.SimpleNamespace(\n",
                "    datasets   = ['celebdf'],\n",
                "    epochs     = int(EPOCHS),\n",
                "    batch_size = int(BATCH_SIZE),\n",
                "    patience   = 7,\n",
                "    pilot      = False,\n",
                "    skip_yolo  = bool(SKIP_YOLO),\n",
                "    force      = True,\n",
                ")\n",
                "\n",
                "print(f'Starting Celeb-DF training grid for {EPOCHS} epochs...')\n",
                "run_all(train_args)\n",
                "print('✅ Celeb-DF models trained successfully.')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ── Evaluate Ensemble on Celeb-DF Test Split ───────────────────\n",
                "import subprocess\n",
                "subprocess.check_call(['python', 'main.py', 'evaluate-ensemble', '--dataset', 'celebdf'])\n",
                "subprocess.check_call(['python', 'main.py', 'generate-reports', '--allow-toy'])\n",
                "print('✅ Evaluation complete.')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ── Package Checkpoints & Reports ──────────────────────────────\n",
                "import shutil\n",
                "\n",
                "ckpt_archive = REPO_DIR / 'celebdf_checkpoints'\n",
                "shutil.make_archive(str(ckpt_archive), 'zip', root_dir=str(REPO_DIR / 'models'), base_dir='checkpoints')\n",
                "\n",
                "reports_archive = REPO_DIR / 'celebdf_reports'\n",
                "shutil.make_archive(str(reports_archive), 'zip', root_dir=str(REPO_DIR), base_dir='reports')\n",
                "\n",
                "print(f'✅ Checkpoints archived → {ckpt_archive}.zip')\n",
                "print(f'✅ Reports archived     → {reports_archive}.zip')\n"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbformat": 4,
            "nbformat_minor": 4
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = Path('notebooks/kaggle_celebdf_train.ipynb')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f'Generated {out_path}')

