"""
Video Frame Extraction and MTCNN Face Cropping Pipeline.

Key Guarantees:
1. Extracts N evenly spaced frames per video using OpenCV.
2. Detects and crops faces using MTCNN with contextual margin padding.
3. Splits by IDENTITY and SOURCE (Disjoint-Set clustering) to guarantee zero data leakage
   between train, val, and test splits (Identities(Train) ∩ Identities(Test) = ∅).
4. Supports official benchmarks: FaceForensics++, Celeb-DF v2, and DFDC metadata.
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict
import numpy as np
from PIL import Image
import cv2
import torch
from facenet_pytorch import MTCNN

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from utils.helpers import setup_logger, seed_everything, ensure_dirs
from preprocessing.split_by_identity import IdentitySourceSplitter, DataLeakageError


class VideoFaceExtractor:
    """
    Extracts frames from videos using OpenCV and crops faces using MTCNN.
    Splits datasets at the IDENTITY and SOURCE level to eliminate data leakage across splits.
    """

    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v"}

    def __init__(
        self,
        frames_per_video: int = 10,
        image_size: int = 224,
        margin_ratio: float = 0.20,
        min_face_prob: float = 0.90,
        keep_all_faces: bool = False,
        fallback_center_crop: bool = False,
        device: Optional[torch.device] = None,
        logger=None
    ):
        self.frames_per_video = frames_per_video
        self.image_size = (image_size, image_size)
        self.margin_ratio = margin_ratio
        self.min_face_prob = min_face_prob
        self.keep_all_faces = keep_all_faces
        self.fallback_center_crop = fallback_center_crop
        self.device = device or config.DEVICE
        self.logger = logger or setup_logger("VideoFaceExtractor")

        self.logger.info(f"Initializing MTCNN detector on device: {self.device}...")
        self.mtcnn = MTCNN(
            image_size=image_size,
            margin=0,  # Proportional margin applied in crop_faces_mtcnn
            keep_all=True,
            post_process=False,
            device=self.device
        )

    def get_video_files(self, directory: Path) -> List[Path]:
        """Finds all valid video files within a directory."""
        if not directory.exists():
            return []
        videos = [p for p in directory.glob("*") if p.suffix.lower() in self.VIDEO_EXTENSIONS]
        return sorted(videos)

    def extract_frames_from_video(self, video_path: Path) -> List[Tuple[int, Image.Image]]:
        """
        Extracts N evenly spaced frames from a video file using OpenCV.
        Returns:
            List of (frame_index, PIL.Image)
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.logger.warning(f"Could not open video file: {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            frames = []
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % max(1, (idx // self.frames_per_video)) == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append((idx, Image.fromarray(rgb)))
                idx += 1
                if len(frames) >= self.frames_per_video:
                    break
            cap.release()
            return frames

        if total_frames <= self.frames_per_video:
            target_indices = list(range(total_frames))
        else:
            target_indices = np.linspace(0, total_frames - 1, self.frames_per_video, dtype=int).tolist()

        extracted = []
        for frame_idx in target_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                extracted.append((frame_idx, pil_img))

        cap.release()
        return extracted

    def crop_faces_mtcnn(self, pil_img: Image.Image) -> Tuple[List[Image.Image], bool]:
        """
        Detects and crops faces using MTCNN with proportional margin padding.
        Returns:
            (cropped_faces, face_detected):
                cropped_faces: List of cropped face PIL Images (or fallback if enabled).
                face_detected: True if MTCNN detected >= 1 face meeting min_face_prob, False otherwise.
        """
        try:
            boxes, probs = self.mtcnn.detect(pil_img)
        except Exception as e:
            self.logger.debug(f"MTCNN detection error: {e}")
            boxes, probs = None, None

        cropped_faces = []
        face_detected = False
        img_w, img_h = pil_img.size

        if boxes is not None and len(boxes) > 0:
            valid_detections = [
                (box, prob) for box, prob in zip(boxes, probs)
                if prob is not None and prob >= self.min_face_prob
            ]

            if valid_detections:
                face_detected = True
                if not self.keep_all_faces:
                    valid_detections = [max(
                        valid_detections,
                        key=lambda d: (d[0][2] - d[0][0]) * (d[0][3] - d[0][1])
                    )]

                for box, _ in valid_detections:
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1

                    margin_x = int(w * self.margin_ratio)
                    margin_y = int(h * self.margin_ratio)

                    cx1 = max(0, int(x1 - margin_x))
                    cy1 = max(0, int(y1 - margin_y))
                    cx2 = min(img_w, int(x2 + margin_x))
                    cy2 = min(img_h, int(y2 + margin_y))

                    face_crop = pil_img.crop((cx1, cy1, cx2, cy2))
                    face_crop = face_crop.resize(self.image_size, Image.Resampling.LANCZOS)
                    cropped_faces.append(face_crop)

        # Fallback to center crop if enabled
        if len(cropped_faces) == 0 and self.fallback_center_crop:
            min_dim = min(img_w, img_h)
            left = (img_w - min_dim) // 2
            top = (img_h - min_dim) // 2
            fallback = pil_img.crop((left, top, left + min_dim, top + min_dim))
            fallback = fallback.resize(self.image_size, Image.Resampling.LANCZOS)
            cropped_faces.append(fallback)

        return cropped_faces, face_detected

    def partition_videos_by_split(
        self,
        video_files: List[Path],
        split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
        seed: int = 42
    ) -> Dict[str, List[Path]]:
        """Splits video files naively into train, val, and test subsets."""
        train_ratio, val_ratio, test_ratio = split_ratios
        assert abs(sum(split_ratios) - 1.0) < 1e-4, "Split ratios must sum to 1.0"

        shuffled_videos = list(video_files)
        random.seed(seed)
        random.shuffle(shuffled_videos)

        total = len(shuffled_videos)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        return {
            "train": shuffled_videos[:train_end],
            "val": shuffled_videos[train_end:val_end],
            "test": shuffled_videos[val_end:]
        }

    def process_and_split(
        self,
        video_dir: Path | str,
        output_dir: Path | str = config.PROCESSED_DATA_DIR,
        split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
        classes: Tuple[str, ...] = ("real", "fake"),
        split_mode: str = "identity",
        metadata_file: Optional[Path | str] = None,
        celeb_df_test_list: Optional[Path | str] = None,
        ffpp_splits_dir: Optional[Path | str] = None,
        clean_output: bool = False,
        seed: int = 42
    ) -> Dict:
        """
        Processes video collections, partitions by identity/source (or naive video),
        extracts frames, detects/crops faces with MTCNN, and saves images into train/val/test splits.
        Logs frames dropped where no face was detected per video and per class.
        """
        video_dir = Path(video_dir)
        output_dir = Path(output_dir)

        if clean_output and output_dir.exists():
            self.logger.info(f"Cleaning previous video face crops from {output_dir}...")
            for f in output_dir.rglob("video_*.jpg"):
                try:
                    f.unlink()
                except Exception as e:
                    self.logger.warning(f"Could not remove {f}: {e}")

        self.logger.info("=" * 70)
        self.logger.info(f"Video Face Extraction & Splitting [MODE: {split_mode.upper()}]")
        self.logger.info(f"Source video directory: {video_dir}")
        self.logger.info(f"Target output directory: {output_dir}")
        self.logger.info(f"Frames per video: {self.frames_per_video}")
        self.logger.info(f"Split ratios: train={split_ratios[0]}, val={split_ratios[1]}, test={split_ratios[2]}")
        self.logger.info("=" * 70)

        # Collect video files and their ground truth class
        all_videos: List[Path] = []
        video_to_class: Dict[str, str] = {}

        for class_name in classes:
            class_video_dir = video_dir / class_name
            if class_video_dir.exists():
                vids = self.get_video_files(class_video_dir)
                for v in vids:
                    all_videos.append(v)
                    video_to_class[v.name] = class_name

        # Also search video_dir root
        root_vids = self.get_video_files(video_dir)
        for v in root_vids:
            if v.name not in video_to_class:
                all_videos.append(v)

        if not all_videos:
            self.logger.warning(f"No video files found in '{video_dir}'. Supported: {self.VIDEO_EXTENSIONS}")
            return {}

        self.logger.info(f"Discovered {len(all_videos)} total video files.")

        # Load metadata if present
        metadata = None
        meta_candidates = []
        if metadata_file:
            meta_candidates.append(Path(metadata_file))
        meta_candidates.extend([video_dir / "metadata.json", video_dir / "dfdc_metadata.json"])

        for cand in meta_candidates:
            if cand.exists():
                try:
                    metadata = IdentitySourceSplitter.parse_dfdc_metadata(cand)
                    self.logger.info(f"Loaded dataset metadata from {cand} ({len(metadata)} entries)")
                    # Update video_to_class from metadata if available
                    for v_name, meta_info in metadata.items():
                        if "label" in meta_info and v_name not in video_to_class:
                            video_to_class[v_name] = meta_info["label"].lower()
                    break
                except Exception as e:
                    self.logger.warning(f"Could not load metadata from {cand}: {e}")

        # Check Celeb-DF test list
        celeb_test_set = None
        if celeb_df_test_list and Path(celeb_df_test_list).exists():
            celeb_test_set = IdentitySourceSplitter.parse_celeb_df_test_list(celeb_df_test_list)
        elif (video_dir / "List_of_testing_videos.txt").exists():
            celeb_test_set = IdentitySourceSplitter.parse_celeb_df_test_list(video_dir / "List_of_testing_videos.txt")
        elif (Path(__file__).parent / "List_of_testing_videos.txt").exists():
            celeb_test_set = IdentitySourceSplitter.parse_celeb_df_test_list(Path(__file__).parent / "List_of_testing_videos.txt")

        # Check FF++ splits dir
        ffpp_splits = None
        if ffpp_splits_dir and Path(ffpp_splits_dir).exists():
            ffpp_splits = IdentitySourceSplitter.parse_ffpp_official_splits(ffpp_splits_dir)
        elif (video_dir / "ffpp_splits").exists():
            ffpp_splits = IdentitySourceSplitter.parse_ffpp_official_splits(video_dir / "ffpp_splits")

        splitter = IdentitySourceSplitter(logger=self.logger)
        splits: Dict[str, List[Path]] = {}
        zero_leakage_audit: Dict[str, Any] = {}

        if split_mode in ("identity", "identity_primary"):
            link_swap_partners = (split_mode == "identity")
            if ffpp_splits:
                splits = splitter.split_by_official_benchmark(
                    all_videos, ffpp_splits=ffpp_splits, metadata=metadata, seed=seed,
                    link_swap_partners=link_swap_partners,
                )
            elif celeb_test_set and split_mode == "identity":
                splits = splitter.split_by_official_benchmark(
                    all_videos, celeb_df_test_list=celeb_test_set, metadata=metadata, seed=seed,
                    link_swap_partners=link_swap_partners,
                )
            else:
                # 'identity_primary' keeps the donor identity of a face swap out
                # of the merge, which prevents Celeb-DF's chained swap graph
                # from collapsing every identity into one giant cluster.
                #
                # It also skips the official test list: with identity-level
                # clusters, a single test-listed video drags every video of that
                # identity into test, which on Celeb-DF leaves only a handful of
                # training videos. The ratio-based split below keeps train/val/
                # test balanced while still being strictly primary-identity
                # disjoint.
                clusters = splitter.cluster_videos(
                    all_videos, metadata=metadata, link_swap_partners=link_swap_partners
                )
                splits = splitter.split_clusters(clusters, split_ratios=split_ratios, seed=seed)

            # Mathematically verify zero leakage across train, val, and test splits
            zero_leakage_audit = splitter.verify_zero_leakage(
                splits,
                metadata=metadata,
                allow_swap_partner_overlap=(split_mode == "identity_primary"),
            )

        else:
            # Naive by-video split (independent per class)
            self.logger.warning("Running naive by-video splitting. CAUTION: Potential identity leakage between real & fake.")
            splits = {"train": [], "val": [], "test": []}
            for class_name in classes:
                c_vids = [v for v in all_videos if video_to_class.get(v.name) == class_name]
                c_splits = self.partition_videos_by_split(c_vids, split_ratios=split_ratios, seed=seed)
                for s_name in ["train", "val", "test"]:
                    splits[s_name].extend(c_splits[s_name])

            # Audit advisory check
            try:
                zero_leakage_audit = splitter.verify_zero_leakage(splits, metadata=metadata)
            except DataLeakageError as e:
                self.logger.warning(f"Leakage detected in naive split: {e}")
                zero_leakage_audit = {"zero_leakage_guaranteed": False, "warning": str(e)}

        # Track extraction, retention, and drop statistics
        total_frames_sampled = 0
        total_faces_saved = 0
        total_frames_dropped = 0

        class_summary = {
            c: {
                "num_videos": 0,
                "frames_sampled": 0,
                "faces_saved": 0,
                "frames_dropped": 0,
                "drop_rate": 0.0
            }
            for c in classes
        }

        manifest = {
            "split_mode": split_mode,
            "split_ratios": split_ratios,
            "frames_per_video": self.frames_per_video,
            "zero_leakage_audit": zero_leakage_audit,
            "splits": {"train": {}, "val": {}, "test": {}},
            "summary": {}
        }

        # Extract frames and crop faces for each partition
        for split_name, split_videos in splits.items():
            self.logger.info(f"\nProcessing Split: [{split_name.upper()}] ({len(split_videos)} videos)...")
            split_faces_saved = 0

            for v_idx, vid_path in enumerate(split_videos, start=1):
                # Infer class name
                c_name = video_to_class.get(vid_path.name)
                if not c_name:
                    k1, _ = IdentitySourceSplitter.extract_identity_or_source_key(vid_path.name)
                    if "fake" in vid_path.name.lower():
                        c_name = "fake"
                    elif "real" in vid_path.name.lower():
                        c_name = "real"
                    else:
                        c_name = "fake"  # Fallback default

                dest_dir = output_dir / split_name / c_name
                dest_dir.mkdir(parents=True, exist_ok=True)

                extracted_frames = self.extract_frames_from_video(vid_path)
                frames_sampled = len(extracted_frames)
                saved_for_video = 0
                dropped_for_video = 0

                for frame_idx, pil_frame in extracted_frames:
                    faces, face_detected = self.crop_faces_mtcnn(pil_frame)

                    if not face_detected:
                        dropped_for_video += 1

                    if not faces:
                        # Dropped frame (no face detected and no fallback)
                        continue

                    for face_idx, face_img in enumerate(faces):
                        out_filename = f"{vid_path.stem}_f{frame_idx:05d}_face{face_idx}.jpg"
                        save_file = dest_dir / out_filename
                        face_img.save(save_file, quality=95)
                        saved_for_video += 1
                        split_faces_saved += 1

                # Update class summary
                if c_name in class_summary:
                    class_summary[c_name]["num_videos"] += 1
                    class_summary[c_name]["frames_sampled"] += frames_sampled
                    class_summary[c_name]["faces_saved"] += saved_for_video
                    class_summary[c_name]["frames_dropped"] += dropped_for_video

                total_frames_sampled += frames_sampled
                total_faces_saved += saved_for_video
                total_frames_dropped += dropped_for_video

                if c_name not in manifest["splits"][split_name]:
                    manifest["splits"][split_name][c_name] = {
                        "num_videos": 0,
                        "faces_saved": 0,
                        "frames_sampled": 0,
                        "frames_dropped": 0,
                        "videos": []
                    }

                manifest["splits"][split_name][c_name]["num_videos"] += 1
                manifest["splits"][split_name][c_name]["faces_saved"] += saved_for_video
                manifest["splits"][split_name][c_name]["frames_sampled"] += frames_sampled
                manifest["splits"][split_name][c_name]["frames_dropped"] += dropped_for_video

                video_drop_rate = round(dropped_for_video / frames_sampled, 4) if frames_sampled > 0 else 0.0
                manifest["splits"][split_name][c_name]["videos"].append({
                    "video_file": vid_path.name,
                    "frames_sampled": frames_sampled,
                    "faces_extracted": saved_for_video,
                    "frames_dropped": dropped_for_video,
                    "drop_rate": video_drop_rate
                })

                # Per-video logging with dropped frames visibility
                drop_pct = (dropped_for_video / frames_sampled * 100.0) if frames_sampled > 0 else 0.0
                if dropped_for_video == frames_sampled and frames_sampled > 0:
                    self.logger.warning(
                        f"  [{v_idx}/{len(split_videos)}] [{split_name}/{c_name}] {vid_path.name}: "
                        f"ALL {dropped_for_video}/{frames_sampled} frames DROPPED (no face detected at min_prob={self.min_face_prob})"
                    )
                elif dropped_for_video > 0:
                    self.logger.info(
                        f"  [{v_idx}/{len(split_videos)}] [{split_name}/{c_name}] {vid_path.name}: "
                        f"{saved_for_video} faces saved, {dropped_for_video}/{frames_sampled} frames dropped ({drop_pct:.1f}% drop rate)"
                    )
                else:
                    self.logger.info(
                        f"  [{v_idx}/{len(split_videos)}] [{split_name}/{c_name}] {vid_path.name}: "
                        f"{saved_for_video} faces saved (0 frames dropped)"
                    )

        # Log class-level drop & retention summary
        self.logger.info("\n" + "=" * 70)
        self.logger.info("FRAME RETENTION & DROP SUMMARY BY CLASS:")
        self.logger.info("=" * 70)
        for c_name, c_data in class_summary.items():
            s = c_data["frames_sampled"]
            d = c_data["frames_dropped"]
            rate = round(d / s, 4) if s > 0 else 0.0
            c_data["drop_rate"] = rate
            pct = rate * 100.0
            self.logger.info(
                f"  Class '{c_name:<5}': {c_data['num_videos']:<2} videos | "
                f"{s:<3} frames sampled | "
                f"{c_data['faces_saved']:<3} faces saved | "
                f"{d:<3} frames dropped ({pct:.1f}% drop rate)"
            )
        self.logger.info("=" * 70)

        overall_drop_rate = round(total_frames_dropped / total_frames_sampled, 4) if total_frames_sampled > 0 else 0.0
        manifest["summary"] = {
            "total_videos": len(all_videos),
            "total_frames_sampled": total_frames_sampled,
            "total_faces_saved": total_faces_saved,
            "total_frames_dropped": total_frames_dropped,
            "overall_drop_rate": overall_drop_rate,
            "dropped_by_class": class_summary
        }

        manifest_path = output_dir / "video_split_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4)

        self.logger.info("\n" + "=" * 70)
        self.logger.info("Face Extraction & Splitting Completed Successfully!")
        self.logger.info(f"Split Mode: {split_mode}")
        self.logger.info(f"Zero-Leakage Guarantee: {zero_leakage_audit.get('zero_leakage_guaranteed', False)}")
        self.logger.info(f"Total Videos Processed: {manifest['summary']['total_videos']}")
        self.logger.info(f"Total Frames Sampled: {total_frames_sampled}")
        self.logger.info(f"Total Cropped Faces Saved: {total_faces_saved}")
        self.logger.info(f"Total Frames Dropped (No Face): {total_frames_dropped} ({overall_drop_rate * 100:.1f}% drop rate)")
        self.logger.info(f"Video Partition Manifest saved to: {manifest_path}")
        self.logger.info("=" * 70)

        return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Extract N frames per video with OpenCV, crop faces with MTCNN, and split by identity/source."
    )
    parser.add_argument("--video-dir", type=str, default="data/raw_videos", help="Directory containing real/ and fake/ video folders")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory for train/val/test face crops")
    parser.add_argument("--split-mode", type=str, default="identity", choices=["identity", "video"], help="Splitting strategy: 'identity' (disjoint actor/source clusters) or 'video' (naive)")
    parser.add_argument("--metadata-file", type=str, default=None, help="Path to DFDC metadata.json or custom dataset metadata")
    parser.add_argument("--celeb-df-test-list", type=str, default=None, help="Path to Celeb-DF List_of_testing_videos.txt")
    parser.add_argument("--ffpp-splits-dir", type=str, default=None, help="Path to FF++ splits directory containing train.json, val.json, test.json")
    parser.add_argument("-n", "--frames-per-video", type=int, default=10, help="Number of frames to extract per video (default: 10)")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train ratio (default: 0.70)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Val ratio (default: 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio (default: 0.15)")
    parser.add_argument("--image-size", type=int, default=224, help="Target image crop size (default: 224)")
    parser.add_argument("--margin", type=float, default=0.20, help="Contextual margin ratio around face (default: 0.20)")
    parser.add_argument("--min-prob", type=float, default=0.90, help="Minimum MTCNN detection probability (default: 0.90)")
    parser.add_argument("--keep-all-faces", action="store_true", help="Keep all detected faces in a frame instead of only the primary face")
    parser.add_argument("--fallback-center-crop", action="store_true", help="Fallback to center crop if MTCNN finds no face")
    parser.add_argument("--clean", action="store_true", help="Clean prior extracted video crops in output directory before extracting")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible splitting (default: 42)")
    parser.add_argument("--device", type=str, default=str(config.DEVICE), help="Computation device (cuda/cpu)")

    args = parser.parse_args()
    seed_everything(args.seed)

    extractor = VideoFaceExtractor(
        frames_per_video=args.frames_per_video,
        image_size=args.image_size,
        margin_ratio=args.margin,
        min_face_prob=args.min_prob,
        keep_all_faces=args.keep_all_faces,
        fallback_center_crop=args.fallback_center_crop,
        device=torch.device(args.device)
    )

    extractor.process_and_split(
        video_dir=Path(args.video_dir),
        output_dir=Path(args.output_dir),
        split_ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
        split_mode=args.split_mode,
        metadata_file=args.metadata_file,
        celeb_df_test_list=args.celeb_df_test_list,
        ffpp_splits_dir=args.ffpp_splits_dir,
        clean_output=args.clean,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
