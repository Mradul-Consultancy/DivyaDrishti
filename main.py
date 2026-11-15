"""
DivyaDrishti (दिव्यदृष्टि) AI — Master CLI Entrypoint for Deepfake Image & Video Detection.
Allows running individual stages (preprocess, train, evaluate, predict) or a complete end-to-end pipeline.
"""

import sys
import argparse
from pathlib import Path

import config
from utils.helpers import ensure_dirs
from preprocessing.split_data import split_dataset
from preprocessing.extract_faces_from_videos import VideoFaceExtractor
from training.train import run_training
from evaluation.evaluate import evaluate_model
from evaluation.predict import run_inference
from scripts.generate_sample_data import generate_samples


def main():
    parser = argparse.ArgumentParser(
        description="DivyaDrishti (दिव्यदृष्टि) AI: Deepfake Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 1. Generate sample data to test the pipeline:
  python main.py demo-data

  # 2. Split dataset into train/val/test splits:
  python main.py split --source data/raw --dest data/processed

  # 3. Train a detection model:
  python main.py train --model resnet18 --epochs 10 --batch-size 32

  # 4. Evaluate the trained model:
  python main.py evaluate --checkpoint models/checkpoints/resnet18_best.pth

  # 5. Predict single image:
  python main.py predict --checkpoint models/checkpoints/resnet18_best.pth --image path/to/image.jpg
        """
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline command to execute")

    # Command: demo-data
    p_demo = subparsers.add_parser("demo-data", help="Generate synthetic sample data for pipeline testing")
    p_demo.add_argument("--real", type=int, default=30, help="Number of real images")
    p_demo.add_argument("--fake", type=int, default=30, help="Number of fake images")

    # Command: split
    p_split = subparsers.add_parser("split", help="Split raw images into train/val/test")
    p_split.add_argument("--source", type=str, default=str(config.RAW_DATA_DIR), help="Source directory")
    p_split.add_argument("--dest", type=str, default=str(config.PROCESSED_DATA_DIR), help="Destination directory")
    p_split.add_argument("--train-ratio", type=float, default=0.70)
    p_split.add_argument("--val-ratio", type=float, default=0.15)
    p_split.add_argument("--test-ratio", type=float, default=0.15)

    # Command: train
    p_train = subparsers.add_parser("train", help="Train detection model")
    p_train.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet50", "mesonet", "efficientnet_b0"])
    p_train.add_argument("--data-dir", type=str, default=str(config.PROCESSED_DATA_DIR))
    p_train.add_argument("--checkpoint-dir", type=str, default=str(config.CHECKPOINT_DIR))
    p_train.add_argument("--reports-dir", type=str, default=str(config.REPORTS_DIR))
    p_train.add_argument("--epochs", type=int, default=config.EPOCHS)
    p_train.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p_train.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p_train.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    p_train.add_argument("--patience", type=int, default=config.EARLY_STOPPING_PATIENCE)
    p_train.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--device", type=str, default=str(config.DEVICE))
    p_train.add_argument("--no-pretrained", dest="pretrained", action="store_false")

    # Command: evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate model checkpoint")
    p_eval.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    p_eval.add_argument("--model", type=str, default="resnet18")
    p_eval.add_argument("--data-dir", type=str, default=str(config.PROCESSED_DATA_DIR))
    p_eval.add_argument("--test-dir", type=str, default=None, help="Explicit test folder (overrides --data-dir auto-detection)")
    p_eval.add_argument("--reports-dir", type=str, default=str(config.REPORTS_DIR))
    p_eval.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p_eval.add_argument("--device", type=str, default=str(config.DEVICE))

    # Command: predict
    p_pred = subparsers.add_parser("predict", help="Run inference on single image or folder")
    p_pred.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    p_pred.add_argument("--model", type=str, default="resnet18")
    p_pred.add_argument("--image", type=str, help="Path to single image")
    p_pred.add_argument("--input-dir", type=str, help="Directory of images for batch inference")
    p_pred.add_argument("--output-dir", type=str, help="Directory to save annotated images")
    p_pred.add_argument("--no-crop", action="store_true", help="Skip face extraction")
    p_pred.add_argument("--device", type=str, default=str(config.DEVICE))
    p_pred.add_argument("--threshold", type=float, default=None,
                        help="Decision threshold. Defaults to the calibrated value for the "
                             "checkpoint's domain, else 0.5.")
    p_pred.add_argument("--margin", type=float, default=0.2,
                        help="Contextual margin around the detected face (default: 0.2)")
    p_pred.add_argument("--min-confidence", type=float, default=0.90,
                        help="Minimum face-detector confidence (default: 0.90)")

    # Command: extract-videos
    p_vid = subparsers.add_parser("extract-videos", help="Extract N frames per video, crop faces with MTCNN, and split by identity/source")
    p_vid.add_argument("--video-dir", type=str, default="data/raw_videos", help="Directory with real/ and fake/ video folders")
    p_vid.add_argument("--output-dir", type=str, default=str(config.PROCESSED_DATA_DIR), help="Destination split directory")
    p_vid.add_argument("--split-mode", type=str, default="identity",
                       choices=["identity", "identity_primary", "video"],
                       help="Splitting strategy: 'identity' (disjoint clusters), "
                            "'identity_primary' (primary-identity clusters; donor identities "
                            "of face swaps are reported as advisory residual), or 'video' (naive)")
    p_vid.add_argument("--metadata-file", type=str, default=None, help="Path to DFDC metadata.json or custom dataset metadata")
    p_vid.add_argument("--celeb-df-test-list", type=str, default=None, help="Path to Celeb-DF List_of_testing_videos.txt")
    p_vid.add_argument("--ffpp-splits-dir", type=str, default=None, help="Path to FF++ splits directory containing train.json, val.json, test.json")
    p_vid.add_argument("-n", "--frames-per-video", type=int, default=10, help="Frames per video")
    p_vid.add_argument("--train-ratio", type=float, default=0.70)
    p_vid.add_argument("--val-ratio", type=float, default=0.15)
    p_vid.add_argument("--test-ratio", type=float, default=0.15)
    p_vid.add_argument("--image-size", type=int, default=224)
    p_vid.add_argument("--margin", type=float, default=0.20)
    p_vid.add_argument("--min-prob", type=float, default=0.90)
    p_vid.add_argument("--keep-all-faces", action="store_true")
    p_vid.add_argument("--fallback-center-crop", action="store_true")
    p_vid.add_argument("--clean", action="store_true", help="Clean prior extracted video crops in output directory before extracting")
    p_vid.add_argument("--seed", type=int, default=42)
    p_vid.add_argument("--device", type=str, default=str(config.DEVICE))

    # Command: verify-leakage
    p_leak = subparsers.add_parser("verify-leakage", help="Audit dataset splits or manifest to mathematically verify zero identity leakage")
    p_leak.add_argument("--manifest", type=str, default=str(config.PROCESSED_DATA_DIR / "video_split_manifest.json"), help="Path to video_split_manifest.json")
    p_leak.add_argument("--metadata-file", type=str, default=None, help="Path to DFDC metadata.json or dataset metadata")

    # Command: download-dfdc
    p_dfdc = subparsers.add_parser("download-dfdc", help="Download and organize DFDC video dataset subset from Kaggle")
    p_dfdc.add_argument("--part", type=str, default="dfdc_train_part_00.zip", help="DFDC archive part to download (default: dfdc_train_part_00.zip)")
    p_dfdc.add_argument("--archive-path", type=str, default=None, help="Optional local path to an existing DFDC zip archive")
    p_dfdc.add_argument("--output-dir", type=str, default=str(config.BASE_DIR / "data" / "raw_videos"), help="Destination directory for raw videos")
    p_dfdc.add_argument("--download-dir", type=str, default=str(config.RAW_DATA_DIR / "downloads"), help="Temporary download cache directory")
    p_dfdc.add_argument("--max-videos", type=int, default=None, help="Optional limit on number of videos to extract (preserves pairs)")

    # Command: train-all-datasets
    p_all = subparsers.add_parser(
        "train-all-datasets",
        help="Train EfficientNet-B3, ConvNeXt-Tiny, and YOLO26n-cls on every dataset"
    )
    p_all.add_argument("--datasets", nargs="+", default=None, choices=["celebdf", "dfdc"],
                       help="Datasets to train on (default: all)")
    p_all.add_argument("--epochs",     type=int, default=25)
    p_all.add_argument("--batch-size", type=int, default=32)
    p_all.add_argument("--patience",   type=int, default=7)
    p_all.add_argument("--num-workers", type=int, default=config.NUM_WORKERS,
                       help="DataLoader workers. On a many-core CPU box a low value "
                            "makes GPU training dataloader-bound.")
    p_all.add_argument("--pilot",      action="store_true", help="Quick 5-epoch smoke-test")
    p_all.add_argument("--skip-yolo",  action="store_true", help="Skip YOLO training")
    p_all.add_argument("--force",      action="store_true", help="Re-train even if checkpoint exists")

    # Command: build-cross-table
    p_ct = subparsers.add_parser(
        "build-cross-table",
        help="Evaluate all checkpoints across every dataset and produce cross-dataset table"
    )
    p_ct.add_argument("--batch-size", type=int, default=32)
    p_ct.add_argument("--device",     type=str, default=str(config.DEVICE))

    # Command: train-yolo
    p_yolo = subparsers.add_parser("train-yolo", help="Fine-tune YOLO-cls (yolo26n-cls) with class weights")
    p_yolo.add_argument("--data", type=str, default=str(config.PROCESSED_DATA_DIR), help="Dataset root with train/ and val/")
    p_yolo.add_argument("--epochs", type=int, default=30)
    p_yolo.add_argument("--imgsz", type=int, default=224)
    p_yolo.add_argument("--batch-size", type=int, default=32)
    p_yolo.add_argument("--lr", type=float, default=1e-3)
    p_yolo.add_argument("--patience", type=int, default=10)
    p_yolo.add_argument("--project", type=str, default=str(config.BASE_DIR / "runs" / "yolo_cls"))
    p_yolo.add_argument("--name", type=str, default="deepfake_yolo")
    p_yolo.add_argument("--workers", type=int, default=config.NUM_WORKERS)
    p_yolo.add_argument("--seed", type=int, default=42)
    p_yolo.add_argument("--device", type=str, default=str(config.DEVICE))

    # Command: train-timm
    p_timm = subparsers.add_parser("train-timm", help="Fine-tune timm backbone (convnext_tiny, efficientnet_b3) with class weights")
    p_timm.add_argument("--model-name", type=str, required=True, help="timm model (e.g. convnext_tiny.fb_in22k_ft_in1k, efficientnet_b3.ra2_in1k)")
    p_timm.add_argument("--data-dir", type=str, default=str(config.PROCESSED_DATA_DIR))
    p_timm.add_argument("--checkpoint-dir", type=str, default=str(config.CHECKPOINT_DIR))
    p_timm.add_argument("--reports-dir", type=str, default=str(config.REPORTS_DIR))
    p_timm.add_argument("--epochs", type=int, default=25)
    p_timm.add_argument("--batch-size", type=int, default=32)
    p_timm.add_argument("--lr", type=float, default=1e-4)
    p_timm.add_argument("--weight-decay", type=float, default=1e-4)
    p_timm.add_argument("--patience", type=int, default=7)
    p_timm.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p_timm.add_argument("--seed", type=int, default=42)
    p_timm.add_argument("--device", type=str, default=str(config.DEVICE))
    p_timm.add_argument("--no-pretrained", dest="pretrained", action="store_false")

    # Command: evaluate-ensemble
    p_ens = subparsers.add_parser("evaluate-ensemble", help="Evaluate YOLO26n-cls, ConvNeXt-Tiny, EfficientNet-B3 and soft-voting ensemble")
    p_ens.add_argument("--data-dir", type=str, default=str(config.PROCESSED_DATA_DIR), help="Dataset root containing test/real and test/fake")
    p_ens.add_argument("--dataset", type=str, default=None, choices=["celebdf", "dfdc", "real_life"], help="Specific dataset to evaluate on (overrides --data-dir)")
    p_ens.add_argument("--yolo-checkpoint", type=str, default=str(config.BASE_DIR / "runs" / "yolo_cls" / "deepfake_yolo" / "weights" / "best.pt"))
    p_ens.add_argument("--convnext-checkpoint", type=str, default=str(config.CHECKPOINT_DIR / "convnext_tiny.fb_in22k_ft_in1k_best.pth"))
    p_ens.add_argument("--effnet-checkpoint", type=str, default=str(config.CHECKPOINT_DIR / "efficientnet_b3.ra2_in1k_best.pth"))
    p_ens.add_argument("--convnext-model", type=str, default="convnext_tiny.fb_in22k_ft_in1k")
    p_ens.add_argument("--effnet-model", type=str, default="efficientnet_b3.ra2_in1k")
    p_ens.add_argument("--weights", type=float, nargs=3, default=[1.0, 1.0, 1.0], help="Weights for [yolo, convnext, effnet] (default: 1.0 1.0 1.0)")
    p_ens.add_argument("--tune-on-val", action="store_true", help="Tune ensemble weights strictly on the validation set before testing (zero test leakage)")
    p_ens.add_argument("--val-dir", type=str, default=str(config.PROCESSED_DATA_DIR / "val"), help="Validation directory for tuning weights")
    p_ens.add_argument("--tune-metric", type=str, default="roc_auc", choices=["roc_auc", "f1", "accuracy", "log_loss", "brier"], help="Metric to optimize on validation set (default: roc_auc)")
    p_ens.add_argument("--weights-file", type=str, default=None, help="Path to JSON file containing pre-tuned ensemble weights")
    p_ens.add_argument("--threshold", type=float, default=0.5)
    p_ens.add_argument("--reports-dir", type=str, default=str(config.REPORTS_DIR / "ensemble"))
    p_ens.add_argument("--device", type=str, default=str(config.DEVICE))
    p_ens.add_argument("--image", type=str, help="Single image inference")

    # Command: evaluate-cross-dataset
    p_cross = subparsers.add_parser("evaluate-cross-dataset", help="Evaluate models across multiple datasets (Celeb-DF, DFDC, Real-Life)")
    p_cross.add_argument("--celebdf-dir", type=str, default=str(config.PROCESSED_CELEBDF_DIR), help="Path to Celeb-DF processed directory")
    p_cross.add_argument("--dfdc-dir", type=str, default=str(config.PROCESSED_DFDC_DIR), help="Path to DFDC processed directory")
    p_cross.add_argument("--real-life-dir", type=str, default=str(config.REAL_LIFE_DIR), help="Path to Real-Life dataset directory")
    p_cross.add_argument("--yolo-checkpoint", type=str, default=str(config.BASE_DIR / "runs" / "yolo_cls" / "deepfake_yolo" / "weights" / "best.pt"))
    p_cross.add_argument("--convnext-checkpoint", type=str, default=str(config.CHECKPOINT_DIR / "convnext_tiny.fb_in22k_ft_in1k_best.pth"))
    p_cross.add_argument("--effnet-checkpoint", type=str, default=str(config.CHECKPOINT_DIR / "efficientnet_b3.ra2_in1k_best.pth"))
    p_cross.add_argument("--convnext-model", type=str, default="convnext_tiny.fb_in22k_ft_in1k")
    p_cross.add_argument("--effnet-model", type=str, default="efficientnet_b3.ra2_in1k")
    p_cross.add_argument("--weights-file", type=str, default=str(config.CHECKPOINT_DIR / "ensemble_weights.json"))
    p_cross.add_argument("--weights", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    p_cross.add_argument("--threshold", type=float, default=0.5)
    p_cross.add_argument("--reports-dir", type=str, default=str(config.REPORTS_DIR / "cross_dataset"))
    p_cross.add_argument("--device", type=str, default=str(config.DEVICE))

    # Command: test-robustness
    p_rob = subparsers.add_parser("test-robustness", help="Benchmark robustness under JPEG, Blur, and Resize perturbations")
    p_rob.add_argument("--data-dir", type=str, default=str(config.PROCESSED_DATA_DIR))
    p_rob.add_argument("--yolo-checkpoint", type=str, default=str(config.BASE_DIR / "runs" / "yolo_cls" / "deepfake_yolo" / "weights" / "best.pt"))
    p_rob.add_argument("--convnext-checkpoint", type=str, default=str(config.CHECKPOINT_DIR / "convnext_tiny.fb_in22k_ft_in1k_best.pth"))
    p_rob.add_argument("--effnet-checkpoint", type=str, default=str(config.CHECKPOINT_DIR / "efficientnet_b3.ra2_in1k_best.pth"))
    p_rob.add_argument("--weights-file", type=str, default=None, help="Path to JSON file containing pre-tuned ensemble weights")
    p_rob.add_argument("--jpeg-qualities", type=int, nargs="+", default=[90, 75, 50, 30, 15])
    p_rob.add_argument("--blur-radii", type=float, nargs="+", default=[1.0, 2.0, 3.0, 5.0])
    p_rob.add_argument("--resize-scales", type=float, nargs="+", default=[0.75, 0.50, 0.35, 0.20])
    p_rob.add_argument("--threshold", type=float, default=0.5)
    p_rob.add_argument("--reports-dir", type=str, default=str(config.REPORTS_DIR / "robustness"))
    p_rob.add_argument("--device", type=str, default=str(config.DEVICE))

    # Command: generate-reports
    p_rep = subparsers.add_parser("generate-reports", help="Generate results table and ROC plots from saved predictions")
    p_rep.add_argument("--predictions-file", type=str, default=str(config.REPORTS_DIR / "saved_predictions.csv"))
    p_rep.add_argument("--output-dir", type=str, default=str(config.REPORTS_DIR))
    p_rep.add_argument("--threshold", type=float, default=0.5)
    p_rep.add_argument("--min-videos-per-class", type=int, default=10, help="Minimum video clusters per class required")
    p_rep.add_argument("--allow-toy", action="store_true", help="Bypass sample size guard for smoke tests")

    # Command: benchmark-speed
    p_bench = subparsers.add_parser("benchmark-speed", help="Benchmark model parameters, GFLOPs, latency and throughput")
    p_bench.add_argument("--checkpoint-dir", type=str, default=str(config.CHECKPOINT_DIR),
                         help="Directory containing timm checkpoint .pth files")
    p_bench.add_argument("--yolo-checkpoint", type=str,
                         default=str(config.BASE_DIR / "runs" / "yolo_cls" / "deepfake_yolo" / "weights" / "best.pt"),
                         help="Path to YOLO weights (.pt)")
    p_bench.add_argument("--output-dir", type=str, default=str(config.REPORTS_DIR),
                         help="Directory to write benchmark table and chart")
    p_bench.add_argument("--device", type=str, default=str(config.DEVICE),
                         help="Device string: cpu or cuda")

    # Command: gradcam
    p_gcam = subparsers.add_parser("gradcam", help="Generate Grad-CAM heatmaps for a face image across all models")
    p_gcam.add_argument("--image", type=str, required=True, help="Path to input face image (JPEG/PNG)")
    p_gcam.add_argument("--checkpoint-dir", type=str, default=str(config.CHECKPOINT_DIR),
                        help="Directory containing timm checkpoint .pth files")
    p_gcam.add_argument("--yolo-checkpoint", type=str,
                        default=str(config.BASE_DIR / "runs" / "yolo_cls" / "deepfake_yolo" / "weights" / "best.pt"),
                        help="Path to YOLO weights (.pt)")
    p_gcam.add_argument("--output", type=str, default=str(config.REPORTS_DIR / "gradcam_output.png"),
                        help="Output path for the comparative Grad-CAM grid image")
    p_gcam.add_argument("--target-class", type=int, default=config.FAKE_LABEL_IDX, choices=[0, 1],
                        help="Class index to explain (0=fake, 1=real)")
    p_gcam.add_argument("--device", type=str, default=str(config.DEVICE),
                        help="Device string: cpu or cuda")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "demo-data":
        generate_samples(num_real=args.real, num_fake=args.fake)

    elif args.command == "split":
        split_dataset(
            source_dir=args.source,
            target_dir=args.dest,
            split_ratios=(args.train_ratio, args.val_ratio, args.test_ratio)
        )

    elif args.command == "extract-videos":
        import torch
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

    elif args.command == "verify-leakage":
        import json
        from preprocessing.split_by_identity import IdentitySourceSplitter, DataLeakageError
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            print(f"Error: Manifest not found at {manifest_path}")
            sys.exit(1)
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        metadata = None
        if args.metadata_file and Path(args.metadata_file).exists():
            metadata = IdentitySourceSplitter.parse_dfdc_metadata(args.metadata_file)

        splitter = IdentitySourceSplitter()
        # Reconstruct splits dict of Paths
        splits_dict = {}
        for s_name, s_data in manifest_data.get("splits", {}).items():
            splits_dict[s_name] = []
            for c_name, c_data in s_data.items():
                for v_info in c_data.get("videos", []):
                    splits_dict[s_name].append(Path(v_info["video_file"]))

        try:
            audit = splitter.verify_zero_leakage(splits_dict, metadata=metadata)
            print("\n" + "=" * 60)
            print("ZERO-LEAKAGE VERIFICATION AUDIT PASSED")
            print("=" * 60)
            print(f"Train sources: {audit['train_sources_count']}")
            print(f"Val sources:   {audit['val_sources_count']}")
            print(f"Test sources:  {audit['test_sources_count']}")
            print("No source or identity overlap across train, val, and test splits.")
            print("=" * 60)
        except DataLeakageError as e:
            print("\n" + "!" * 60)
            print(f"VERIFICATION FAILED: {e}")
            print("!" * 60)
            sys.exit(1)

    elif args.command == "download-dfdc":
        from scripts.download_dfdc_subset import (
            check_kaggle_auth,
            print_auth_instructions,
            download_dfdc_part,
            organize_dfdc_subset,
            download_dfdc_from_dataset,
        )
        if args.archive_path:
            local_archive = Path(args.archive_path)
            if not local_archive.exists():
                print(f"Error: Archive not found at {local_archive}")
                sys.exit(1)
            organize_dfdc_subset(local_archive, Path(args.output_dir), max_videos=args.max_videos)
        else:
            if not check_kaggle_auth():
                print_auth_instructions()
                sys.exit(1)
            # Prefer the public dataset route (no competition acceptance needed)
            download_dfdc_from_dataset(
                output_dir=Path(args.output_dir),
                download_dir=Path(args.download_dir),
                max_videos=args.max_videos,
            )

    elif args.command == "train-all-datasets":
        from scripts.train_all_datasets import run_all
        run_all(args)

    elif args.command == "build-cross-table":
        from scripts.build_cross_table import run_cross_table
        run_cross_table(args)

    elif args.command == "train":
        run_training(args)

    elif args.command == "train-yolo":
        from training.train_yolo import run_yolo_training
        run_yolo_training(args)

    elif args.command == "train-timm":
        from training.train_timm import run_timm_training
        run_timm_training(args)

    elif args.command == "evaluate":
        evaluate_model(args)

    elif args.command == "evaluate-ensemble":
        if getattr(args, "dataset", None) and args.dataset in config.DATASETS:
            args.data_dir = str(config.DATASETS[args.dataset])
        from evaluation.evaluate_ensemble import run_ensemble_evaluation
        run_ensemble_evaluation(args)

    elif args.command == "evaluate-cross-dataset":
        from evaluation.evaluate_cross_dataset import run_cross_dataset_evaluation
        run_cross_dataset_evaluation(args)

    elif args.command == "test-robustness":
        from evaluation.robustness_test import run_robustness_benchmark
        run_robustness_benchmark(args)

    elif args.command == "generate-reports":
        from evaluation.generate_reports import generate_reports_from_predictions
        generate_reports_from_predictions(
            predictions_file=args.predictions_file,
            output_dir=args.output_dir,
            threshold=args.threshold,
            min_videos_per_class=getattr(args, "min_videos_per_class", 10),
            allow_toy=getattr(args, "allow_toy", False)
        )

    elif args.command == "benchmark-speed":
        from pathlib import Path as _Path
        from evaluation.benchmark_speed import benchmark_all_models
        benchmark_all_models(
            checkpoint_dir=_Path(args.checkpoint_dir),
            yolo_checkpoint=_Path(args.yolo_checkpoint),
            output_dir=_Path(args.output_dir),
            device_str=args.device,
        )

    elif args.command == "gradcam":
        import torch as _torch
        import timm as _timm
        from pathlib import Path as _Path
        from evaluation.gradcam import generate_comparative_gradcam

        _device = _torch.device(args.device)
        _ckpt_dir = _Path(args.checkpoint_dir)

        # Load ConvNeXt-Tiny
        _cn = _timm.create_model("convnext_tiny.fb_in22k_ft_in1k", pretrained=False, num_classes=2)
        _cn_path = _ckpt_dir / "convnext_tiny.fb_in22k_ft_in1k_best.pth"
        if _cn_path.exists():
            _sd = _torch.load(_cn_path, map_location="cpu")
            _cn.load_state_dict(_sd.get("state_dict", _sd))

        # Load EfficientNet-B3
        _eff = _timm.create_model("efficientnet_b3.ra2_in1k", pretrained=False, num_classes=2)
        _eff_path = _ckpt_dir / "efficientnet_b3.ra2_in1k_best.pth"
        if _eff_path.exists():
            _sd = _torch.load(_eff_path, map_location="cpu")
            _eff.load_state_dict(_sd.get("state_dict", _sd))

        # Load YOLO
        from ultralytics import YOLO as _YOLO
        _yolo_w = _YOLO(str(args.yolo_checkpoint))

        models_dict = {
            "ConvNeXt-Tiny": (_cn, "timm"),
            "EfficientNet-B3": (_eff, "timm"),
            "YOLO26n-cls": (_yolo_w.model, "yolo"),
        }
        generate_comparative_gradcam(
            image_path=args.image,
            models_dict=models_dict,
            output_path=_Path(args.output),
            target_class=args.target_class,
            device=_device,
        )
        print(f"Grad-CAM saved to: {args.output}")

    elif args.command == "predict":
        run_inference(args)


if __name__ == "__main__":
    main()

