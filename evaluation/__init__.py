"""Evaluation package for deepfake detection."""

from evaluation.evaluate import evaluate_model
from evaluation.predict import predict_single_image, run_inference
from evaluation.evaluate_ensemble import ModelEnsembleEvaluator, run_ensemble_evaluation
from evaluation.robustness_test import (
    RobustnessTester,
    run_robustness_benchmark,
    apply_jpeg_compression,
    apply_gaussian_blur,
    apply_resize_downscale,
    apply_combined_distortion
)

from evaluation.generate_reports import PredictionReportGenerator, generate_reports_from_predictions
from evaluation.gradcam import GradCAM, generate_comparative_gradcam
from evaluation.benchmark_speed import benchmark_all_models

__all__ = [
    "evaluate_model",
    "predict_single_image",
    "run_inference",
    "ModelEnsembleEvaluator",
    "run_ensemble_evaluation",
    "RobustnessTester",
    "run_robustness_benchmark",
    "apply_jpeg_compression",
    "apply_gaussian_blur",
    "apply_resize_downscale",
    "apply_combined_distortion",
    "PredictionReportGenerator",
    "generate_reports_from_predictions",
    "GradCAM",
    "generate_comparative_gradcam",
    "benchmark_all_models",
]

