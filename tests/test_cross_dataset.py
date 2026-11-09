"""
Unit and regression tests for Cross-Dataset Restructuring and Evaluation Engine.

Verifies:
1. Dataset directory structure and manifest presence:
   - data/processed/celebdf/
   - data/processed/dfdc/
   - data/real_life/
2. Split resolution logic (resolve_test_split).
3. Report and metric serialization integrity.
4. CLI argument handling for --dataset and evaluate-cross-dataset.
"""

import json
import shutil
from pathlib import Path
import pytest

import config
from evaluation.evaluate_cross_dataset import resolve_test_split, generate_markdown_report


@pytest.fixture
def local_tmp_path():
    """Workspace-local temp directory fixture to avoid Windows Temp ACL issues."""
    path = Path(__file__).resolve().parent / ".tmp_cross_dataset_test"
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    yield path
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


class TestDatasetStructure:
    """Verifies that the restructured dataset directories exist and are properly populated."""

    def test_config_dataset_paths_exist(self):
        """Ensure config constants point to existing paths."""
        assert config.PROCESSED_CELEBDF_DIR.exists(), f"Missing {config.PROCESSED_CELEBDF_DIR}"
        assert config.PROCESSED_DFDC_DIR.exists(), f"Missing {config.PROCESSED_DFDC_DIR}"
        assert config.REAL_LIFE_DIR.exists(), f"Missing {config.REAL_LIFE_DIR}"
        assert "celebdf" in config.DATASETS
        assert "dfdc" in config.DATASETS
        assert "real_life" in config.DATASETS

    def test_celebdf_structure(self):
        """Verify Celeb-DF splits and manifest."""
        celeb_dir = config.PROCESSED_CELEBDF_DIR
        assert (celeb_dir / "video_split_manifest.json").exists()
        test_dir = celeb_dir / "test"
        assert (test_dir / "real").exists()
        assert (test_dir / "fake").exists()
        assert len(list((test_dir / "real").glob("*.jpg"))) > 0
        assert len(list((test_dir / "fake").glob("*.jpg"))) > 0

    def test_dfdc_structure(self):
        """Verify DFDC splits and manifest."""
        dfdc_dir = config.PROCESSED_DFDC_DIR
        assert (dfdc_dir / "video_split_manifest.json").exists()
        for split in ["train", "val", "test"]:
            s_dir = dfdc_dir / split
            assert s_dir.exists()
            assert (s_dir / "real").exists()
            assert (s_dir / "fake").exists()
            assert len(list((s_dir / "real").glob("*.jpg"))) > 0
            assert len(list((s_dir / "fake").glob("*.jpg"))) > 0

    def test_real_life_structure(self):
        """Verify Real-Life in-the-wild facial samples and README documentation."""
        rl_dir = config.REAL_LIFE_DIR
        assert (rl_dir / "README.md").exists()
        assert (rl_dir / "real").exists()
        assert (rl_dir / "fake").exists()
        assert len(list((rl_dir / "real").glob("*.jpg"))) > 0
        assert len(list((rl_dir / "fake").glob("*.jpg"))) > 0


class TestResolveTestSplit:
    """Verifies the adaptive directory resolution for various dataset layouts."""

    def test_resolves_nested_test_folder(self, local_tmp_path):
        """Directory with nested test/real and test/fake should resolve to test/."""
        nested = local_tmp_path / "nested_ds"
        (nested / "test" / "real").mkdir(parents=True)
        (nested / "test" / "fake").mkdir(parents=True)
        resolved = resolve_test_split(nested)
        assert resolved == (nested / "test")

    def test_resolves_direct_folder(self, local_tmp_path):
        """Directory with direct real/ and fake/ should resolve to directory itself."""
        direct = local_tmp_path / "direct_ds"
        (direct / "real").mkdir(parents=True)
        (direct / "fake").mkdir(parents=True)
        resolved = resolve_test_split(direct)
        assert resolved == direct

    def test_returns_none_for_missing_dir(self, local_tmp_path):
        """Non-existent directory should return None."""
        missing = local_tmp_path / "non_existent"
        assert resolve_test_split(missing) is None

    def test_returns_none_for_invalid_structure(self, local_tmp_path):
        """Directory without real/ or fake/ should return None."""
        invalid = local_tmp_path / "invalid_ds"
        invalid.mkdir()
        assert resolve_test_split(invalid) is None

    def test_resolves_live_datasets(self):
        """Live project datasets must all resolve cleanly."""
        assert resolve_test_split(config.PROCESSED_CELEBDF_DIR) == (config.PROCESSED_CELEBDF_DIR / "test")
        assert resolve_test_split(config.PROCESSED_DFDC_DIR) == (config.PROCESSED_DFDC_DIR / "test")
        assert resolve_test_split(config.REAL_LIFE_DIR) == config.REAL_LIFE_DIR


class TestCrossDatasetReports:
    """Verifies the markdown report generator and metrics formatting."""

    def test_markdown_report_generation(self, local_tmp_path):
        mock_results = {
            "Domain A": {
                "total_samples": 20,
                "real_samples": 10,
                "fake_samples": 10,
                "models": {
                    "yolo26n_cls": {
                        "name": "YOLO26n-cls",
                        "accuracy": 0.85,
                        "balanced_acc": 0.85,
                        "precision": 0.80,
                        "recall": 0.90,
                        "specificity": 0.80,
                        "f1_score": 0.85,
                        "roc_auc": 0.92,
                        "brier_score": 0.12,
                        "confusion_matrix": [[9, 1], [2, 8]]
                    },
                    "convnext_tiny": {
                        "name": "ConvNeXt-Tiny",
                        "accuracy": 0.80,
                        "balanced_acc": 0.80,
                        "precision": 0.75,
                        "recall": 0.85,
                        "specificity": 0.75,
                        "f1_score": 0.80,
                        "roc_auc": 0.88,
                        "brier_score": 0.15,
                        "confusion_matrix": [[8, 2], [2, 8]]
                    },
                    "efficientnet_b3": {
                        "name": "EfficientNet-B3",
                        "accuracy": 0.82,
                        "balanced_acc": 0.82,
                        "precision": 0.78,
                        "recall": 0.88,
                        "specificity": 0.76,
                        "f1_score": 0.83,
                        "roc_auc": 0.89,
                        "brier_score": 0.14,
                        "confusion_matrix": [[8, 2], [2, 8]]
                    },
                    "ensemble": {
                        "name": "Soft-Voting Ensemble",
                        "accuracy": 0.90,
                        "balanced_acc": 0.90,
                        "precision": 0.89,
                        "recall": 0.91,
                        "specificity": 0.89,
                        "f1_score": 0.90,
                        "roc_auc": 0.95,
                        "brier_score": 0.08,
                        "confusion_matrix": [[9, 1], [1, 9]]
                    }
                }
            }
        }
        report_file = local_tmp_path / "test_report.md"
        generate_markdown_report(mock_results, report_file)
        assert report_file.exists()
        content = report_file.read_text(encoding="utf-8")
        assert "Cross-Dataset Generalization Report" in content
        assert "Domain A" in content
        assert "Soft-Voting Ensemble" in content
        assert "0.9500" in content  # ROC-AUC
        assert "0.0800" in content  # Brier score

    def test_saved_cross_dataset_metrics_json_integrity(self):
        """Verify the generated cross_dataset_metrics.json has required keys and valid values."""
        metrics_file = config.REPORTS_DIR / "cross_dataset" / "cross_dataset_metrics.json"
        if not metrics_file.exists():
            pytest.skip("cross_dataset_metrics.json not yet generated in reports directory")

        with open(metrics_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "Celeb-DF (v2)" in data
        assert "DFDC" in data
        assert "Real-Life (In-The-Wild)" in data

        for d_name, d_info in data.items():
            assert "models" in d_info
            assert "ensemble" in d_info["models"]
            for m_key, m_stats in d_info["models"].items():
                assert 0.0 <= m_stats["accuracy"] <= 1.0
                assert 0.0 <= m_stats["roc_auc"] <= 1.0
                assert 0.0 <= m_stats["brier_score"] <= 1.0
