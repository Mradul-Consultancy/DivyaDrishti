"""
Regression tests for metric orientation / label mapping.

Background
----------
The project uses a fixed label convention (see config.py):

    FAKE = 0, REAL = 1

and every probability produced by the pipeline is **P(fake)**.

These tests pin down the two failure modes that previously corrupted the
client deliverable:

1.  **Inverted ROC-AUC.** ``sklearn.metrics.roc_auc_score`` defaults to
    ``pos_label=1`` (i.e. REAL). Feeding it P(fake) scores returns
    ``1 - AUC``, silently inverting every AUC in the report. The pipeline
    must therefore use ``roc_curve(..., pos_label=FAKE_LABEL_IDX)`` + ``auc``.

2.  **Inverted predictions.** ``(probs >= 0.5).astype(int)`` maps a high
    P(fake) to label 1 (REAL), which transposes the confusion matrix and
    corrupts precision/recall/F1.

Both engines that produce reported numbers
(scripts/build_cross_table.py and evaluation/evaluate_ensemble.py) must
agree with a hand-computed reference.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_curve, auc

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def _reference_auc(y_true, prob_fake):
    """Reference AUC with FAKE as the positive class."""
    fpr, tpr, _ = roc_curve(y_true, prob_fake, pos_label=config.FAKE_LABEL_IDX)
    return float(auc(fpr, tpr))


class TestLabelConvention:
    """The label convention itself must never drift."""

    def test_fake_is_zero_and_real_is_one(self):
        assert config.FAKE_LABEL_IDX == 0
        assert config.REAL_LABEL_IDX == 1
        assert config.CLASS_TO_IDX["fake"] == config.FAKE_LABEL_IDX
        assert config.CLASS_TO_IDX["real"] == config.REAL_LABEL_IDX


class TestAucOrientation:
    """ROC-AUC must treat FAKE as the positive class."""

    def test_perfect_detector_has_auc_one(self):
        # y_true: fake=0, real=1 ; prob_fake high for fakes, low for reals
        y_true = np.array([0, 0, 0, 1, 1, 1])
        prob_fake = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
        assert _reference_auc(y_true, prob_fake) == pytest.approx(1.0)

    def test_inverted_detector_has_auc_zero(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        prob_fake = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])
        assert _reference_auc(y_true, prob_fake) == pytest.approx(0.0)

    def test_default_pos_label_would_invert(self):
        """
        Guards the exact bug: using sklearn's default pos_label=1 with a
        P(fake) score returns 1 - AUC, which silently flips the metric.
        """
        from sklearn.metrics import roc_auc_score

        y_true = np.array([0, 0, 0, 1, 1, 1])
        prob_fake = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])

        correct = _reference_auc(y_true, prob_fake)
        wrong = roc_auc_score(y_true, prob_fake)  # default pos_label=1

        assert wrong == pytest.approx(1.0 - correct)
        assert correct == pytest.approx(1.0)
        assert wrong == pytest.approx(0.0)


class TestCrossTableMetricHelper:
    """scripts/build_cross_table.py must use the corrected convention."""

    @pytest.fixture(scope="class")
    def helper(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "build_cross_table",
            Path(__file__).resolve().parent.parent / "scripts" / "build_cross_table.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_perfect_detector(self, helper):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        prob_fake = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
        res = helper._compute_metrics(y_true, prob_fake)

        assert res["roc_auc"] == pytest.approx(1.0)
        assert res["accuracy"] == pytest.approx(1.0)
        assert res["f1_score"] == pytest.approx(1.0)
        # cm layout [FAKE, REAL]: cm[0][0]=TP, cm[1][1]=TN
        assert res["confusion_matrix"] == [[3, 0], [0, 3]]

    def test_inverted_detector_scores_wrong(self, helper):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        prob_fake = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])
        res = helper._compute_metrics(y_true, prob_fake)

        assert res["roc_auc"] == pytest.approx(0.0)
        assert res["accuracy"] == pytest.approx(0.0)

    def test_prediction_direction_is_fake_on_high_prob(self, helper):
        """High P(fake) must map to label FAKE (0), not REAL (1)."""
        y_true = np.array([0, 1])
        prob_fake = np.array([0.99, 0.01])
        res = helper._compute_metrics(y_true, prob_fake)

        # Both predicted correctly -> diagonal confusion matrix
        assert res["confusion_matrix"] == [[1, 0], [0, 1]]
        assert res["accuracy"] == pytest.approx(1.0)
        assert res["recall"] == pytest.approx(1.0)  # recall for FAKE class

    def test_constant_prediction_detected(self, helper):
        """Predicting 'fake' for everything must show a degenerate CM."""
        y_true = np.array([0, 0, 1, 1])
        prob_fake = np.array([0.9, 0.9, 0.9, 0.9])
        res = helper._compute_metrics(y_true, prob_fake)

        assert res["confusion_matrix"] == [[2, 0], [2, 0]]
        assert res["accuracy"] == pytest.approx(0.5)
        assert res["precision"] == pytest.approx(0.5)
        assert res["recall"] == pytest.approx(1.0)
        assert res["f1_score"] == pytest.approx(2 / 3)

    def test_brier_score_uses_fake_binary(self, helper):
        """Brier score must compare P(fake) against the is-fake indicator."""
        y_true = np.array([0, 1])
        prob_fake = np.array([1.0, 0.0])  # perfectly calibrated
        res = helper._compute_metrics(y_true, prob_fake)
        assert res["brier_score"] == pytest.approx(0.0)


class TestCrossTableHasNoHardcodedMetrics:
    """The 'no hand-written numbers' rule must be enforced mechanically."""

    @pytest.fixture(scope="class")
    def source(self):
        return (
            Path(__file__).resolve().parent.parent / "scripts" / "build_cross_table.py"
        ).read_text(encoding="utf-8")

    def test_no_eval_cache_dict(self, source):
        assert "_EVAL_CACHE" not in source, (
            "A hardcoded metrics cache was reintroduced. All numbers in the "
            "client deliverable must be measured, not hardcoded."
        )

    def test_no_bare_roc_auc_score_call(self, source):
        """
        A bare roc_auc_score(y_true, prob_fake) inverts the metric because
        sklearn defaults to pos_label=1 (REAL).
        """
        offending = [
            line.strip()
            for line in source.splitlines()
            if "roc_auc_score(" in line and not line.strip().startswith("#")
        ]
        assert not offending, (
            "roc_auc_score() must not be used with P(fake) scores: it defaults "
            f"to pos_label=1 and returns 1 - AUC. Offending lines: {offending}"
        )


class TestEnsembleSampleAlignment:
    """
    The soft-voting ensemble must combine probabilities for the SAME image.

    Regression test: the YOLO branch used to build its own file list in
    ["real", "fake"] order while the timm branches used DeepfakeDataset's
    alphabetical ("fake", "real") order. The two probability arrays were
    therefore misaligned, and the ensemble averaged a fake image's YOLO
    probability with a real image's timm probabilities - producing nonsense
    AUCs (e.g. 0.1100 on Celeb-DF, below both of its members).
    """

    @pytest.fixture(scope="class")
    def module(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "build_cross_table",
            Path(__file__).resolve().parent.parent / "scripts" / "build_cross_table.py",
        )
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_collect_samples_matches_dataset_order(self, module):
        """_collect_samples must produce the same order as DeepfakeDataset."""
        from training.dataset import DeepfakeDataset

        # NOTE: pytest's tmp_path fixture cannot be used here - on this
        # machine the pytest temp root raises PermissionError. Use
        # tempfile.TemporaryDirectory, matching tests/test_min_videos_per_class.py.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for cls in ("fake", "real"):
                d = root / cls
                d.mkdir()
                for i in range(3):
                    (d / f"img_{i}.jpg").write_bytes(b"x")

            ds = DeepfakeDataset(root)
            expected = [(str(p), lbl) for p, lbl in ds.samples]

            actual = [(str(p), lbl) for p, lbl in module._collect_samples(root)]
            assert actual == expected, (
                "_collect_samples order diverges from DeepfakeDataset; the "
                "ensemble would combine probabilities from different images."
            )

    def test_collect_samples_is_fake_first(self, module):
        """Alphabetical class order means fake (0) precedes real (1)."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for cls in ("fake", "real"):
                d = root / cls
                d.mkdir()
                (d / "a.jpg").write_bytes(b"x")

            samples = module._collect_samples(root)
            labels = [lbl for _, lbl in samples]
            assert labels == sorted(labels), (
                f"Expected fake-then-real ordering, got {labels}"
            )
