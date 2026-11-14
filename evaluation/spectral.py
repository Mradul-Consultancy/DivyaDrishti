"""
Frequency-domain (spectral) analysis for deepfake detection.

Generative face synthesis leaves traces in the frequency domain that are
invisible spatially: upsampling grids, checkerboard artefacts, and an
abnormal fall-off of the spectrum compared with a camera-captured face.

This module computes a small set of interpretable spectral descriptors for
a face crop:

* **High-frequency energy ratio** - how much of the image energy lives in
  the upper half of the radial spectrum.
* **Radial spectral slope** - how quickly energy decays with frequency. A
  camera's sensor noise gives a characteristic slope; a generator's
  resampling produces a different one.
* **Periodic peak strength** - the prominence of the single strongest
  non-DC peak. Resampling grids concentrate energy at one frequency, so a
  sharp peak is a grid artefact.
* **Azimuthal anisotropy** - energy imbalance between horizontal and
  vertical frequencies, which rises when a generator upsamples in one
  axis only.

These are descriptors, not a classifier. They are exposed so they can be
measured and combined with the neural score - the honest way to evaluate
whether they add anything is to measure it, not to assume it.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:  # pragma: no cover
    CV2_AVAILABLE = False


@dataclass
class SpectralFeatures:
    """Frequency-domain descriptors of a face crop."""

    high_freq_energy_ratio: float
    radial_spectral_slope: float
    periodic_peak_strength: float
    azimuthal_anisotropy: float
    spectral_entropy: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "high_freq_energy_ratio": round(self.high_freq_energy_ratio, 6),
            "radial_spectral_slope": round(self.radial_spectral_slope, 6),
            "periodic_peak_strength": round(self.periodic_peak_strength, 6),
            "azimuthal_anisotropy": round(self.azimuthal_anisotropy, 6),
            "spectral_entropy": round(self.spectral_entropy, 6),
        }

    def as_vector(self) -> np.ndarray:
        return np.array([
            self.high_freq_energy_ratio,
            self.radial_spectral_slope,
            self.periodic_peak_strength,
            self.azimuthal_anisotropy,
            self.spectral_entropy,
        ], dtype=float)


def compute_spectral_features(gray: np.ndarray) -> SpectralFeatures:
    """
    Computes the spectral descriptors of a single-channel image.

    Args:
        gray: 2D grayscale array (H x W).
    """
    gray = np.asarray(gray, dtype=np.float64)
    if gray.ndim != 2 or gray.size == 0:
        raise ValueError("expected a non-empty 2D grayscale array")

    h, w = gray.shape
    # Remove the mean so the DC term does not dominate the spectrum.
    gray = gray - gray.mean()

    window = np.hanning(h)[:, None] * np.hanning(w)[None, :]
    spectrum = np.fft.fftshift(np.fft.fft2(gray * window))
    magnitude = np.abs(spectrum)
    power = magnitude ** 2

    total = power.sum()
    if total <= 0:
        return SpectralFeatures(0.0, 0.0, 0.0, 0.0, 0.0)

    # Radial frequency coordinates.
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_r = radius.max()
    if max_r <= 0:
        return SpectralFeatures(0.0, 0.0, 0.0, 0.0, 0.0)
    norm_r = radius / max_r

    # 1. High-frequency energy ratio.
    high_mask = norm_r > 0.5
    high_ratio = float(power[high_mask].sum() / total)

    # 2. Radial spectral slope: least-squares slope of log power vs radius.
    r_bins = np.linspace(0, 1, 32)
    radial_profile = np.zeros(len(r_bins) - 1)
    for i in range(len(r_bins) - 1):
        band = (norm_r >= r_bins[i]) & (norm_r < r_bins[i + 1])
        radial_profile[i] = power[band].mean() if band.any() else 0.0
    valid = radial_profile > 0
    if valid.sum() >= 2:
        x = (r_bins[:-1] + r_bins[1:]) / 2.0
        slope = float(np.polyfit(x[valid], np.log(radial_profile[valid]), 1)[0])
    else:
        slope = 0.0

    # 3. Periodic peak strength: strongest non-DC bin relative to its ring.
    masked = magnitude.copy()
    masked[max(0, cy - 2):cy + 3, max(0, cx - 2):cx + 3] = 0.0
    peak = float(masked.max())
    ring = (norm_r > 0.2) & (norm_r < 0.8)
    ring_mean = float(masked[ring].mean()) if ring.any() else 0.0
    peak_strength = float(peak / ring_mean) if ring_mean > 0 else 0.0

    # 4. Azimuthal anisotropy: horizontal vs vertical frequency energy.
    horiz = (np.abs(yy - cy) > np.abs(xx - cx))
    vert = (np.abs(xx - cx) > np.abs(yy - cy))
    e_h = float(power[horiz].sum())
    e_v = float(power[vert].sum())
    anisotropy = float((e_h - e_v) / max(1e-9, e_h + e_v))

    # 5. Spectral entropy of the radial profile (flatness of the spectrum).
    p = radial_profile[valid]
    if p.size > 1 and p.sum() > 0:
        pn = p / p.sum()
        entropy = float(-(pn * np.log(pn + 1e-12)).sum() / np.log(pn.size))
    else:
        entropy = 0.0

    return SpectralFeatures(
        high_freq_energy_ratio=high_ratio,
        radial_spectral_slope=slope,
        periodic_peak_strength=peak_strength,
        azimuthal_anisotropy=anisotropy,
        spectral_entropy=entropy,
    )


def compute_for_image(image) -> SpectralFeatures:
    """
    Computes the spectral descriptors of a PIL image or numpy array.
    """
    if CV2_AVAILABLE:
        if hasattr(image, "convert"):  # PIL
            arr = np.array(image.convert("RGB"))
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        else:
            arr = np.asarray(image)
            if arr.ndim == 3:
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            else:
                gray = arr
    else:
        arr = np.asarray(image)
        gray = arr if arr.ndim == 2 else arr.mean(axis=2)
    return compute_spectral_features(gray)


def feature_names() -> list:
    """Names of the descriptors, in the order of ``as_vector``."""
    return [
        "high_freq_energy_ratio",
        "radial_spectral_slope",
        "periodic_peak_strength",
        "azimuthal_anisotropy",
        "spectral_entropy",
    ]
