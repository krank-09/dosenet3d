"""DVH and resampling helpers shared with the companion dvhnet repo.

Copied from dvhnet (preprocessing.py / metrics.py, same author) so that this
repo runs on its own; previously these were imported from a sibling
`../dvhnet` checkout at runtime.

Cumulative DVH convention used throughout:
    y[k] = fraction of a structure's voxels receiving dose >= D_k,
    D_k = linspace(0, dose_max_gy, num_bins)[k]
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import zoom

# np.trapz was renamed to np.trapezoid in NumPy 2.0; support both.
_trapz = getattr(np, "trapezoid", None) or np.trapz


# --------------------------------------------------------------------------- #
# resampling
# --------------------------------------------------------------------------- #

def resample_volume(volume: np.ndarray,
                    src_spacing: Tuple[float, float, float],
                    dst_spacing: Tuple[float, float, float],
                    dst_shape: Optional[Tuple[int, int, int]] = None,
                    order: int = 1) -> np.ndarray:
    """Resample a [Z, H, W] volume from src_spacing to dst_spacing (mm).
    order=1 (linear) for continuous CT/dose; order=0 (nearest) for masks so
    hard boundaries stay hard."""
    zoom_factors = [s / d for s, d in zip(src_spacing, dst_spacing)]
    resampled = zoom(volume, zoom_factors, order=order)
    if dst_shape is not None:
        resampled = center_crop_or_pad(resampled, dst_shape)
    return resampled


def center_crop_or_pad(vol: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    out = np.zeros(target_shape, dtype=vol.dtype)
    slices_src, slices_dst = [], []
    for s_src, s_dst in zip(vol.shape, target_shape):
        if s_src >= s_dst:
            start = (s_src - s_dst) // 2
            slices_src.append(slice(start, start + s_dst))
            slices_dst.append(slice(0, s_dst))
        else:
            start = (s_dst - s_src) // 2
            slices_src.append(slice(0, s_src))
            slices_dst.append(slice(start, start + s_src))
    out[tuple(slices_dst)] = vol[tuple(slices_src)]
    return out


# --------------------------------------------------------------------------- #
# DVH construction and dose-metric inversion
# --------------------------------------------------------------------------- #

def cumulative_dvh(dose: np.ndarray, mask: np.ndarray,
                   dose_max_gy: float = 80.0, num_bins: int = 256) -> Optional[np.ndarray]:
    """Cumulative DVH of the voxels where mask > 0 (any shape). None if empty."""
    voxel_doses = dose[mask > 0]
    if voxel_doses.size == 0:
        return None
    edges = np.linspace(0.0, dose_max_gy, num_bins)
    return (voxel_doses[:, None] >= edges[None, :]).mean(axis=0).astype(np.float32)


def dose_at_volume(dvh: np.ndarray, volume_percent: float, dose_max_gy: float) -> float:
    """D_x%: the dose at which the cumulative DVH crosses x% of the volume."""
    doses = np.linspace(0.0, dose_max_gy, dvh.shape[-1])
    target = volume_percent / 100.0
    y = np.minimum.accumulate(dvh)          # enforce non-increasing
    if target >= y[0]:
        return float(doses[0])
    if target <= y[-1]:
        return float(doses[-1])
    return float(np.interp(target, y[::-1], doses[::-1]))  # interp needs increasing x


def d2(dvh: np.ndarray, dose_max_gy: float) -> float:
    """Near-maximum dose: dose received by the hottest 2% of the structure."""
    return dose_at_volume(dvh, 2.0, dose_max_gy)


def d50(dvh: np.ndarray, dose_max_gy: float) -> float:
    """Median dose."""
    return dose_at_volume(dvh, 50.0, dose_max_gy)


def d_mean(dvh: np.ndarray, dose_max_gy: float) -> float:
    """Mean dose = area under the cumulative DVH (E[X] = ∫ P(X ≥ d) dd for X ≥ 0)."""
    doses = np.linspace(0.0, dose_max_gy, dvh.shape[-1])
    return float(_trapz(np.minimum.accumulate(dvh), doses))
