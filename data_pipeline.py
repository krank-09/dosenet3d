"""
data_pipeline.py
=================
Real-data preprocessing logic for dosenet3d, structured so it can already be
exercised on synthetic data (see synthetic_data.py) with zero code changes
once real GDP-HMM data is available -- only `load_and_preprocess_patient`'s
DICOM-reading calls (currently stubbed) need to be wired up, or replaced by
a GDP-HMM-specific adapter that calls `preprocess_arrays` directly.

Tensor layout (see module docstring in synthetic_data.py for the full spec):
    input:  [C, H, W, D]  (C=3 by default: CT, PTV, OAR)
    target: [1, H, W, D]  dose in Gy (or normalized by prescription dose)

Two layers:
  - `preprocess_arrays` and its helpers (window_and_normalize_ct,
    build_ptv_channel, build_oar_channel, resample_inplane_only,
    crop_or_pad_depth, ...): pure numpy, fully implemented, real-data-ready.
    These operate on already-loaded [Z, H, W] arrays regardless of how they
    were loaded, so they're directly testable without any DICOM/GDP-HMM
    access -- synthetic_data.py exercises the channel-construction pieces
    directly.
  - `load_and_preprocess_patient`: the real-data entry point. Raises
    NotImplementedError -- GDP-HMM Hugging Face access is still pending.
    When wiring it up, reuse dvhnet/preprocessing.py's DICOM/RTDOSE/RTSTRUCT
    loaders (do not reimplement them); see its docstring below for exactly
    which functions to call.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Reuse dvhnet's DICOM/resampling helpers rather than reimplementing them.
#
# Loaded via importlib under a private module name (NOT sys.path.insert +
# plain `import`) because dvhnet/ has its own model.py, losses.py, dataset.py
# and train.py -- putting dvhnet/ on sys.path would silently shadow
# dosenet3d's same-named modules (hit exactly this bug the first time this
# was written as a plain sys.path insert: `from losses import
# CompositeDoseLoss` in this package's train.py resolved to dvhnet/losses.py
# instead). This way there's no shared namespace at all.
# --------------------------------------------------------------------------- #

def _load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses' internal type resolution looks the module up via
    # sys.modules[cls.__module__] -- must register before exec_module.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_DVHNET_DIR = Path(__file__).resolve().parent.parent / "dvhnet"
_dvhnet_preprocessing = _load_module_from_path("_dvhnet_preprocessing", _DVHNET_DIR / "preprocessing.py")
_dvhnet_resample_volume = _dvhnet_preprocessing.resample_volume


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class PreprocessConfig:
    ct_clip_hu: Tuple[float, float] = (-200.0, 1000.0)   # H&N windowing (~WW 1200 / WL 400 equivalent range)
    ct_normalize: str = "minmax"                          # "minmax" -> [0,1], or "zscore"
    target_hw: Tuple[int, int] = (256, 256)
    target_depth: int = 80
    ptv_scale_by_dose: bool = False                        # False: binary union mask; True: scaled by Rx/ptv_max_dose_gy per level
    ptv_max_dose_gy: float = 80.0                          # denominator when ptv_scale_by_dose=True
    oar_one_hot: bool = False                              # False (default): single integer-encoded channel; True: one-hot stack (changes total channel count -- see total_input_channels)
    dose_normalize_by_prescription: bool = False           # False: keep Gy; True: divide by prescription dose
    resample_order_ct: int = 1                             # linear
    resample_order_dose: int = 1                            # linear
    resample_order_mask: int = 0                            # nearest -- keep hard boundaries, see dvhnet/README.md's same caveat
    dose_rescale_to_prescription_d97: bool = False          # rescale dose so D97 in the highest-Rx PTV exactly equals its prescription (see rescale_dose_to_prescription_d97) -- convention from gdp_hmm_reference/full_source/data_loader.py; off by default, changes the target's numeric scale
    dose_rescale_clip_factor: float = 1.2                   # when the above is on, clip dose to [0, Rx_high * this] after rescaling


DEFAULT_OAR_LABEL_MAP: Dict[str, int] = {
    "Brainstem": 1,
    "SpinalCord": 2,
    "Parotid_L": 3,
    "Parotid_R": 4,
}


def total_input_channels(cfg: PreprocessConfig, num_oar_labels: int) -> int:
    """2 (CT + PTV) + 1 (integer-encoded OAR) or + num_oar_labels (one-hot)."""
    return 2 + (num_oar_labels if cfg.oar_one_hot else 1)


# --------------------------------------------------------------------------- #
# Channel construction (axis-agnostic: works on [Z,H,W] or already-[H,W,D] arrays)
# --------------------------------------------------------------------------- #

def window_and_normalize_ct(ct_hu: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Clip to the configured HU window, then normalize to [0,1] (minmax) or
    zero-mean/unit-variance (zscore, computed on the clipped volume)."""
    lo, hi = cfg.ct_clip_hu
    clipped = np.clip(ct_hu, lo, hi).astype(np.float32)
    if cfg.ct_normalize == "minmax":
        return ((clipped - lo) / (hi - lo)).astype(np.float32)
    if cfg.ct_normalize == "zscore":
        mean, std = clipped.mean(), clipped.std()
        return ((clipped - mean) / (std + 1e-8)).astype(np.float32)
    raise ValueError(f"unknown ct_normalize mode: {cfg.ct_normalize!r}")


def build_ptv_channel(ptv_masks: Dict[str, np.ndarray],
                       prescription_doses: Dict[str, float],
                       cfg: PreprocessConfig) -> np.ndarray:
    """
    Single-channel PTV map. If cfg.ptv_scale_by_dose is False: binary union of
    all given PTV-level masks. If True: each level's voxels are set to
    prescription_dose / cfg.ptv_max_dose_gy (e.g. PTV69.96/PTV60/PTV54 all
    present at once) -- where levels overlap (they're typically nested), the
    higher dose level wins.
    """
    ref_shape = next(iter(ptv_masks.values())).shape
    channel = np.zeros(ref_shape, dtype=np.float32)
    if cfg.ptv_scale_by_dose:
        for name, mask in ptv_masks.items():
            level_value = prescription_doses.get(name, 0.0) / cfg.ptv_max_dose_gy
            channel = np.where(mask > 0, np.maximum(channel, level_value), channel)
    else:
        for mask in ptv_masks.values():
            channel = np.where(mask > 0, 1.0, channel)
    return channel.astype(np.float32)


def build_oar_channel(oar_masks: Dict[str, np.ndarray],
                       label_map: Dict[str, int],
                       cfg: PreprocessConfig) -> np.ndarray:
    """
    Integer-encoded (default) or one-hot OAR channel(s). Masks whose name
    isn't in `label_map` are silently skipped (GDP-HMM structure naming is
    heterogeneous -- callers should have already resolved names to this
    map's keys). Overlapping OARs: integer mode keeps the last-assigned
    label at a voxel (arbitrary but deterministic given dict order); one-hot
    mode keeps all labels independently.
    """
    ref_shape = next(iter(oar_masks.values())).shape
    if cfg.oar_one_hot:
        num_labels = max(label_map.values())
        channel = np.zeros((num_labels,) + ref_shape, dtype=np.float32)
        for name, mask in oar_masks.items():
            if name not in label_map:
                continue
            idx = label_map[name] - 1
            channel[idx] = np.maximum(channel[idx], (mask > 0).astype(np.float32))
        return channel
    channel = np.zeros(ref_shape, dtype=np.int64)
    for name, mask in oar_masks.items():
        if name not in label_map:
            continue
        channel = np.where(mask > 0, label_map[name], channel)
    return channel.astype(np.float32)


def normalize_dose(dose_gy: np.ndarray, prescription_dose_gy: float, cfg: PreprocessConfig) -> np.ndarray:
    if cfg.dose_normalize_by_prescription:
        return (dose_gy / prescription_dose_gy).astype(np.float32)
    return dose_gy.astype(np.float32)


def apply_body_mask(dose: np.ndarray, body_mask: np.ndarray) -> np.ndarray:
    """Zero dose outside the patient's Body contour before it's used as a
    training label -- avoids training the model to fit meaningless dose
    values in air/outside-patient voxels (resampling/interpolation can
    otherwise leak small nonzero values across the body boundary).
    Convention from gdp_hmm_reference/full_source/data_loader.py."""
    return (dose * (body_mask > 0)).astype(np.float32)


def rescale_dose_to_prescription_d97(dose: np.ndarray, ptv_high_mask: np.ndarray,
                                      prescription_dose_gy: float, clip_factor: float = 1.2,
                                      eps: float = 1e-5) -> np.ndarray:
    """
    Rescale dose so D97 (the dose received by 97% of the highest-Rx PTV's
    volume, i.e. the 3rd percentile of dose within that structure) exactly
    equals its prescribed dose -- corrects for the fact that raw dose
    doesn't necessarily land exactly on Rx due to planning/optimization
    slop. Then clips to [0, Rx * clip_factor] to guard against extreme
    hot-spot outliers dominating the rescale. Convention from
    gdp_hmm_reference/full_source/data_loader.py's `MyDataset.__getitem__`.

    If `ptv_high_mask` has no voxels, dose is returned unrescaled (can't
    anchor to a percentile of nothing) -- not an error, since a slice/crop
    might genuinely miss the PTV.
    """
    voxel_doses = dose[ptv_high_mask > 0]
    if voxel_doses.size == 0:
        return dose.astype(np.float32)
    d97 = np.percentile(voxel_doses, 3)
    scale = prescription_dose_gy / (d97 + eps)
    rescaled = dose * scale
    return np.clip(rescaled, 0.0, prescription_dose_gy * clip_factor).astype(np.float32)


# --------------------------------------------------------------------------- #
# In-plane-only resampling + fixed-depth crop (real-data shape adaptation)
# --------------------------------------------------------------------------- #

def resample_inplane_only(volume_zhw: np.ndarray, spacing: Tuple[float, float, float],
                           target_hw: Tuple[int, int], order: int) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Resample only the in-plane (H, W) axes of a [Z, H, W] volume to
    `target_hw`, leaving the Z axis (native slice thickness) untouched --
    CRITICAL per spec: do NOT make the z-axis isotropic like
    dvhnet/preprocessing.py's pipeline does (that pipeline resamples all
    three axes to a uniform target_slice_thickness, which is wrong for
    dosenet3d). Reuses dvhnet.preprocessing.resample_volume by pinning the
    destination z-spacing equal to the source (zoom factor 1 on that axis)
    rather than reimplementing the interpolation logic.
    """
    dz, dy, dx = spacing
    z, h, w = volume_zhw.shape
    dst_spacing = (dz, dy * h / target_hw[0], dx * w / target_hw[1])
    dst_shape = (z, target_hw[0], target_hw[1])
    resampled = _dvhnet_resample_volume(volume_zhw, spacing, dst_spacing, dst_shape=dst_shape, order=order)
    return resampled, dst_spacing


def ptv_center_z_index(ptv_mask_hwd: np.ndarray) -> int:
    """Index (along the last, D/z axis) of the PTV's midpoint; falls back to
    the volume's center if no PTV voxels are present."""
    z_indices = np.nonzero(ptv_mask_hwd.sum(axis=(0, 1)))[0]
    if len(z_indices) == 0:
        return ptv_mask_hwd.shape[-1] // 2
    return int(round((int(z_indices.min()) + int(z_indices.max())) / 2))


def crop_or_pad_depth(volume_hwd: np.ndarray, center_index: int, depth: int) -> np.ndarray:
    """Extract a fixed-length window of `depth` slices along the last (D/z)
    axis, centered on `center_index`; zero-pads if the volume is shorter
    than `depth` or the window runs off either end."""
    h, w, d = volume_hwd.shape
    out = np.zeros((h, w, depth), dtype=volume_hwd.dtype)
    start = center_index - depth // 2
    src_start = max(0, start)
    src_end = min(d, start + depth)
    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)
    if src_end > src_start:
        out[:, :, dst_start:dst_end] = volume_hwd[:, :, src_start:src_end]
    return out


# --------------------------------------------------------------------------- #
# Shared preprocessing core (pure arrays -- real-data-ready today)
# --------------------------------------------------------------------------- #

def preprocess_arrays(ct_hu_zhw: np.ndarray,
                       ptv_masks_zhw: Dict[str, np.ndarray],
                       oar_masks_zhw: Dict[str, np.ndarray],
                       dose_gy_zhw: np.ndarray,
                       spacing: Tuple[float, float, float],
                       cfg: PreprocessConfig,
                       prescription_doses: Dict[str, float],
                       oar_label_map: Dict[str, int] = None,
                       body_mask_zhw: Optional[np.ndarray] = None,
                       ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    The single place the windowing / channel-construction / in-plane
    resample / depth-crop logic lives. Takes already-loaded [Z, H, W]
    CT/dose/mask arrays -- however they were loaded (DICOM via
    dvhnet.preprocessing today, or gdp_hmm_adapter.py) -- and returns the
    final [C, H, W, D] input tensor, [1, H, W, D] target tensor, and a
    {structure_name: [H, W, D] binary} dict of every PTV/OAR mask at the
    SAME post-resample/post-crop grid as the input/target -- callers (the
    GDP-HMM adapter's Dataset) need this exact alignment for
    losses.RegionWeightedOrganLoss, and recomputing it separately would
    risk drifting out of sync with the resample/crop done here.

    `body_mask_zhw` is optional: when given, dose outside it is zeroed
    (apply_body_mask) before use as the target. When
    `cfg.dose_rescale_to_prescription_d97` is set, dose is additionally
    rescaled so D97 in the highest-Rx PTV (the max of `prescription_doses`)
    matches its prescribed dose. Both run before `normalize_dose`.
    """
    oar_label_map = oar_label_map or DEFAULT_OAR_LABEL_MAP

    ct_resampled, _ = resample_inplane_only(ct_hu_zhw, spacing, cfg.target_hw, cfg.resample_order_ct)
    dose_resampled, _ = resample_inplane_only(dose_gy_zhw, spacing, cfg.target_hw, cfg.resample_order_dose)
    ptv_resampled = {
        name: (resample_inplane_only(m.astype(np.float32), spacing, cfg.target_hw, cfg.resample_order_mask)[0] > 0.5).astype(np.uint8)
        for name, m in ptv_masks_zhw.items()
    }
    oar_resampled = {
        name: (resample_inplane_only(m.astype(np.float32), spacing, cfg.target_hw, cfg.resample_order_mask)[0] > 0.5).astype(np.uint8)
        for name, m in oar_masks_zhw.items()
    }
    body_resampled = None
    if body_mask_zhw is not None:
        body_resampled = (resample_inplane_only(body_mask_zhw.astype(np.float32), spacing, cfg.target_hw,
                                                  cfg.resample_order_mask)[0] > 0.5).astype(np.uint8)

    # [Z, H, W] -> [H, W, D]
    ct_hwd = np.transpose(ct_resampled, (1, 2, 0))
    dose_hwd = np.transpose(dose_resampled, (1, 2, 0))
    ptv_hwd = {name: np.transpose(m, (1, 2, 0)) for name, m in ptv_resampled.items()}
    oar_hwd = {name: np.transpose(m, (1, 2, 0)) for name, m in oar_resampled.items()}
    body_hwd = np.transpose(body_resampled, (1, 2, 0)) if body_resampled is not None else None

    union_ptv = np.zeros(ct_hwd.shape, dtype=np.uint8)
    for m in ptv_hwd.values():
        union_ptv = np.maximum(union_ptv, m)
    center_z = ptv_center_z_index(union_ptv)

    ct_hwd = crop_or_pad_depth(ct_hwd, center_z, cfg.target_depth)
    dose_hwd = crop_or_pad_depth(dose_hwd, center_z, cfg.target_depth)
    ptv_hwd = {name: crop_or_pad_depth(m, center_z, cfg.target_depth) for name, m in ptv_hwd.items()}
    oar_hwd = {name: crop_or_pad_depth(m, center_z, cfg.target_depth) for name, m in oar_hwd.items()}
    if body_hwd is not None:
        body_hwd = crop_or_pad_depth(body_hwd, center_z, cfg.target_depth)
        dose_hwd = apply_body_mask(dose_hwd, body_hwd)

    max_prescription = max(prescription_doses.values()) if prescription_doses else 1.0
    if cfg.dose_rescale_to_prescription_d97 and prescription_doses:
        ptv_high_name = max(prescription_doses, key=prescription_doses.get)
        if ptv_high_name in ptv_hwd:
            dose_hwd = rescale_dose_to_prescription_d97(dose_hwd, ptv_hwd[ptv_high_name],
                                                         prescription_doses[ptv_high_name],
                                                         cfg.dose_rescale_clip_factor)

    ct_channel = window_and_normalize_ct(ct_hwd, cfg)
    ptv_channel = build_ptv_channel(ptv_hwd, prescription_doses, cfg)
    oar_channel = build_oar_channel(oar_hwd, oar_label_map, cfg)

    channels = [ct_channel, ptv_channel]
    channels += list(oar_channel) if oar_channel.ndim == 4 else [oar_channel]
    input_tensor = np.stack(channels, axis=0).astype(np.float32)

    dose_final = normalize_dose(dose_hwd, max_prescription, cfg)
    target_tensor = dose_final[None, ...].astype(np.float32)

    masks_hwd = {name: m.astype(np.float32) for name, m in {**ptv_hwd, **oar_hwd}.items()}

    return input_tensor, target_tensor, masks_hwd


# --------------------------------------------------------------------------- #
# Real-data entry point -- NOT YET IMPLEMENTED
# --------------------------------------------------------------------------- #

def load_and_preprocess_patient(ct_dir: str, rtstruct_path: str, rtdose_path: str,
                                 cfg: PreprocessConfig,
                                 prescription_doses: Dict[str, float],
                                 oar_label_map: Dict[str, int] = None,
                                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Real-data entry point: DICOM -> [C, H, W, D] input tensor + [1, H, W, D]
    dose target.

    NOT YET IMPLEMENTED -- GDP-HMM Hugging Face dataset access is still
    pending approval. When it lands, do NOT reimplement DICOM/RTSTRUCT/
    RTDOSE loading here -- reuse dvhnet/preprocessing.py's existing helpers:
        - dvhnet.preprocessing.load_ct_series       CT dir -> ([Z,H,W] HU, spacing, origin, slices)
        - dvhnet.preprocessing.load_rtdose          RTDOSE -> ([Z,H,W] Gy*, spacing, origin)  (*raw grid; multiply by DoseGridScaling if loading GDP-HMM's own npz format instead, see note below)
        - dvhnet.preprocessing.load_rtstruct_masks  RTSTRUCT -> {name: [Z,H,W] binary}
        - dvhnet.preprocessing.align_dose_to_ct     origin-aware dose->CT alignment (fixed bug: see dvhnet/README.md)
    then feed the results into `preprocess_arrays` above (which is already
    fully implemented and real-data-ready) rather than duplicating any of
    its windowing/channel/resample/crop logic.

    Note: GDP-HMM's actual distribution format is a pre-packaged per-patient
    .npz dict (documented in GDP-HMM_AAPMChallenge/data_visual_understand.ipynb),
    not raw DICOM -- the GDP-HMM-specific adapter (built once HF access is
    approved) will most likely skip this DICOM path entirely and call
    `preprocess_arrays` directly with arrays read straight out of that npz.
    This function stays as the generic DICOM fallback for the case where you
    have your own DICOM data instead of/in addition to GDP-HMM's npz files.
    """
    raise NotImplementedError(
        "DICOM loading not wired up yet -- GDP-HMM Hugging Face access is "
        "still pending. See this function's docstring for the "
        "dvhnet.preprocessing helpers to call once real data is available; "
        "preprocess_arrays() in this module is already implemented and "
        "ready to receive whatever arrays that loading step produces."
    )
