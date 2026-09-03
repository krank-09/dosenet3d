"""
synthetic_data.py
==================
SMOKE-TEST DATA ONLY. Nothing in this file reads or approximates real
DICOM/GDP-HMM data -- it fabricates shape-correct volumes with plausible
anatomical *structure* (a blob-shaped PTV, a few nearby OAR blobs, a dose
field that falls off with distance from the PTV and dips inside the OARs)
purely so dosenet3d's model/loss/training code can be exercised end-to-end
before real GDP-HMM data is available (Hugging Face access is still
pending). Do not draw any conclusions about model *quality* from results on
this data -- it exists to catch shape and wiring bugs, not to validate
dose-prediction accuracy.

Tensor layout matches data_pipeline.py's real-data spec exactly:
    input:  [C, H, W, D] = [3, 256, 256, 80]
            channel 0 = normalized CT in [0, 1]
            channel 1 = PTV channel (binary by default; dose-scaled if
                        cfg.ptv_scale_by_dose)
            channel 2 = OAR channel (integer-encoded by default; one-hot
                        stack -- extra channels -- if cfg.oar_one_hot)
    target: [1, H, W, D], dose in Gy (or normalized, if
            cfg.dose_normalize_by_prescription), always >= 0

Channel construction is delegated to data_pipeline.py's
window_and_normalize_ct / build_ptv_channel / build_oar_channel /
normalize_dose -- the exact same functions the real GDP-HMM adapter will
call -- so this smoke test exercises real preprocessing code, not a
parallel reimplementation of it. Only the blob/dose fabrication below is
synthetic-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from data_pipeline import (
    DEFAULT_OAR_LABEL_MAP,
    PreprocessConfig,
    build_oar_channel,
    build_ptv_channel,
    total_input_channels,
    window_and_normalize_ct,
)

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object

H, W, D = 256, 256, 80


@dataclass
class SyntheticExample:
    patient_id: str
    input: np.ndarray                  # [C, H, W, D] float32
    target: np.ndarray                 # [1, H, W, D] float32, Gy (or normalized)
    ptv_masks: Dict[str, np.ndarray]   # each [H, W, D] uint8, for eval/debugging
    oar_masks: Dict[str, np.ndarray]   # each [H, W, D] uint8
    prescription_dose_gy: float


def _ellipsoid_mask(center: Tuple[int, int, int], radii: Tuple[int, int, int],
                     shape: Tuple[int, int, int] = (H, W, D)) -> np.ndarray:
    hh, ww, dd = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij")
    ch, cw, cd = center
    rh, rw, rd = radii
    val = ((hh - ch) / rh) ** 2 + ((ww - cw) / rw) ** 2 + ((dd - cd) / rd) ** 2
    return (val <= 1.0).astype(np.uint8)


def _ellipsoid_distance(center: Tuple[int, int, int], radii: Tuple[int, int, int],
                         shape: Tuple[int, int, int] = (H, W, D)) -> np.ndarray:
    """Normalized ellipsoidal distance field (0 at center, 1 at the given
    radii, growing beyond) -- used to fabricate a dose field that falls off
    away from the PTV rather than being pure noise."""
    hh, ww, dd = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij")
    ch, cw, cd = center
    rh, rw, rd = radii
    return np.sqrt(((hh - ch) / rh) ** 2 + ((ww - cw) / rw) ** 2 + ((dd - cd) / rd) ** 2)


def generate_patient(patient_id: str, cfg: PreprocessConfig, seed: int) -> SyntheticExample:
    rng = np.random.RandomState(seed)

    # --- PTV: one ellipsoid blob roughly centered in the volume ---
    ptv_center = (
        H // 2 + rng.randint(-20, 20),
        W // 2 + rng.randint(-20, 20),
        D // 2 + rng.randint(-8, 8),
    )
    ptv_radii = (
        int(rng.randint(25, 40)),
        int(rng.randint(25, 40)),
        int(rng.randint(10, 18)),
    )
    ptv_mask = _ellipsoid_mask(ptv_center, ptv_radii)
    prescription_dose_gy = float(rng.choice([54.0, 60.0, 66.0, 70.0]))
    ptv_masks = {"PTV": ptv_mask}

    # --- OARs: a few smaller blobs placed around the PTV ---
    oar_masks: Dict[str, np.ndarray] = {}
    oar_layout = [
        ("Brainstem", (-1.0, 0.0, 0.0), (10, 10, 10)),
        ("SpinalCord", (0.0, 0.0, -1.0), (6, 6, 25)),
        ("Parotid_L", (0.6, 0.9, 0.1), (14, 14, 12)),
        ("Parotid_R", (0.6, -0.9, 0.1), (14, 14, 12)),
    ]
    shape = (H, W, D)
    for name, direction, radii in oar_layout:
        jitter = rng.randint(-6, 7, size=3)
        raw_center = [
            ptv_center[i] + direction[i] * (ptv_radii[i] + radii[i] + 8) + jitter[i]
            for i in range(3)
        ]
        center = tuple(
            int(np.clip(raw_center[i], radii[i] + 2, shape[i] - radii[i] - 2))
            for i in range(3)
        )
        oar_masks[name] = _ellipsoid_mask(center, radii)

    # --- pseudo-CT: air background + soft-tissue body + slightly denser PTV,
    # normalized with the SAME windowing function the real pipeline uses ---
    body_mask = _ellipsoid_mask((H // 2, W // 2, D // 2), (110, 110, 36))
    ct_hu = np.full(shape, -1000.0, dtype=np.float32)
    ct_hu[body_mask > 0] = 40.0
    ct_hu += rng.normal(0, 15, size=shape).astype(np.float32) * body_mask
    ct_hu[ptv_mask > 0] = 60.0 + float(rng.normal(0, 10))
    ct_channel = window_and_normalize_ct(ct_hu, cfg)

    # --- PTV / OAR channels via the shared, real-pipeline functions ---
    ptv_channel = build_ptv_channel(ptv_masks, {"PTV": prescription_dose_gy}, cfg)
    oar_channel = build_oar_channel(oar_masks, DEFAULT_OAR_LABEL_MAP, cfg)

    channels = [ct_channel, ptv_channel]
    channels += list(oar_channel) if oar_channel.ndim == 4 else [oar_channel]
    input_tensor = np.stack(channels, axis=0).astype(np.float32)

    # --- dose: hot and near-uniform inside the PTV, falls off with distance
    # outside it, dips inside OARs (sparing), plus small noise. Serial OARs
    # (brainstem/cord) get sparer relative doses than parotids, so the
    # region-weighted loss has non-trivial, structure-dependent signal. ---
    dist = _ellipsoid_distance(ptv_center, tuple(r + 6 for r in ptv_radii), shape)
    dose = prescription_dose_gy * np.exp(-np.clip(dist - 1.0, 0.0, None) * 0.9)
    ptv_noise = 0.97 + 0.03 * rng.random(size=int(ptv_mask.sum()))
    dose[ptv_mask > 0] = prescription_dose_gy * ptv_noise
    sparing_factor = {"Brainstem": 0.30, "SpinalCord": 0.35, "Parotid_L": 0.55, "Parotid_R": 0.55}
    for name, mask in oar_masks.items():
        dose[mask > 0] *= sparing_factor.get(name, 0.5)
    dose += rng.normal(0, 0.5, size=shape).astype(np.float32)
    dose = np.clip(dose, 0.0, None).astype(np.float32)

    from data_pipeline import normalize_dose
    dose_final = normalize_dose(dose, prescription_dose_gy, cfg)
    target_tensor = dose_final[None, ...].astype(np.float32)

    return SyntheticExample(
        patient_id=patient_id,
        input=input_tensor,
        target=target_tensor,
        ptv_masks=ptv_masks,
        oar_masks=oar_masks,
        prescription_dose_gy=prescription_dose_gy,
    )


def generate_synthetic_cohort(n_patients: int = 6, cfg: Optional[PreprocessConfig] = None,
                               seed: int = 0) -> List[SyntheticExample]:
    cfg = cfg or PreprocessConfig()
    return [generate_patient(f"SYN{idx:03d}", cfg, seed=seed + idx) for idx in range(n_patients)]


class SyntheticDoseDataset(Dataset):
    """torch Dataset wrapping generate_synthetic_cohort -- smoke-test only,
    see module docstring."""

    def __init__(self, n_patients: int = 6, cfg: Optional[PreprocessConfig] = None, seed: int = 0):
        if torch is None:  # pragma: no cover
            raise ImportError("torch is required for SyntheticDoseDataset")
        self.cfg = cfg or PreprocessConfig()
        self.examples = generate_synthetic_cohort(n_patients, self.cfg, seed)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        ex = self.examples[idx]
        # `masks` carries each structure's binary mask independently of the
        # input tensor's encoding (integer-label OAR channel, possibly
        # dose-scaled PTV channel, etc.) -- losses.py's region-weighted term
        # consumes these directly rather than reverse-decoding them from the
        # model input, so it stays correct regardless of cfg.oar_one_hot /
        # cfg.ptv_scale_by_dose. Every synthetic patient has the same
        # structure-name set, so this collates fine with the default
        # DataLoader collate_fn.
        masks = {name: torch.from_numpy(m.astype("float32")) for name, m in
                  {**ex.ptv_masks, **ex.oar_masks}.items()}
        return {
            "input": torch.from_numpy(ex.input),
            "target": torch.from_numpy(ex.target),
            "patient_id": ex.patient_id,
            "masks": masks,
        }


if __name__ == "__main__":
    cfg = PreprocessConfig()
    cohort = generate_synthetic_cohort(n_patients=2, cfg=cfg)
    for ex in cohort:
        print(f"{ex.patient_id}: input={ex.input.shape} target={ex.target.shape} "
              f"Rx={ex.prescription_dose_gy}Gy dose_max={ex.target.max():.2f} "
              f"dose_min={ex.target.min():.2f}")
    print("in_channels (per cfg):", total_input_channels(cfg, len(DEFAULT_OAR_LABEL_MAP)))
