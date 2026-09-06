"""
evaluate.py
===========
Evaluation for DoseNet3D predictions:
  - Voxel-level MAE (Gy) over the predicted volume.
  - DVH-based clinical metrics, reusing dvhnet's DVH math directly rather
    than reimplementing it: dvhnet.preprocessing.compute_slice_cumulative_dvh
    (its boolean-mask-based histogram works identically on a full 3D volume,
    despite the "slice" in its name -- `dose[mask > 0]` flattens regardless
    of dimensionality) to build each structure's cumulative DVH curve, then
    dvhnet.metrics.dose_at_volume/d2/d50/d_mean to invert those curves into
    clinical dose metrics.
  - Per-structure reporting: D95%/D98% target coverage for the PTV, D2%+Dmax
    for serial organs (spinal cord, brainstem, chiasm, optic nerves --
    max-dose violations are the clinically critical failure mode), Dmean for
    parallel organs (parotids).
  - A dose-difference acceptance flag per structure: passes if the error is
    <= 2 Gy OR <= 3% of the prescription dose (whichever is more lenient --
    implemented as err <= max(2.0, 0.03 * Rx), which is exactly equivalent
    to that OR).

Right now this only has synthetic ground truth to check itself against --
real GDP-HMM data access is still pending. Running this file's __main__
confirms the metric code runs end-to-end and produces finite, sane-shaped
numbers; it does NOT mean anything about real accuracy -- the model here is
freshly initialized, not trained.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from losses import DEFAULT_SERIAL_ORGANS

# --------------------------------------------------------------------------- #
# Reuse dvhnet's DVH construction + dose-metric inversion (see data_pipeline.py
# for why this uses importlib under a private module name rather than plain
# `import` -- dvhnet/ has its own model.py/losses.py that would otherwise
# shadow this package's same-named files).
# --------------------------------------------------------------------------- #

def _load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_DVHNET_DIR = Path(__file__).resolve().parent.parent / "dvhnet"
_dvhnet_preprocessing = _load_module_from_path("_dvhnet_preprocessing", _DVHNET_DIR / "preprocessing.py")
_dvhnet_metrics = _load_module_from_path("_dvhnet_metrics", _DVHNET_DIR / "metrics.py")


def structure_dvh(dose: np.ndarray, mask: np.ndarray, dose_max_gy: float, num_bins: int = 256) -> Optional[np.ndarray]:
    """Cumulative DVH curve for one structure's voxels, via dvhnet's own
    per-slice DVH function applied to the whole 3D volume at once."""
    cfg = _dvhnet_preprocessing.PreprocessConfig(dose_max_gy=dose_max_gy, num_bins=num_bins)
    return _dvhnet_preprocessing.compute_slice_cumulative_dvh(dose, mask, cfg)


def voxel_mae(pred_dose: np.ndarray, true_dose: np.ndarray) -> float:
    """Whole-volume voxel-level MAE, in Gy."""
    return float(np.mean(np.abs(pred_dose - true_dose)))


@dataclass
class StructureReport:
    structure: str
    kind: str          # "ptv" | "serial" | "parallel"
    metric: str         # e.g. "D95%", "D2%", "Dmax", "Dmean"
    predicted_gy: float
    ground_truth_gy: float
    abs_error_gy: float
    within_tolerance: bool


def _tolerance(prescription_dose_gy: float, abs_threshold_gy: float = 2.0, pct_threshold: float = 0.03) -> float:
    """err <= max(abs_threshold, pct_threshold * Rx) is exactly equivalent
    to 'err <= abs_threshold OR err <= pct_threshold * Rx'."""
    return max(abs_threshold_gy, pct_threshold * prescription_dose_gy)


def evaluate_structure(name: str, pred_dose: np.ndarray, true_dose: np.ndarray, mask: np.ndarray,
                        dose_max_gy: float, prescription_dose_gy: float) -> List[StructureReport]:
    if mask.sum() == 0:
        return []

    tol = _tolerance(prescription_dose_gy)
    reports: List[StructureReport] = []

    if name.upper().startswith("PTV"):
        pred_dvh = structure_dvh(pred_dose, mask, dose_max_gy)
        true_dvh = structure_dvh(true_dose, mask, dose_max_gy)
        for percent, label in [(95.0, "D95%"), (98.0, "D98%")]:
            pv = _dvhnet_metrics.dose_at_volume(pred_dvh, percent, dose_max_gy)
            tv = _dvhnet_metrics.dose_at_volume(true_dvh, percent, dose_max_gy)
            err = abs(pv - tv)
            reports.append(StructureReport(name, "ptv", label, pv, tv, err, err <= tol))

    elif name in DEFAULT_SERIAL_ORGANS:
        pred_dvh = structure_dvh(pred_dose, mask, dose_max_gy)
        true_dvh = structure_dvh(true_dose, mask, dose_max_gy)
        pv_d2 = _dvhnet_metrics.d2(pred_dvh, dose_max_gy)
        tv_d2 = _dvhnet_metrics.d2(true_dvh, dose_max_gy)
        err_d2 = abs(pv_d2 - tv_d2)
        reports.append(StructureReport(name, "serial", "D2%", pv_d2, tv_d2, err_d2, err_d2 <= tol))

        pv_max = float(pred_dose[mask > 0].max())
        tv_max = float(true_dose[mask > 0].max())
        err_max = abs(pv_max - tv_max)
        reports.append(StructureReport(name, "serial", "Dmax", pv_max, tv_max, err_max, err_max <= tol))

    else:
        pred_dvh = structure_dvh(pred_dose, mask, dose_max_gy)
        true_dvh = structure_dvh(true_dose, mask, dose_max_gy)
        pv = _dvhnet_metrics.d_mean(pred_dvh, dose_max_gy)
        tv = _dvhnet_metrics.d_mean(true_dvh, dose_max_gy)
        err = abs(pv - tv)
        reports.append(StructureReport(name, "parallel", "Dmean", pv, tv, err, err <= tol))

    return reports


def evaluate_patient(pred_dose: np.ndarray, true_dose: np.ndarray, masks: Dict[str, np.ndarray],
                      dose_max_gy: float, prescription_dose_gy: float) -> Dict:
    """pred_dose, true_dose: [H, W, D] (or [1, H, W, D] -- squeezed if so).
    masks: {structure_name: [H, W, D] binary}."""
    pred_dose = np.asarray(pred_dose)
    true_dose = np.asarray(true_dose)
    if pred_dose.ndim == 4:
        pred_dose = pred_dose[0]
    if true_dose.ndim == 4:
        true_dose = true_dose[0]

    reports: List[StructureReport] = []
    for name, mask in masks.items():
        reports.extend(evaluate_structure(name, pred_dose, true_dose, mask, dose_max_gy, prescription_dose_gy))

    return {
        "voxel_mae_gy": voxel_mae(pred_dose, true_dose),
        "structures": reports,
    }


def print_report(result: Dict) -> None:
    print(f"voxel_mae_gy: {result['voxel_mae_gy']:.4f}")
    print(f"{'structure':15s} {'kind':9s} {'metric':6s} {'pred':>8s} {'truth':>8s} {'err':>8s}  ok")
    for r in result["structures"]:
        flag = "OK" if r.within_tolerance else "FAIL"
        print(f"{r.structure:15s} {r.kind:9s} {r.metric:6s} {r.predicted_gy:8.2f} "
              f"{r.ground_truth_gy:8.2f} {r.abs_error_gy:8.2f}  {flag}")


if __name__ == "__main__":
    import torch

    from data_pipeline import PreprocessConfig
    from model import DoseNet3D
    from synthetic_data import generate_patient

    print("=== SYNTHETIC SANITY CHECK -- confirms the metric code runs and produces ===")
    print("=== finite, sane-shaped numbers. The model below is UNTRAINED (random  ===")
    print("=== init), so the actual error values mean nothing about real accuracy. ===\n")

    cfg = PreprocessConfig()
    ex = generate_patient("SYN_EVAL", cfg, seed=123)

    model = DoseNet3D(in_channels=3)
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(ex.input).unsqueeze(0))[0, 0].numpy()

    true_dose = ex.target[0]
    masks = {**ex.ptv_masks, **ex.oar_masks}

    result = evaluate_patient(pred, true_dose, masks, dose_max_gy=80.0,
                               prescription_dose_gy=ex.prescription_dose_gy)
    print_report(result)

    assert np.isfinite(result["voxel_mae_gy"])
    assert all(np.isfinite(r.abs_error_gy) for r in result["structures"])
    assert len(result["structures"]) > 0
    print("\nEVALUATE.PY SANITY CHECK OK")
