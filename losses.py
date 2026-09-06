"""
losses.py
=========
Composite DoseNet3D training objective:

    L_total = L_voxel + lambda1 * L_gradient + lambda2 * L_organ

- L_voxel:    whole-volume L1 (or Smooth L1, flag) between predicted and
              ground-truth dose.
- L_gradient: L1 distance between finite-difference spatial gradients of
              predicted vs ground-truth dose, summed over the H, W, and D
              axes -- forces sharp PTV/OAR boundary falloff instead of
              letting the model smooth it away (a plain voxel loss alone
              tends to blur sharp gradients since it doesn't penalize
              *where* the error is spatially structured).
- L_organ:    region-weighted L1, applied separately inside the PTV and
              each named OAR, with per-structure weights (higher for
              serial organs -- spinal cord, brainstem -- where a
              max-dose violation is clinically much worse than a
              similar-magnitude error in a parallel organ like a parotid).

All three components are returned separately (not just the total) so
train.py can log them independently -- if e.g. loss_mono-equivalent terms
stall while loss_voxel keeps moving, that's diagnostic, not noise.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Serial organs: a max-dose (not mean-dose) violation is the clinically
# critical failure mode, so they default to the high end of the weight
# range even if not explicitly listed in `structure_weights`.
#
# Includes both "Brainstem" (data_pipeline.DEFAULT_OAR_LABEL_MAP /
# synthetic_data.py's naming, following dvhnet's original convention) and
# "BrainStem" (the real GDP-HMM naming found in dvhnet/gdp_hmm_adapter.py,
# e.g. "BrainStem_03") -- these two tracks use different casing for the same
# structure and this set has to match both, or synthetic data silently gets
# the wrong (parallel) weight, which is exactly what happened here before
# this was caught.
DEFAULT_SERIAL_ORGANS = {"SpinalCord", "SpinalCord_05", "Brainstem", "BrainStem", "BrainStem_03",
                          "Chiasm", "OpticNerve_L", "OpticNerve_R", "OpticChiasm"}
DEFAULT_SERIAL_WEIGHT = 5.0
DEFAULT_PARALLEL_WEIGHT = 2.0
DEFAULT_PTV_WEIGHT = 3.0


def _grad_l1(pred: torch.Tensor, target: torch.Tensor, dim: int) -> torch.Tensor:
    """Mean L1 distance between finite-difference gradients along `dim`."""
    pred_grad = pred.narrow(dim, 1, pred.shape[dim] - 1) - pred.narrow(dim, 0, pred.shape[dim] - 1)
    target_grad = target.narrow(dim, 1, target.shape[dim] - 1) - target.narrow(dim, 0, target.shape[dim] - 1)
    return (pred_grad - target_grad).abs().mean()


class GradientLoss(nn.Module):
    """Sum of finite-difference gradient L1 error over the H, W, D axes.
    Expects [B, 1, H, W, D] tensors."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return _grad_l1(pred, target, 2) + _grad_l1(pred, target, 3) + _grad_l1(pred, target, 4)


class RegionWeightedOrganLoss(nn.Module):
    """
    Region-weighted L1: for each named structure mask, computes the mean L1
    error restricted to that structure's voxels and scales it by that
    structure's weight, then sums across structures.

    `structure_weights` is a config dict keyed by structure name (matching
    the keys of the `masks` dict passed to forward -- i.e. dataset/adapter
    structure names, e.g. "PTV", "SpinalCord", "Parotids"). Names not in
    `structure_weights` fall back to DEFAULT_SERIAL_WEIGHT if they're in
    `serial_organs`, else DEFAULT_PARALLEL_WEIGHT ("PTV" itself falls back
    to DEFAULT_PTV_WEIGHT). A structure with zero voxels in a given sample
    contributes nothing (guarded, not a divide-by-zero).
    """

    def __init__(self, structure_weights: Optional[Dict[str, float]] = None,
                 serial_organs: Iterable[str] = DEFAULT_SERIAL_ORGANS,
                 serial_weight: float = DEFAULT_SERIAL_WEIGHT,
                 parallel_weight: float = DEFAULT_PARALLEL_WEIGHT,
                 ptv_weight: float = DEFAULT_PTV_WEIGHT):
        super().__init__()
        self.structure_weights = structure_weights or {}
        self.serial_organs = set(serial_organs)
        self.serial_weight = serial_weight
        self.parallel_weight = parallel_weight
        self.ptv_weight = ptv_weight

    def _weight_for(self, name: str) -> float:
        if name in self.structure_weights:
            return self.structure_weights[name]
        if name.upper().startswith("PTV"):
            return self.ptv_weight
        if name in self.serial_organs:
            return self.serial_weight
        return self.parallel_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                masks: Dict[str, torch.Tensor]) -> torch.Tensor:
        """pred, target: [B, 1, H, W, D]. masks: {name: [B, H, W, D]} binary."""
        total = pred.new_zeros(())
        abs_err = (pred - target).abs()  # [B, 1, H, W, D]
        for name, mask in masks.items():
            m = mask.unsqueeze(1) if mask.dim() == abs_err.dim() - 1 else mask  # -> [B,1,H,W,D]
            voxel_count = m.sum()
            if voxel_count.item() == 0:
                continue
            region_l1 = (abs_err * m).sum() / voxel_count
            total = total + self._weight_for(name) * region_l1
        return total


class CompositeDoseLoss(nn.Module):
    def __init__(self, lambda_gradient: float = 0.5, lambda_organ: float = 0.3,
                 use_smooth_l1: bool = False,
                 structure_weights: Optional[Dict[str, float]] = None,
                 serial_organs: Iterable[str] = DEFAULT_SERIAL_ORGANS,
                 serial_weight: float = DEFAULT_SERIAL_WEIGHT,
                 parallel_weight: float = DEFAULT_PARALLEL_WEIGHT,
                 ptv_weight: float = DEFAULT_PTV_WEIGHT):
        super().__init__()
        self.lambda_gradient = lambda_gradient
        self.lambda_organ = lambda_organ
        self.use_smooth_l1 = use_smooth_l1
        self.gradient_loss = GradientLoss()
        self.organ_loss = RegionWeightedOrganLoss(structure_weights, serial_organs,
                                                    serial_weight, parallel_weight, ptv_weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.use_smooth_l1:
            l_voxel = F.smooth_l1_loss(pred, target, reduction="mean")
        else:
            l_voxel = F.l1_loss(pred, target, reduction="mean")

        l_gradient = self.gradient_loss(pred, target)
        l_organ = self.organ_loss(pred, target, masks)

        total = l_voxel + self.lambda_gradient * l_gradient + self.lambda_organ * l_organ

        return {
            "loss": total,
            "loss_voxel": l_voxel.detach(),
            "loss_gradient": l_gradient.detach(),
            "loss_organ": l_organ.detach(),
        }


if __name__ == "__main__":
    import torch as _torch
    from synthetic_data import SyntheticDoseDataset

    ds = SyntheticDoseDataset(n_patients=1)
    batch = ds[0]
    pred = _torch.rand_like(batch["target"]).unsqueeze(0) * 70.0
    target = batch["target"].unsqueeze(0)
    masks = {k: v.unsqueeze(0) for k, v in batch["masks"].items()}

    criterion = CompositeDoseLoss()
    out = criterion(pred, target, masks)
    for k, v in out.items():
        print(f"{k:15s} {v.item():.4f}")
    assert all(_torch.isfinite(v) for v in out.values())
    print("LOSS SMOKE TEST OK")
