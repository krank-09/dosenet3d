"""
train.py
========
DoseNet3D training script. Right now this can only be smoke-tested against
synthetic_data.py's fabricated cohort (real GDP-HMM data access is still
pending) -- a short synthetic run validates that the architecture and
training loop execute cleanly (forward, backward, loss components finite
and moving, checkpoints saved). It does NOT validate model quality; do not
read anything into the loss values beyond "did they stay finite and trend
down a bit."

Usage:
    python train.py --epochs 3 --batch_size 1 --n_synthetic_patients 4
"""

from __future__ import annotations

import argparse
import json
import math
import os

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data_pipeline import PreprocessConfig, DEFAULT_OAR_LABEL_MAP
from losses import CompositeDoseLoss
from model import DoseNet3D, fold_depth_into_batch, unfold_batch_to_depth, print_model_summary
from synthetic_data import SyntheticDoseDataset


# --------------------------------------------------------------------------- #
# In-plane-only augmentation
# --------------------------------------------------------------------------- #
#
# CRITICAL: a horizontal flip mirrors patient left<->right. Any lateralized
# structure baked into the integer-encoded OAR channel (e.g. Parotid_L=3,
# Parotid_R=4 in DEFAULT_OAR_LABEL_MAP) must have its label value swapped
# post-flip, or the flipped mask keeps the WRONG side's label -- this bug is
# invisible by eye (the flip still "looks right") and only corrupts the
# label channel. OAR_LABEL_SWAP below is exactly that swap table, derived
# from DEFAULT_OAR_LABEL_MAP; if you add more lateralized OARs to the label
# map, add their swap pairs here too.
OAR_LABEL_SWAP = {
    DEFAULT_OAR_LABEL_MAP["Parotid_L"]: DEFAULT_OAR_LABEL_MAP["Parotid_R"],
    DEFAULT_OAR_LABEL_MAP["Parotid_R"]: DEFAULT_OAR_LABEL_MAP["Parotid_L"],
}
OAR_CHANNEL_INDEX = 2  # channel 0=CT, 1=PTV, 2=OAR (integer-encoded, default cfg)


def hflip_batch(input_t: torch.Tensor, target_t: torch.Tensor) -> tuple:
    """Flip the W axis (patient left<->right) and swap lateralized OAR
    labels in the same op so the flipped mask stays anatomically correct."""
    input_flipped = torch.flip(input_t, dims=[3])
    target_flipped = torch.flip(target_t, dims=[3])

    oar = input_flipped[:, OAR_CHANNEL_INDEX].clone()
    swapped = oar.clone()
    for old_label, new_label in OAR_LABEL_SWAP.items():
        swapped[oar == old_label] = new_label
    input_flipped[:, OAR_CHANNEL_INDEX] = swapped

    return input_flipped, target_flipped


def _inplane_affine_theta(batch_size: int, max_rotation_deg: float, max_translate_frac: float,
                           device: torch.device) -> torch.Tensor:
    angles = (torch.rand(batch_size, device=device) * 2 - 1) * max_rotation_deg * math.pi / 180.0
    tx = (torch.rand(batch_size, device=device) * 2 - 1) * max_translate_frac
    ty = (torch.rand(batch_size, device=device) * 2 - 1) * max_translate_frac
    cos, sin = torch.cos(angles), torch.sin(angles)
    theta = torch.zeros(batch_size, 2, 3, device=device)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    theta[:, 1, 2] = ty
    return theta


def _apply_inplane_affine_5d(x: torch.Tensor, theta: torch.Tensor, mode: str) -> torch.Tensor:
    """Apply the SAME (per-patient) in-plane affine to every D slice of a
    [B, C, H, W, D] tensor -- i.e. a single rigid in-plane transform per
    volume, never a different one per slice (that would be an elastic-like
    warp along z, which the spec explicitly excludes) and never anything
    touching the D axis itself. Reuses model.py's batch<->depth fold so the
    slice ordering exactly matches the convention RDSEM relies on."""
    x2d, b, d = fold_depth_into_batch(x)
    theta_rep = theta.repeat_interleave(d, dim=0)
    grid = F.affine_grid(theta_rep, x2d.shape, align_corners=False)
    x2d = F.grid_sample(x2d, grid, mode=mode, padding_mode="zeros", align_corners=False)
    return unfold_batch_to_depth(x2d, b, d)


def rotate_translate_batch(input_t: torch.Tensor, target_t: torch.Tensor,
                            max_rotation_deg: float, max_translate_frac: float) -> tuple:
    """
    In-plane-only rotation + translation, identical across all D slices of
    each volume (no elastic deformation, no z-axis augmentation).

    Uses 'nearest' interpolation for the PTV/OAR channels (indices 1, 2) so
    the integer-encoded OAR labels and binary PTV mask are never blended
    into fractional garbage values by 'bilinear' resampling -- e.g.
    interpolating between OAR label 1 and label 3 could produce 2, which is
    a different, wrong structure. 'bilinear' is used only for the
    continuous CT channel (index 0) and the dose target.
    """
    b = input_t.shape[0]
    theta = _inplane_affine_theta(b, max_rotation_deg, max_translate_frac, input_t.device)

    ct = _apply_inplane_affine_5d(input_t[:, 0:1], theta, mode="bilinear")
    mask_channels = _apply_inplane_affine_5d(input_t[:, 1:], theta, mode="nearest")
    input_aug = torch.cat([ct, mask_channels], dim=1)
    target_aug = _apply_inplane_affine_5d(target_t, theta, mode="bilinear")
    return input_aug, target_aug


def augment_batch(input_t: torch.Tensor, target_t: torch.Tensor, cfg: "AugConfig") -> tuple:
    if cfg.hflip_prob > 0 and torch.rand(1).item() < cfg.hflip_prob:
        input_t, target_t = hflip_batch(input_t, target_t)
    if cfg.affine_prob > 0 and torch.rand(1).item() < cfg.affine_prob:
        input_t, target_t = rotate_translate_batch(input_t, target_t, cfg.max_rotation_deg, cfg.max_translate_frac)
    return input_t, target_t


class AugConfig:
    def __init__(self, max_rotation_deg=5.0, max_translate_frac=0.03, hflip_prob=0.5, affine_prob=0.5):
        self.max_rotation_deg = max_rotation_deg
        self.max_translate_frac = max_translate_frac
        self.hflip_prob = hflip_prob
        self.affine_prob = affine_prob


# --------------------------------------------------------------------------- #
# Train / eval loop
# --------------------------------------------------------------------------- #

def run_epoch(model, loader, criterion, device, optimizer=None, augment: AugConfig = None,
              use_amp: bool = False, scaler=None):
    is_train = optimizer is not None
    model.train(is_train)

    totals = {"loss": 0.0, "loss_voxel": 0.0, "loss_gradient": 0.0, "loss_organ": 0.0}
    n_batches = 0

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            masks = {k: v.to(device) for k, v in batch["masks"].items()}

            if is_train and augment is not None:
                x, y = augment_batch(x, y, augment)

            if use_amp:
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    pred = model(x)
                    losses = criterion(pred, y, masks)
            else:
                pred = model(x)
                losses = criterion(pred, y, masks)

            if is_train:
                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(losses["loss"]).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

            for k in totals:
                totals[k] += losses[k].item()
            n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1, help="memory-heavy architecture; 1 or 2 recommended")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_gradient", type=float, default=0.5)
    parser.add_argument("--lambda_organ", type=float, default=0.3)
    parser.add_argument("--n_synthetic_patients", type=int, default=4,
                         help="synthetic smoke-test cohort size; real data isn't wired up yet")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default="./runs/dosenet3d_smoke")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # NOTE: DeformConv2d (torchvision.ops) has no MPS kernel as of this
    # writing, so on Apple Silicon (this environment) we intentionally stay
    # on CPU rather than silently falling back mid-forward-pass -- CUDA is
    # used when available, CPU otherwise.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_amp = True
    else:
        device = torch.device("cpu")
        use_amp = False
        print("No CUDA GPU available in this environment -- running on CPU. "
              "This smoke test uses a tiny synthetic batch specifically because "
              "full-resolution [3,256,256,80] volumes are slow on CPU; a real "
              "100-epoch run should be done on a CUDA GPU.")

    torch.manual_seed(args.seed)

    print(f"\n=== SYNTHETIC SMOKE TEST -- validates the training loop, NOT model quality ===")
    print(f"({args.n_synthetic_patients} fabricated patients, {args.epochs} epochs, "
          f"batch_size={args.batch_size}, device={device})\n")

    cfg = PreprocessConfig()
    dataset = SyntheticDoseDataset(n_patients=args.n_synthetic_patients, cfg=cfg, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = DoseNet3D(in_channels=3).to(device)
    print_model_summary(model)

    criterion = CompositeDoseLoss(lambda_gradient=args.lambda_gradient, lambda_organ=args.lambda_organ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    augment = AugConfig()

    history = []
    for epoch in range(1, args.epochs + 1):
        stats = run_epoch(model, loader, criterion, device, optimizer=optimizer,
                           augment=augment, use_amp=use_amp, scaler=scaler)
        scheduler.step()
        history.append({"epoch": epoch, **stats})
        print(f"[epoch {epoch:02d}] loss={stats['loss']:.4f} "
              f"voxel={stats['loss_voxel']:.4f} "
              f"gradient={stats['loss_gradient']:.4f} "
              f"organ={stats['loss_organ']:.4f}")
        for k, v in stats.items():
            if k != "epoch" and not math.isfinite(v):
                raise RuntimeError(f"non-finite loss component {k}={v} at epoch {epoch} -- smoke test FAILED")

    ckpt_path = os.path.join(args.out_dir, "dosenet3d_smoke_last.pt")
    torch.save(model.state_dict(), ckpt_path)
    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nCheckpoint saved: {ckpt_path}")
    print("SMOKE TEST COMPLETE -- loop executed cleanly, loss components stayed finite. "
          "This validates architecture + training-loop wiring on FABRICATED data only; "
          "it says nothing about real dose-prediction accuracy.")


if __name__ == "__main__":
    main()
