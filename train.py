"""
train.py
========
DoseNet3D training script. Trains on real GDP-HMM data via
gdp_hmm_adapter.GDPHMMDoseDataset (synthetic_data.py's fabricated cohort was
smoke-test-only scaffolding, no longer used here now that real data is
available).

Usage:
    python train.py --epochs 50 --batch_size 1 \
        --data_splits_json ../data_splits.json --out_dir runs/dosenet3d_real
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data_pipeline import PreprocessConfig
from pipeline_adapter import GDPHMMDoseDataset, GDP_HMM_OAR_LABEL_MAP, DEFAULT_META_DIR, convert_patient
from losses import CompositeDoseLoss
from model import DoseNet3D, fold_depth_into_batch, unfold_batch_to_depth, print_model_summary


def _load_module_from_path(module_name: str, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# In-plane-only augmentation
# --------------------------------------------------------------------------- #
#
# CRITICAL: a horizontal flip mirrors patient left<->right. Any lateralized
# structure baked into the integer-encoded OAR channel must have its label
# value swapped post-flip, or the flipped mask keeps the WRONG side's label
# -- this bug is invisible by eye (the flip still "looks right") and only
# corrupts the label channel. build_oar_label_swap derives the swap table
# from whichever oar_label_map is actually in use (matching "*_L"/"*_R"
# name pairs), rather than hardcoding one map's keys -- GDP_HMM_OAR_LABEL_MAP
# has no lateralized pair (real GDP-HMM ships one combined "Parotids" mask,
# not separate Parotid_L/Parotid_R), so it correctly produces an empty swap
# table: flipping a combined/symmetric structure's label needs no swap.
OAR_CHANNEL_INDEX = 2  # channel 0=CT, 1=PTV, 2=OAR (integer-encoded, default cfg)


def build_oar_label_swap(oar_label_map: dict) -> dict:
    swap = {}
    for name, label in oar_label_map.items():
        if name.endswith("_L") and (name[:-2] + "_R") in oar_label_map:
            swap[label] = oar_label_map[name[:-2] + "_R"]
        elif name.endswith("_R") and (name[:-2] + "_L") in oar_label_map:
            swap[label] = oar_label_map[name[:-2] + "_L"]
    return swap


def hflip_batch(input_t: torch.Tensor, target_t: torch.Tensor, oar_label_swap: dict) -> tuple:
    """Flip the W axis (patient left<->right) and swap lateralized OAR
    labels in the same op so the flipped mask stays anatomically correct."""
    input_flipped = torch.flip(input_t, dims=[3])
    target_flipped = torch.flip(target_t, dims=[3])

    oar = input_flipped[:, OAR_CHANNEL_INDEX].clone()
    swapped = oar.clone()
    for old_label, new_label in oar_label_swap.items():
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
        input_t, target_t = hflip_batch(input_t, target_t, cfg.oar_label_swap)
    if cfg.affine_prob > 0 and torch.rand(1).item() < cfg.affine_prob:
        input_t, target_t = rotate_translate_batch(input_t, target_t, cfg.max_rotation_deg, cfg.max_translate_frac)
    return input_t, target_t


class AugConfig:
    def __init__(self, max_rotation_deg=5.0, max_translate_frac=0.03, hflip_prob=0.5, affine_prob=0.5,
                 oar_label_swap: dict = None):
        self.max_rotation_deg = max_rotation_deg
        self.max_translate_frac = max_translate_frac
        self.hflip_prob = hflip_prob
        self.affine_prob = affine_prob
        self.oar_label_swap = oar_label_swap or {}


# --------------------------------------------------------------------------- #
# Train / eval loop
# --------------------------------------------------------------------------- #

def run_epoch(model, loader, criterion, device, optimizer=None, augment: AugConfig = None,
              use_amp: bool = False, scaler=None, checkpoint_every: int = 0, checkpoint_fn=None):
    """checkpoint_every>0 + checkpoint_fn: called every N training batches with
    (n_batches_done) -- mid-epoch checkpointing. A single epoch here can take
    ~1 hour (real per-item CPU preprocessing dominates), which is longer than
    a Colab session reliably survives; without this, a disconnect anywhere
    mid-epoch loses ALL of that epoch's gradient updates since checkpointing
    otherwise only happens at epoch end."""
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

            if is_train and checkpoint_every and checkpoint_fn and n_batches % checkpoint_every == 0:
                checkpoint_fn(n_batches)

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1, help="memory-heavy architecture; 1 or 2 recommended")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_gradient", type=float, default=0.5)
    parser.add_argument("--lambda_organ", type=float, default=0.3)
    parser.add_argument("--data_splits_json", default="../../data/data_splits.json",
                         help="shared patient-level train/val/test manifest (see data_splits.json at repo root)")
    parser.add_argument("--meta_dir", default=str(DEFAULT_META_DIR))
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default="./runs/dosenet3d_real")
    parser.add_argument("--resume", default=None, help="checkpoint path to resume from (e.g. after a dropped Colab session)")
    parser.add_argument("--checkpoint_every_batches", type=int, default=20,
                         help="mid-epoch checkpoint frequency (batches); 0 disables")
    parser.add_argument("--deadline_epoch_seconds", type=float, default=None,
                         help="Unix epoch seconds (UTC) hard stop -- train-to-time, not train-to-epoch. "
                              "Checked before each epoch; whatever epoch was reached is the deliverable. "
                              "Best checkpoint is always what gets loaded for final eval, not last.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # NOTE: DeformConv2d (torchvision.ops) has no MPS kernel as of this
    # writing, so on Apple Silicon we intentionally stay on CPU rather than
    # silently falling back mid-forward-pass -- CUDA is used when available,
    # CPU otherwise. Real full-resolution [3,256,256,80] training is only
    # practical on CUDA; run this on Colab, not locally, without a GPU.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_amp = True
    else:
        device = torch.device("cpu")
        use_amp = False
        print("No CUDA GPU available -- running on CPU. Full-resolution "
              "[3,256,256,80] real-data training will be very slow; use a "
              "CUDA GPU (e.g. Colab) for a real run.")

    torch.manual_seed(args.seed)

    with open(args.data_splits_json) as f:
        manifest = json.load(f)
    chosen = manifest["chosen_file_per_patient"]
    train_paths = [chosen[pid] for pid in manifest["split"]["train"] if pid in chosen]
    val_paths = [chosen[pid] for pid in manifest["split"]["val"] if pid in chosen]

    print(f"\n=== DoseNet3D training on real GDP-HMM data ===")
    print(f"train patients: {len(train_paths)}, val patients: {len(val_paths)}, "
          f"epochs={args.epochs}, batch_size={args.batch_size}, device={device}\n")

    cfg = PreprocessConfig()
    meta_dir = args.meta_dir or None
    train_dataset = GDPHMMDoseDataset(train_paths, cfg=cfg, meta_dir=meta_dir)
    val_dataset = GDPHMMDoseDataset(val_paths, cfg=cfg, meta_dir=meta_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               drop_last=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             drop_last=False, num_workers=args.num_workers)

    model = DoseNet3D(in_channels=3).to(device)
    print_model_summary(model)

    start_epoch = 1
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed weights from {args.resume}")

    criterion = CompositeDoseLoss(lambda_gradient=args.lambda_gradient, lambda_organ=args.lambda_organ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    oar_label_swap = build_oar_label_swap(GDP_HMM_OAR_LABEL_MAP)
    augment = AugConfig(oar_label_swap=oar_label_swap)

    def save_mid_epoch(n_batches_done):
        torch.save(model.state_dict(), os.path.join(args.out_dir, "dosenet3d_mid_epoch.pt"))
        print(f"  [mid-epoch checkpoint @ batch {n_batches_done}]", flush=True)

    history = []
    best_val_loss = math.inf
    stopped_for_deadline = False
    for epoch in range(start_epoch, args.epochs + 1):
        if args.deadline_epoch_seconds is not None and time.time() >= args.deadline_epoch_seconds:
            print(f"\n[deadline] {time.time():.0f} >= {args.deadline_epoch_seconds:.0f} -- "
                  f"stopping before epoch {epoch} (train-to-time, not train-to-epoch)")
            stopped_for_deadline = True
            break

        train_stats = run_epoch(model, train_loader, criterion, device, optimizer=optimizer,
                                 augment=augment, use_amp=use_amp, scaler=scaler,
                                 checkpoint_every=args.checkpoint_every_batches, checkpoint_fn=save_mid_epoch)
        val_stats = run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step()
        history.append({"epoch": epoch, "train": train_stats, "val": val_stats})
        print(f"[epoch {epoch:03d}] train_loss={train_stats['loss']:.4f} "
              f"(voxel={train_stats['loss_voxel']:.4f} gradient={train_stats['loss_gradient']:.4f} "
              f"organ={train_stats['loss_organ']:.4f}) "
              f"val_loss={val_stats['loss']:.4f}")
        for phase, stats in (("train", train_stats), ("val", val_stats)):
            for k, v in stats.items():
                if not math.isfinite(v):
                    raise RuntimeError(f"non-finite {phase} loss component {k}={v} at epoch {epoch}")

        # Checkpoint every epoch -- Colab free-tier sessions can disconnect
        # without warning; losing more than one epoch of an expensive
        # full-resolution 3D run is avoidable, so always overwrite "last"
        # and additionally keep "best" by val loss.
        last_path = os.path.join(args.out_dir, "dosenet3d_last.pt")
        torch.save(model.state_dict(), last_path)
        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            torch.save(model.state_dict(), os.path.join(args.out_dir, "dosenet3d_best.pt"))
        with open(os.path.join(args.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    status = "stopped at deadline" if stopped_for_deadline else "completed all epochs"
    print(f"\nTraining {status}. Last checkpoint: {os.path.join(args.out_dir, 'dosenet3d_last.pt')}, "
          f"best (val_loss={best_val_loss:.4f}): {os.path.join(args.out_dir, 'dosenet3d_best.pt')}")

    # --- Loss curve (presentation artifact) ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        epochs_x = [h["epoch"] for h in history]
        plt.figure(figsize=(7, 4))
        plt.plot(epochs_x, [h["train"]["loss"] for h in history], label="train")
        plt.plot(epochs_x, [h["val"]["loss"] for h in history], label="val")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("DoseNet3D loss")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "loss_curve.png"), dpi=120)
        plt.close()
        print(f"Loss curve saved: {os.path.join(args.out_dir, 'loss_curve.png')}")
    except Exception as e:
        print(f"[loss curve skipped] {e}")

    # --- Final test-set eval using the BEST checkpoint (by val loss), not last ---
    best_path = os.path.join(args.out_dir, "dosenet3d_best.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"Loaded best checkpoint for final eval: {best_path}")
    model.eval()

    test_paths = [chosen[pid] for pid in manifest["split"]["test"] if pid in chosen]
    _eval_mod = _load_module_from_path("_dosenet3d_evaluate", Path(__file__).resolve().parent / "evaluate.py")

    per_patient = []
    with torch.no_grad():
        for path in test_paths:
            try:
                input_arr, target_arr, masks, summary = convert_patient(path, cfg, meta_dir, GDP_HMM_OAR_LABEL_MAP)
                x = torch.from_numpy(input_arr).unsqueeze(0).to(device)
                pred = model(x).cpu().numpy()[0]
                prescription = max(summary["prescription_doses_gy"].values()) if summary["prescription_doses_gy"] else 70.0
                result = _eval_mod.evaluate_patient(pred, target_arr, masks, dose_max_gy=80.0,
                                                     prescription_dose_gy=prescription)
                per_patient.append({"patient": summary["meta_patient_id"], "voxel_mae_gy": result["voxel_mae_gy"]})
            except Exception as e:
                print(f"[eval skip] {path}: {e}")

    if per_patient:
        import statistics
        maes = [p["voxel_mae_gy"] for p in per_patient]
        final_summary = {
            "model": "dosenet3d",
            "n_test_patients_evaluated": len(per_patient),
            "n_test_patients_total": len(test_paths),
            "voxel_mae_gy_mean": statistics.mean(maes),
            "voxel_mae_gy_std": statistics.pstdev(maes) if len(maes) > 1 else 0.0,
            "best_val_loss": best_val_loss,
            "epochs_completed": len(history),
            "stopped_for_deadline": stopped_for_deadline,
            "per_patient": per_patient,
        }
        with open(os.path.join(args.out_dir, "final_eval.json"), "w") as f:
            json.dump(final_summary, f, indent=2)
        with open(os.path.join(args.out_dir, "final_eval.md"), "w") as f:
            f.write(f"# DoseNet3D final evaluation\n\n"
                    f"- Epochs completed: {len(history)} ({'stopped at deadline' if stopped_for_deadline else 'full run'})\n"
                    f"- Best val loss: {best_val_loss:.4f}\n"
                    f"- Test patients evaluated: {len(per_patient)}/{len(test_paths)}\n"
                    f"- Voxel MAE (Gy): {final_summary['voxel_mae_gy_mean']:.3f} ± {final_summary['voxel_mae_gy_std']:.3f}\n")
        print(f"Final eval written: {os.path.join(args.out_dir, 'final_eval.json')}")


if __name__ == "__main__":
    main()
