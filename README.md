# DoseNet3D: full 3-D radiotherapy dose prediction

[![tests](https://github.com/krank-09/dosenet3d/actions/workflows/tests.yml/badge.svg)](https://github.com/krank-09/dosenet3d/actions/workflows/tests.yml)
![License: MIT](https://img.shields.io/badge/code-MIT-green) ![Data: CC BY-NC-SA 4.0](https://img.shields.io/badge/data-CC%20BY--NC--SA%204.0-lightgrey)

An asymmetric 3-D U-Net that predicts the **full dose distribution**, in Gy at every voxel, for a head-and-neck radiotherapy patient from their CT scan and contours, before any treatment plan has been made. It was trained on the GDP-HMM dataset (AAPM 2025 Grand Challenge). This is the volumetric companion to [dvhnet](https://github.com/krank-09/dvhnet), which predicts per-organ DVH curves instead.

**Based on** the hybrid 2D/3D deformable dose-prediction network of Liu et al. (Medical & Biological Engineering & Computing, 2025; full citation in Credits). The architecture follows the paper: a 3-D convolutional encoder, a 2-D deformable decoder for the anisotropic slice spacing, and residual squeeze-and-excitation blocks with deformable convolutions (RDSEM). It's written from scratch in PyTorch here, and trained and evaluated on the public GDP-HMM head-and-neck data rather than the paper's nasopharyngeal cohort.

> Research code for learning purposes. **Not for clinical use.**

## Architecture

```
input  [B, 3, 256, 256, 80]   CT (windowed) · PTV mask · OAR label map
  │
  ├─ enc1  3-D strided conv + RSEM ──────────────────────────────┐ skip
  ├─ enc2  3-D strided conv + RSEM ────────────────────────┐     │
  └─ enc3  3-D strided conv + RSEM   (bottleneck 32×32×10) │     │
        │                                                  │     │
  dec1  ConvTranspose3d ↑ · concat ◄───────────────────────┘     │
        · RDSEM (2-D deformable convs, slice by slice)           │
  dec2  ↑ · concat ◄─────────────────────────────────────────────┘
  dec3  ↑ · concat raw input · RDSEM
  head  1×1×1 conv → ReLU   → dose [B, 1, 256, 256, 80]  (Gy, ≥ 0)
```

- **Encoder: true 3-D convolutions** to capture context across slices. Each stage ends in a **residual squeeze-and-excitation (RSEM)** block, so the network can learn which feature channels matter for each patient.
- **Decoder: 2-D deformable convolutions** applied per axial slice ("2.5-D"). A small sub-network predicts where each kernel should sample, so filters can bend around curved organ boundaries and keep the steep dose fall-off at PTV and OAR edges sharp. The offset layers start at zero, so each block begins as an ordinary convolution, which makes training stable.
- **InstanceNorm** instead of BatchNorm, because 3-D volumes force a batch size of 1–2.
- About **0.98 M parameters**: small for a 3-D network, so it trains on a free 16 GB GPU.

**Loss** = voxel L1 + 0.5 × **gradient loss** (L1 between finite-difference dose gradients, which penalises blurred edges) + 0.3 × **region-weighted organ loss**. Errors inside serial organs (spinal cord, brainstem) are weighted ×5, inside the PTV ×3, and inside parallel organs ×2.

## Results

Held-out **test set of 50 patients** (patient-level split, seed 42, `splits/`), whole-volume **voxel mean absolute error**:

| Training session | Epochs | Test voxel MAE |
|---|---|---|
| 1 (Colab, cut short by session limit) | 19 | 2.60 ± 0.91 Gy |
| 2 (Kaggle GPU, resumed) | 45 | 1.98 ± 1.06 Gy |
| **3 (Kaggle T4, resumed)** | 32 | **1.42 ± 0.82 Gy** |

![training curve](results/kaggle_2026-09-13/loss_curve.png)

All of the improvement came from more training time; the architecture didn't change. Validation loss was still falling when session 3 hit its time limit. Per-patient errors are in `results/kaggle_2026-09-13/final_eval.json`.

**A bug found along the way.** `--resume` originally restored only the model weights, not the optimizer or learning-rate scheduler state. Every resumed session therefore restarted its cosine learning-rate schedule from the top, which showed up as a jump in validation loss (18.0 → 27.8) right after each resume. Checkpoints now store `{model, optimizer, scheduler, epoch, val_loss}`, and older weight-only checkpoints still load.

**For context:** the challenge's official MedNeXt baseline reports 2.54 Gy (validation) and 3.45 Gy (test). That number is on the full head-and-neck + lung cohort with an isodose-masked MAE, so it isn't directly comparable to the unmasked, head-and-neck-only number above.

## Repository layout

```
model.py            DoseNet3D: encoder (3-D + RSEM), decoder (2-D deformable + RDSEM)
losses.py           voxel + gradient + region-weighted organ loss
data_pipeline.py    numpy preprocessing: CT windowing, PTV/OAR channels, resampling, depth crop
pipeline_adapter.py GDP-HMM .npz patient → model tensors (+ canonical mask set for batching)
train.py            AdamW + cosine LR, mixed precision, augmentation, full-state resume,
                    deadline-aware stopping, final test evaluation
evaluate.py         voxel MAE + clinical DVH metrics per structure (PTV D95/D98, serial D2/Dmax,
                    parallel Dmean) with a 2 Gy / 3%-of-prescription tolerance check
dvh_utils.py        DVH construction, dose-metric inversion, resampling (shared with dvhnet)
synthetic_data.py   fabricated shape-correct patients for smoke tests and CI
splits/             patient-level splits (file paths relative to --data_root)
results/            final evaluation, training history and loss curve of the best run
tests/              pytest: model, losses, DVH maths, the .npz → tensor path, evaluation
```

**Augmentation:** in-plane rotation (±5°) and translation, applied identically to every slice of a volume, plus left–right flips that also swap left/right organ labels. CT and dose use bilinear interpolation; masks use nearest-neighbour, so an integer organ label can never be blended into a different organ's label.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q                          # synthetic data only; a few seconds
python model.py                    # full-size forward pass on a synthetic patient (~1 min, 2 GB RAM on CPU)
```

Training needs the data, which is not included:

1. Request access to the **GDP-HMM Challenge** dataset on Hugging Face (gated, CC BY-NC-SA 4.0): https://huggingface.co/datasets/Jungle15/GDP-HMM_Challenge
2. Copy `meta_files/` (`PTV_DICT.json` etc.) from https://github.com/RiqiangGao/GDP-HMM_AAPMChallenge into this repo's root.
3. Train on a CUDA GPU (a 16 GB T4 works at batch size 1):

```bash
python train.py --data_root /path/to/GDP-HMM --data_splits_json splits/data_splits_reduced180.json \
    --epochs 300 --batch_size 1 --out_dir runs/dosenet3d
# continue after a session limit, keeping optimizer and LR schedule:
python train.py --data_root /path/to/GDP-HMM --resume runs/dosenet3d/dosenet3d_best.pt \
    --epochs 300 --out_dir runs/dosenet3d
```

## Limitations and next steps

- Trained on **180 of the 399** available training patients, to fit free GPU quotas. Validation and test sets are the full 50 each.
- About 100 epochs in total, with validation loss still improving. More training is the obvious next step.
- Only the voxel MAE is aggregated over the test set so far. `evaluate.py` already computes per-structure DVH metrics for each patient, but they aren't yet rolled up into the final report.

## Credits

- **Data:** GDP-HMM Challenge, AAPM 2025, licensed CC BY-NC-SA 4.0. R. Gao, M. Diallo, H. Liu, A. Magliari, W. Verbakel, S. Meyers, M. Zarepisheh, R. Mcbeth, S. Arberet, M. Kraus, F. Ghesu, A. Kamen. *Automating RT Planning at Scale: High Quality Data For AI Training.* arXiv:2501.11803, 2025.
- **Data conventions** (`PTV_DICT.json`, dose scaling) follow the challenge repository by Riqiang Gao et al. (MIT): https://github.com/RiqiangGao/GDP-HMM_AAPMChallenge
- **Architecture:** Y. Liu, W. Luo, X. Li, M. Liu. *Deformable dose prediction network based on hybrid 2D and 3D convolution for nasopharyngeal carcinoma radiotherapy.* Medical & Biological Engineering & Computing 63(3):733–747, 2025. doi:[10.1007/s11517-024-03231-8](https://doi.org/10.1007/s11517-024-03231-8)
- **Building blocks it uses:**
  - U-Net (Ronneberger et al., MICCAI 2015) and 3-D U-Net (Çiçek et al., MICCAI 2016)
  - Squeeze-and-Excitation Networks (Hu et al., CVPR 2018)
  - Deformable ConvNets v2 (Zhu et al., CVPR 2019), via `torchvision.ops.DeformConv2d`
  - Instance Normalization (Ulyanov et al., 2016)
  - AdamW (Loshchilov & Hutter, ICLR 2019)

<details>
<summary>BibTeX</summary>

```bibtex
@article{liu2025deformable,
  title   = {Deformable dose prediction network based on hybrid 2D and 3D convolution for nasopharyngeal carcinoma radiotherapy},
  author  = {Liu, Yanhua and Luo, Wang and Li, Xiangchen and Liu, Min},
  journal = {Medical \& Biological Engineering \& Computing},
  volume  = {63},
  number  = {3},
  pages   = {733--747},
  year    = {2025},
  doi     = {10.1007/s11517-024-03231-8}
}
@article{gao2025gdphmm,
  title   = {Automating RT Planning at Scale: High Quality Data For AI Training},
  author  = {Gao, Riqiang and Diallo, Mamadou and Liu, Han and Magliari, Anthony and Verbakel, Wilko and Meyers, Sandra and Zarepisheh, Masoud and Mcbeth, Rafe and Arberet, Simon and Kraus, Martin and Ghesu, Florin and Kamen, Ali},
  journal = {arXiv preprint arXiv:2501.11803},
  year    = {2025}
}
```
</details>

## License

Code: MIT (see `LICENSE`). The GDP-HMM data, and any model weights trained on it, fall under the dataset's CC BY-NC-SA 4.0 licence (non-commercial, share-alike).
