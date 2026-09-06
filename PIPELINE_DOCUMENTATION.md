# Dose-Prediction Pipelines — Comprehensive Documentation

This document covers all three pipelines in this project:

1. **`dvhnet/`** — a 2D CNN that predicts a per-slice cumulative DVH curve from (target, OAR) mask pairs.
2. **`dosenet3d/`** — a 3D asymmetric U-Net that predicts a full dose volume from CT + PTV + OAR channels.
3. **The Colab pipeline** — `Data Requirements and Preparation.ipynb` at the project root, which is *not* a data-prep notebook despite its filename (see Open Items) but a **remote GPU training runner**: it clones the two repos above from GitHub, installs deps, and drives `dvhnet/train.py` and `dosenet3d/train.py` on a Colab GPU, with checkpoints/shards persisted to Google Drive.

It is written once and copied identically into all three project folders (`dvhnet/PIPELINE_DOCUMENTATION.md`, `dosenet3d/PIPELINE_DOCUMENTATION.md`, and this root copy for the Colab pipeline) so that whichever repo you happen to have open, you have the full picture — including how the other two pipelines connect to it.

Every code excerpt below is quoted verbatim from the files as they exist on disk right now (checked 2026-09-04). Where an earlier plan, TODO, or spec said something different from what's actually implemented, that is called out explicitly rather than smoothed over.

---

## Part 0 — Concepts Primer

### Radiotherapy dose and volumes

- **Gy (Gray)** is the SI unit of absorbed radiation dose — joules of energy absorbed per kilogram of tissue. A head-and-neck treatment might prescribe 70 Gy to a tumor, delivered over ~35 daily fractions of 2 Gy each. Every "dose" value in this codebase (`dose_max_gy`, `prescription_dose_gy`, model outputs) is in Gy unless explicitly normalized.
- **CTV (Clinical Target Volume)**: the anatomical region believed to contain the tumor plus microscopic disease spread — a clinician's contour based on biology, not imaging alone.
- **PTV (Planning Target Volume)**: the CTV expanded by a safety margin to account for setup uncertainty and organ motion. The PTV is the volume the treatment plan is actually optimized to cover with the prescription dose. In this codebase, "target" and "PTV" are used interchangeably (`target_mask`, `PTV_Total`) — CTV is tracked separately only where the raw data provides it (`PreprocessConfig.target_names = ["PTV", "CTV"]` in `dvhnet/preprocessing.py`).
- **OAR (Organ at Risk)**: any healthy structure near the target that dose should be minimized in — parotid glands, spinal cord, brainstem, mandible, etc. OARs split into two clinically distinct categories that this codebase's loss functions treat differently:
  - **Serial organs**: functional units are arranged so that damaging *any* small part can disable the *whole* organ — like links in a chain. The spinal cord is the canonical example: a small volume receiving too much dose can cause paralysis regardless of how the rest of the cord fared. For serial organs, the clinically critical number is a **near-maximum** dose (D2%, Dmax) — you care about the worst-case hotspot, not the average.
  - **Parallel organs**: made of many independent functional sub-units where damaging some still leaves the rest working — like a sponge. The parotid (salivary) glands are the standard example: losing part of the gland reduces salivary function proportionally rather than catastrophically. For parallel organs, the clinically meaningful number is **mean dose** (Dmean) — total functional loss scales with how much of the organ, on average, was irradiated.

  This distinction is exactly why `dosenet3d/losses.py` and `dosenet3d/evaluate.py` maintain a `DEFAULT_SERIAL_ORGANS` set with a higher loss weight and a max-dose-based evaluation metric, while parallel organs get a lower weight and mean-dose evaluation.

### Cumulative DVH (Dose-Volume Histogram)

A DVH summarizes an entire 3D (or per-slice) dose distribution inside a structure as a single curve. The **cumulative** form used throughout this codebase is:

```
y(d) = fraction of the structure's voxels receiving dose >= d
```

Plotted with dose on the x-axis and volume fraction on the y-axis, this curve starts at `y(0) = 1.0` (100% of voxels receive at least 0 Gy, trivially) and is **monotonically non-increasing**, falling to `y(d_max) ≈ 0` at the highest dose any voxel received. It is monotonic because the set of voxels receiving at least `d` Gy can only shrink as `d` increases — nothing can un-satisfy that inequality as the threshold rises.

The whole DVHnet pipeline exists to predict this curve directly from geometry (target and OAR masks) without ever running a real dose calculation.

### Reading D2%, D50%, Dmean off a DVH curve

Given the cumulative curve `y(d)`, "D_x%" means: **the dose at which exactly x% of the structure's volume is receiving that dose or more** — i.e., invert the curve to solve `y(d) = x/100` for `d`.

- **D2%** — the dose received by the hottest 2% of the volume. Used as a practical proxy for "near-maximum dose" without being as noise-sensitive as the single hottest voxel (Dmax). This is the number that matters for serial organs.
- **D50%** — the median dose: the dose level such that half the structure's volume is above it and half below. A general summary statistic.
- **Dmean** — the mean dose across the structure. Not read directly off the x-axis; it equals the **area under the cumulative curve**: `Dmean = ∫ y(d) dd` from 0 to `d_max`. This works because `y(d)` is exactly the survival function of the per-voxel dose distribution, and for a non-negative random variable `X`, `E[X] = ∫ P(X ≥ x) dx`. This is precisely how `dvhnet/metrics.py`'s `d_mean()` computes it — via `np.trapz` (numerical integration), not inversion.

Geometrically: D2%/D50% are found by reading *across* from a y-axis value to find the corresponding x (dose); Dmean is the *area* under the whole curve, not a single point on it.

### MAD (Mean Absolute DVH Deviation)

A whole-curve error metric, distinct from any single dose metric: the average absolute difference between a predicted and ground-truth cumulative DVH curve, bin-by-bin:

```
MAD = (1/num_bins) * sum_k |pred[k] - target[k]|
```

Where D2%/D50%/Dmean each collapse the curve to one clinically meaningful number, MAD scores the *entire shape* of the curve at once — useful for catching a model that gets the clinical summary points right by coincidence while getting the overall curve shape wrong.

### dosenet3d-specific concepts

**Anisotropic resampling, and why the z-axis is deliberately NOT made isotropic.** CT/dose volumes are typically much finer in-plane (e.g. 1×1 mm pixels) than along the scan axis (e.g. 2.5–3 mm slice thickness). `dvhnet/preprocessing.py`'s pipeline resamples *all three axes* to a uniform target spacing (`target_slice_thickness_mm`), because DVHnet operates per 2D slice and needs a physically consistent slice pitch for its voxel-count-weighted aggregation. `dosenet3d`, in contrast, works on full 3D volumes at a fixed depth (`target_depth = 80` slices) and explicitly resamples **only the in-plane (H, W) axes**, leaving the Z axis at its native slice thickness (see `data_pipeline.resample_inplane_only`). Forcing Z to be isotropic here would mean either interpolating extra slices into a stack that never had that much real information along that axis (fabricating detail that doesn't exist) or discarding real slices to hit a uniform spacing — both are lossy in directions the model has no way to recover from, and both change how many slices comprise the tumor's craniocaudal extent, an effect that has to stay physically faithful for the model to learn true dose falloff along the beam/couch axis. The code enforces this via a comment marked `CRITICAL per spec`.

**Instance Norm vs. Batch Norm, and why IN suits batch size 1–2.** BatchNorm normalizes activations using statistics (mean/variance) computed *across the batch* dimension. That's a problem when your model can only fit a batch size of 1–2 volumes in GPU memory at once (true here — `dosenet3d/train.py`'s `--batch_size` help text literally says "memory-heavy architecture; 1 or 2 recommended", because a `[3, 256, 256, 80]` volume through a 3D U-Net is large): batch statistics computed from 1–2 samples are noisy and unstable, and BatchNorm's running-mean/variance estimates during training become unreliable. InstanceNorm instead normalizes each sample independently, using statistics computed *per-channel, per-sample* (across only the spatial dimensions) — so its behavior doesn't degrade as batch size shrinks, and train/eval behave consistently since there's no batch-dependent running statistic to get wrong. This is why every normalization layer in `dosenet3d/model.py` is `nn.InstanceNorm3d` / `nn.InstanceNorm2d`, never `BatchNorm`.

**Squeeze-and-Excitation (SE) blocks.** A lightweight channel-attention mechanism: global-average-pool the feature map down to one value per channel, pass that through a small two-layer bottleneck MLP (`Linear → ReLU → Linear → Sigmoid`) to produce a per-channel gate in `[0, 1]`, then multiply the original feature map by that gate. The effect is that the network can learn to amplify channels that matter for the current input and suppress ones that don't, at very low parameter/compute cost — a `SEBlock3D`/`SEBlock2D` pair implements exactly this in `dosenet3d/model.py`.

**Deformable convolutions (DCNv2) and the "2D deformable decoder".** A standard convolution samples its input at a fixed grid of offsets around each output location (e.g. the 9 positions of a 3×3 kernel). A **deformable** convolution instead *learns*, per output location, small additional `(dx, dy)` offsets for each of those sampling positions — so the kernel's effective receptive field can bend to follow the actual shape of an object (like a curved PTV/OAR boundary) instead of always sampling a rigid square. **DCNv2** (the "v2" / modulated variant used here) adds a second learned quantity: a per-location, per-sample-point *modulation mask* in `[0, 1]` that scales how much each deformed sample contributes — so the network can also learn to suppress irrelevant samples entirely, not just move them. Operationally, "2D deformable decoder" means: `dosenet3d`'s decoder does its *sharpening* refinement stage with 2D (not 3D) deformable convolutions, applied independently to each axial slice by folding the depth axis into the batch dimension (`fold_depth_into_batch`) — a "2.5D" scheme, not a true 3D deformable conv (which torchvision doesn't provide, and would be far more expensive). See Part 2 for exactly how this is wired.

**Gradient difference loss.** A voxel-wise loss (L1/L2/Smooth-L1) treats every location independently and doesn't know or care whether the *pattern* of errors is smeared out (blurred) versus sharp. Since a plain reconstruction loss is minimized, on average, by predicting something close to the *smoothed* version of a sharp target (blurring away sharp edges costs less loss than getting them slightly wrong), models trained with only a voxel loss tend to blur exactly the discontinuities that matter clinically — the dose falloff at a PTV/OAR boundary. A gradient-difference loss instead computes the finite-difference spatial gradient of both the prediction and the ground truth (a simple `x[i+1] - x[i]` per axis) and penalizes the L1 distance *between those two gradient fields*. This directly penalizes "the edge is in the wrong place or the wrong sharpness," independent of whether the absolute dose value nearby also happens to be correct — which is exactly the failure mode a voxel-only loss misses. `dosenet3d/losses.py`'s `GradientLoss` implements this, summed over all three spatial axes (H, W, D).

---

## Part 1 — `dvhnet/`

Files covered, in the order data flows through them: `preprocessing.py` → `dataset.py` → `model.py` → `losses.py` → `train.py` → `aggregate.py` → `metrics.py`.

### `preprocessing.py`

**Purpose**: turn paired DICOM CT/RTSTRUCT/RTDOSE studies into the exact slice-level training examples DVHnet consumes — 2-channel (target, OAR) masks plus a 256-bin cumulative-DVH label per slice, and a patient-level train/val/test split.

**Configuration dataclass:**

```python
@dataclass
class PreprocessConfig:
    target_matrix: Tuple[int, int] = (256, 256)     # in-plane resample size
    target_slice_thickness_mm: float = 3.0           # uniform slice spacing
    dose_max_gy: float = 80.0                         # site-specific ceiling (Nasopharynx: 75-80 Gy)
    num_bins: int = 256
    oar_names: List[str] = field(default_factory=lambda: [
        "Brainstem", "SpinalCord", "ParotidL", "ParotidR", "Mandible",
        "Larynx", "Esophagus", "OralCavity", "Lens_L", "Lens_R",
        "OpticNerve_L", "OpticNerve_R", "Chiasm", "TemporalLobe_L",
        "TemporalLobe_R", "InnerEar_L",
    ])
    target_names: List[str] = field(default_factory=lambda: ["PTV", "CTV"])
```

`target_matrix`/`target_slice_thickness_mm` fix the geometry every patient gets resampled to, so the 2D CNN in `model.py` always sees a consistent input resolution regardless of the source scanner's native pixel spacing. `dose_max_gy` fixes the DVH's dose axis range — every patient's dose gets binned against the *same* 0–80 Gy range so DVH curves are directly comparable across patients (a patient prescribed 54 Gy and one prescribed 80 Gy still get DVH vectors of the same length and dose-per-bin, which is what makes bin-wise loss and cross-patient aggregation meaningful). `oar_names`/`target_names` are the working list this DICOM-native pipeline expects to find in the RTSTRUCT; **note this list is a completely different naming convention than what the real GDP-HMM data actually contains** — see Part 3's discussion of the naming mismatch.

**DICOM loading — note the axis-order handling:**

```python
def load_ct_series(ct_dir: str) -> Tuple[np.ndarray, Tuple[float, float, float],
                                          Tuple[float, float, float], List[pydicom.Dataset]]:
    files = glob.glob(os.path.join(ct_dir, "*.dcm"))
    slices = [pydicom.dcmread(f) for f in files]
    slices.sort(key=lambda d: float(d.ImagePositionPatient[2]))

    hu_volume = np.stack([
        (s.pixel_array.astype(np.float32) * float(getattr(s, "RescaleSlope", 1.0))
         + float(getattr(s, "RescaleIntercept", 0.0)))
        for s in slices
    ], axis=0)

    # DICOM PixelSpacing = [row spacing, column spacing]; the array is
    # [Z, H, W] = [Z, rows, cols], so row_spacing maps to H and col_spacing to W.
    row_spacing, col_spacing = [float(v) for v in slices[0].PixelSpacing]
    ...
```

Two non-obvious details worth calling out: (1) slices are explicitly sorted by `ImagePositionPatient[2]` (the physical z-coordinate) rather than trusted to arrive in filename order — DICOM directories are not guaranteed to enumerate slices in anatomical order, and getting this wrong silently scrambles the volume. (2) raw pixel data is not HU by itself; DICOM stores a linear rescale (`RescaleSlope`/`RescaleIntercept`) that must be applied to get real Hounsfield Units — the code defaults these to `1.0`/`0.0` via `getattr` in case a file omits them, rather than crashing.

**Resampling — nearest-neighbor for masks, linear for dose, and why that's not interchangeable:**

```python
def resample_volume(volume: np.ndarray,
                     src_spacing: Tuple[float, float, float],
                     dst_spacing: Tuple[float, float, float],
                     dst_shape: Optional[Tuple[int, int, int]] = None,
                     order: int = 1) -> np.ndarray:
    """
    order=1 (linear) for continuous dose; order=0 (nearest)
    should be used for binary masks to preserve hard edges.
    """
```

This single parameter is one of the most consequential correctness details in the whole pipeline. Dose is a smooth, physically continuous field — linear interpolation (`order=1`) between two known dose values is a legitimate estimate of what a dose meter would read in between. A binary mask is not continuous in that sense: a voxel is either inside the organ or it isn't. Linearly interpolating a 0/1 mask produces fractional values (0.3, 0.7, …) at every boundary voxel, which then get silently treated as "partially in the organ" by anything downstream expecting a hard boundary — this softens organ edges and, critically, changes voxel counts (blurring adds phantom partial-membership voxels around the true boundary), which directly corrupts the DVH label since `compute_slice_cumulative_dvh` counts voxels by `dose_slice[oar_mask_slice > 0]`. Nearest-neighbor (`order=0`) instead snaps every resampled mask voxel to the value of its single nearest source voxel, preserving a hard 0/1 boundary at the cost of some boundary jaggedness — the correct tradeoff for a categorical mask. `README.md` calls this out explicitly as a "mixing these up will silently soften OAR edges and bias the DVH labels" gotcha.

**Origin-aware dose-to-CT alignment — the naive-vs-actual distinction:**

A naive implementation would resample the dose grid to the CT's spacing using only the *ratio* of the two spacings (a plain "zoom"). The actual implementation does more:

```python
def align_dose_to_ct(dose: np.ndarray, dose_spacing, dose_origin,
                      ct_shape: Tuple[int, int, int], ct_spacing, ct_origin,
                      cfg: PreprocessConfig) -> np.ndarray:
    """
    Resample RTDOSE onto the CT grid, respecting the spatial offset between
    the dose and CT volume origins -- RTDOSE grids are usually smaller than,
    and offset from, the CT field of view, so a plain spacing-ratio zoom
    would silently misalign dose against anatomy unless the two origins
    happened to coincide -- then resample both onto the configured target
    in-plane matrix / slice thickness.
    """
    # Step 1: dose grid -> CT-native grid, in the CT's own physical frame.
    dose_on_ct_native = _resample_to_grid(
        dose, dose_spacing, dose_origin,
        ct_shape, ct_spacing, ct_origin,
        order=1, cval=0.0,
    )
    # Step 2: CT-native spacing -> target uniform spacing / matrix
    ...
```

RTDOSE grids in clinical practice are computed on a coarser, smaller grid than the CT (to save calculation time/storage) and are placed at their own physical origin, which is not guaranteed to coincide with the CT's origin. `_resample_to_grid` explicitly maps every *destination* voxel to its real-world `(x, y, z)` physical position using `ImagePositionPatient`, then samples the *source* volume at the corresponding source-voxel coordinate via `scipy.ndimage.map_coordinates` — this is the only way to correctly place a smaller, offset dose grid onto the CT's larger frame of reference. A plain ratio-based zoom implicitly assumes the two volumes' `(0,0,0)` array indices correspond to the same physical point, which is false here; getting this wrong silently misregisters dose against anatomy without throwing any error (the shapes still "work," just wrongly). Destination voxels falling outside the dose grid's actual coverage are filled with `cval=0.0` — the code and README both flag that you must confirm the dose grid's true field of view actually covers every OAR of interest, since an OAR partially outside it would get spuriously zero dose rather than an error.

**Slice-level cumulative DVH construction — the core label-generation function, reused across all three pipelines:**

```python
def compute_slice_cumulative_dvh(dose_slice: np.ndarray, oar_mask_slice: np.ndarray,
                                  cfg: PreprocessConfig) -> Optional[np.ndarray]:
    """
    Build the length-`num_bins` cumulative DVH vector for one axial slice:
        y[k] = fraction of OAR voxels on this slice with dose >= D_k
    Returns None if the OAR is absent on this slice (caller should skip it).
    """
    voxel_doses = dose_slice[oar_mask_slice > 0]
    n = voxel_doses.size
    if n == 0:
        return None

    bin_edges = np.linspace(0.0, cfg.dose_max_gy, cfg.num_bins)
    # Vectorized: for each bin threshold, fraction of voxels >= threshold.
    # voxel_doses[:, None] >= bin_edges[None, :] -> [N, num_bins] boolean
    cumulative = (voxel_doses[:, None] >= bin_edges[None, :]).mean(axis=0)
    return cumulative.astype(np.float32)  # monotonic non-increasing, in [0, 1]
```

`dose_slice[oar_mask_slice > 0]` uses boolean-array indexing to pull out exactly the organ's voxel doses as a flat 1D array — this is why `evaluate.py` in `dosenet3d` can reuse this exact function on a full 3D volume unmodified: boolean masking flattens regardless of the input's dimensionality, so "slice" in the function name describes the intended calling convention here, not a hard dimensional constraint in the implementation. The DVH itself is computed with a single vectorized broadcast comparison (`voxel_doses[:, None] >= bin_edges[None, :]`) rather than a Python loop over bins — for `N` voxels and 256 bins this builds an `[N, 256]` boolean array and averages down axis 0, which is both simpler and much faster than looping. The function returns `None` (not a zero-filled array) when the organ is absent from a slice specifically so the caller (`build_slice_dataset`) can *skip* that slice entirely rather than teaching the model a spurious "empty organ → all-zero DVH" case that doesn't correspond to any real clinical scenario.

**Per-slice example construction:**

```python
def build_slice_dataset(study: PatientStudy, cfg: PreprocessConfig,
                         oar_name: str) -> List[Dict]:
    examples = []
    oar_mask = study.oar_masks.get(oar_name)
    if oar_mask is None:
        return examples

    for z in range(study.dose_volume.shape[0]):
        oar_slice = oar_mask[z]
        if oar_slice.sum() == 0:
            continue  # organ not present on this slice
        ...
        examples.append({
            "patient_id": study.patient_id,
            "oar_name": oar_name,
            "slice_index": z,
            "target_mask": target_slice.astype(np.float32),   # channel 1
            "oar_mask": oar_slice.astype(np.float32),         # channel 2
            "dvh_label": dvh,                                  # [num_bins]
            "voxel_count": int(oar_slice.sum()),               # for weighted aggregation
        })
    return examples
```

Every field name here has a specific downstream consumer: `patient_id`/`slice_index` are pure bookkeeping used later to regroup slice predictions back into per-patient volumes (`aggregate.py`); `target_mask`/`oar_mask` become the model's two input channels; `dvh_label` is the regression target; `voxel_count` is **not used by the model at all** — it exists purely so that `aggregate.py` can weight each slice's contribution to the reconstructed 3D DVH by how much of the organ that slice actually contains (see below).

**Patient-level split — why splitting happens above the slice level:**

```python
def split_patients(patient_ids: List[str], train_frac=0.8, val_frac=0.1,
                    seed: int = 42) -> Dict[str, List[str]]:
    """Patient-level 80/10/10 (or paper's fixed 153/27 style) split. Never split by slice."""
    rng = np.random.RandomState(seed)
    ids = list(patient_ids)
    rng.shuffle(ids)
    ...
```

The split operates on the list of *patient IDs*, before any slice ever exists as an individual example. If the split instead randomly assigned individual slices to train/val/test, slices from the same patient — which are spatially adjacent and share that patient's exact anatomy, dose plan, and organ shape — would end up on both sides of the split. A model could then partially memorize a specific patient's geometry from training slices and get an inflated, non-generalizing score on that same patient's neighboring slices in validation/test. Splitting at the patient level guarantees no patient's data leaks across the split boundary at all, which is the only valid way to estimate how the model generalizes to genuinely unseen patients.

### `dataset.py`

**Purpose**: PyTorch `Dataset`/collation glue, plus `.npz`-shard persistence so slice examples don't have to be regenerated (or held fully in memory) every training run.

**The dataset item:**

```python
class DVHSliceDataset(Dataset):
    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        target = ex["target_mask"]
        oar = ex["oar_mask"]

        if self.augment:
            target, oar = self._augment(target, oar)

        inp = np.stack([target, oar], axis=0).astype(np.float32)  # [2, H, W]
        label = ex["dvh_label"].astype(np.float32)                # [num_bins]

        item = {
            "input": torch.from_numpy(inp),
            "label": torch.from_numpy(label),
            "voxel_count": ex["voxel_count"],
            "patient_id": ex["patient_id"],
            "slice_index": ex["slice_index"],
        }
        return item
```

Channel order is fixed and load-bearing: channel 0 is always the target/PTV mask, channel 1 is always the OAR mask — `model.py`'s docstring and `DVHNet.__init__(in_channels=2, ...)` both assume this exact ordering, so any code producing an `input` tensor for this model (including `gdp_hmm_adapter.py`, transitively, via `build_slice_dataset`) must preserve it.

**Augmentation — why flips are safe here but wouldn't be for a segmentation task with a scalar label tied to absolute position:**

```python
@staticmethod
def _augment(target: np.ndarray, oar: np.ndarray):
    """Light, label-preserving augmentation: masks only, dose curve unaffected
    by pure spatial flips since it's derived from voxel counts, not shape."""
    if np.random.rand() < 0.5:
        target = np.ascontiguousarray(target[:, ::-1])
        oar = np.ascontiguousarray(oar[:, ::-1])
    if np.random.rand() < 0.5:
        target = np.ascontiguousarray(target[::-1, :])
        oar = np.ascontiguousarray(oar[::-1, :])
    return target, oar
```

The DVH label is a function of *voxel counts satisfying a dose threshold*, not of *where in the image* those voxels are. Flipping a mask horizontally or vertically permutes which pixels are "in" the mask but doesn't change how many voxels there are or what doses they received — so the paired `dvh_label` for that example remains exactly correct after a flip with zero relabeling work. This is exactly the property `dosenet3d`'s equivalent flip augmentation does *not* have for its integer-encoded OAR channel (see Part 2's `hflip_batch` discussion) — the difference is that DVHnet's OAR mask here is a single binary channel per organ (no left/right identity baked into a shared channel's *value*), whereas `dosenet3d` stacks multiple organs, including left/right pairs, into one integer-labeled channel where a flip actually needs corresponding label remapping. `np.ascontiguousarray` after each slice-reversal is necessary because NumPy's `[::-1]` produces a negative-stride view, not a new contiguous array, and `torch.from_numpy` cannot wrap a negative-stride array directly.

**Shard I/O:**

```python
def save_patient_shard(shard_dir: str, patient_id: str, examples: List[Dict]) -> None:
    os.makedirs(shard_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(shard_dir, f"{patient_id}.npz"),
        target_masks=np.stack([e["target_mask"] for e in examples]),
        oar_masks=np.stack([e["oar_mask"] for e in examples]),
        dvh_labels=np.stack([e["dvh_label"] for e in examples]),
        voxel_counts=np.array([e["voxel_count"] for e in examples], dtype=np.int32),
        slice_indices=np.array([e["slice_index"] for e in examples], dtype=np.int32),
    )
```

One `.npz` file per patient, keyed by filename (`{patient_id}.npz`) — this is the exact format `gdp_hmm_adapter.py` targets (see Part 3), and it's what makes `get_patient_ids` in `train.py` work by just listing shard filenames: the file system itself encodes the patient-ID-to-examples grouping, so `load_examples_from_shards(shard_dir, patient_ids=splits["train"])` can selectively load only the shards belonging to one split without touching the others. `np.savez_compressed` (not plain `savez`) trades some CPU for smaller files, which matters for the Colab pipeline's Drive-sync step (small files sync faster than large ones over Drive's slow small-file I/O).

### `model.py`

**Purpose**: `DVHNet`, a ResNet-style 2D CNN regressing the 256-bin cumulative DVH vector.

```python
class DVHNet(nn.Module):
    def __init__(self, in_channels: int = 2, num_bins: int = 256,
                 base_channels: int = 32, fc_dims=(1024, 512, 256)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # H/4
        )
        self.stage1 = ConvBlock(base_channels, base_channels * 2, stride=2)      # H/8
        self.stage2 = ConvBlock(base_channels * 2, base_channels * 4, stride=2)  # H/16
        self.stage3 = ConvBlock(base_channels * 4, base_channels * 8, stride=2)  # H/32
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling bottleneck
        ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x).flatten(1)          # [B, feat_dim]
        x = self.head(x)                    # [B, fc_dims[-1]]
        x = self.out(x)                     # [B, num_bins]
        return self.sigmoid(x)              # cumulative DVH in [0, 1]
```

Note `DVHNet` uses `nn.BatchNorm2d`, not `InstanceNorm` — this is a deliberate contrast with `dosenet3d`: `train.py`'s default `--batch_size 32` is large enough for stable batch statistics, since a single 2D slice input is far cheaper than a `dosenet3d` full 3D volume, so there's no batch-size-driven reason to avoid BatchNorm here (see Part 0's IN-vs-BN discussion). The architecture is a plain progressive-downsampling CNN (stem gives H/4, three residual `ConvBlock` stages each halve resolution again down to H/32) feeding a Global Average Pooling bottleneck — GAP is used instead of flattening the full spatial feature map into the FC head specifically so the model's parameter count and behavior don't depend on the input's exact H×W (the FC head only ever sees a fixed `feat_dim`-length vector regardless of input resolution). A `sigmoid` output head is used because the cumulative DVH is bounded in `[0, 1]` by definition (a fraction of voxels) — bounding the output range this way is a stronger inductive bias than letting the network learn unconstrained real values and hoping the loss pushes it into range.

**The monotonicity post-processing fallback:**

```python
def enforce_monotonic(dvh: torch.Tensor) -> torch.Tensor:
    """
    Post-processing fallback: force a non-increasing curve via a running
    cumulative-minimum along the dose axis. Use at inference time if the
    monotonicity loss term hasn't fully eliminated small violations.
    """
    return torch.cummin(dvh, dim=-1).values
```

A neural network's raw output has no hard guarantee of being monotonic even with a monotonicity loss term encouraging it (see `losses.py`) — the loss only *penalizes* violations, it doesn't structurally prevent them. `torch.cummin` runs a cumulative minimum from the low-dose end of the curve to the high-dose end, which by construction can never increase: at each bin it's forced to be `min(current_value, all_previous_values)`. This is applied at evaluation/inference time (`train.py`'s `evaluate()` calls `enforce_monotonic(model(x))`) as a cheap correctness guarantee that doesn't require retraining, on top of — not instead of — the training-time loss penalty.

### `losses.py`

**Purpose**: the composite training objective `L_total = L_fidelity + λ1·L_clinical + λ2·L_mono`.

```python
def _index_for_percent(num_bins: int, percent: float) -> int:
    """
    Cumulative DVH y[k] = P(dose >= D_k). D_x% is the dose at which the curve
    crosses volume fraction x/100. We approximate the *bin index* nearest that
    volume level for a lightweight, differentiable proxy loss (rather than
    inverting the curve, which is done properly at eval time in metrics.py).
    """
    return max(0, min(num_bins - 1, round((1.0 - percent / 100.0) * (num_bins - 1))))
```

This is a genuinely subtle piece of the design and worth spelling out carefully, because the index it computes is **not** an approximation of D2%/D50%'s dose value — it's a fixed bin position used purely to decide *which entries of the label vector* to upweight during training. A true D2%/D50% computation (as done properly in `metrics.py`) inverts the *curve itself*, which varies per-patient — the actual dose at which 50% of the volume is covered differs across patients. But for the *loss*, doing that per-sample inversion at every training step, differentiably, would be expensive and numerically awkward near flat regions of the curve. Instead, the loss cheats in a well-reasoned way: it always looks at the *same fixed dose-axis bin index* (roughly the 50th-percentile-from-the-top *bin position*, not dose value) and applies extra squared-error weight there, on the theory that errors in that region of the curve are the ones most likely to matter for the eventual clinical read-off, without needing to actually solve for where 50% volume falls for each individual sample. The comment is explicit that "true dose-metric inversion happens in metrics.py" — this function is a cheap proxy, not the real computation.

```python
class DVHLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        # 1) Point-wise fidelity
        if self.use_smooth_l1:
            l_fidelity = F.smooth_l1_loss(pred, target, reduction="mean")
        else:
            l_fidelity = F.mse_loss(pred, target, reduction="mean")

        # 2) Clinical metric penalization: extra squared-error weight at D2%/D50% bins
        pred_clin = pred.index_select(dim=1, index=self.clinical_idx)
        target_clin = target.index_select(dim=1, index=self.clinical_idx)
        l_clinical = F.mse_loss(pred_clin, target_clin, reduction="mean") * self.clinical_weight

        # 3) Monotonicity: penalize any increase from bin k to bin k+1
        diffs = pred[:, 1:] - pred[:, :-1]                 # [B, num_bins-1]
        l_mono = torch.clamp(diffs, min=0.0).pow(2).mean()  # squared hinge, smoother gradient than raw max(0,.)

        total = l_fidelity + self.lambda_clinical * l_clinical + self.lambda_mono * l_mono
        return {"loss": total, "loss_fidelity": l_fidelity.detach(),
                "loss_clinical": l_clinical.detach(), "loss_mono": l_mono.detach()}
```

`use_smooth_l1=True` by default chooses Huber loss over raw MSE for the base fidelity term specifically because a DVH curve has a steep transition region (the falloff from ~1.0 to ~0.0) — squared error there produces very large gradients exactly where the curve is hardest to fit precisely, and Smooth L1 caps that gradient magnitude (behaves like L1 beyond a threshold, L2 within it), making training less dominated by that one region. The monotonicity term uses a **squared hinge** (`clamp(diffs, min=0).pow(2)`) rather than a raw hinge (`clamp(diffs, min=0)`) because the squared version has a smoother, vanishing gradient right at the zero-violation boundary — a raw hinge's gradient is a constant 1 for any violation no matter how small, which can cause oscillation right around the monotonic boundary; the squared version's gradient shrinks to zero as the violation shrinks to zero, which converges more cleanly. This is also flagged as a deliberate deviation from the exact original spec ("swap `losses.DVHLoss` back to `torch.clamp(diffs, min=0.0).mean()` if you want the exact formulation from your spec" — README.md), so the design intentionally traded literal spec-fidelity for a training-stability improvement, and says so.

All three loss components are returned in the dict, detached, purely so `train.py` can log and diagnose them independently — this is the same rationale `dosenet3d/losses.py`'s docstring states explicitly ("if e.g. loss_mono-equivalent terms stall while loss_voxel keeps moving, that's diagnostic, not noise").

### `aggregate.py`

**Purpose**: reconstruct a patient-level (3D) DVH from a set of per-slice predictions, via voxel-count-weighted averaging.

```python
def aggregate_patient_dvh(slice_predictions: List[np.ndarray],
                           slice_voxel_counts: List[int]) -> np.ndarray:
    """
    D_patient = sum_s(V_s * d_s) / sum_s(V_s)
    """
    weights = np.asarray(slice_voxel_counts, dtype=np.float64)
    total_weight = weights.sum()
    if total_weight <= 0:
        raise ValueError("Total OAR voxel count across slices must be positive.")
    stacked = np.stack(slice_predictions, axis=0)  # [S, num_bins]
    weighted = (weights[:, None] * stacked).sum(axis=0) / total_weight
    return weighted.astype(np.float32)
```

The naive way to combine per-slice DVH predictions into one patient-level DVH would be a plain unweighted average across slices. That would be wrong: a parotid gland's superior and inferior pole slices — where the organ's cross-section is small — would then count exactly as much toward the patient-level curve as its equatorial slices, where the organ's cross-section (and therefore its true voxel *volume*, i.e. clinical relevance) is largest. Weighting each slice's predicted DVH by that slice's actual OAR voxel count (`voxel_count`, produced back in `preprocessing.build_slice_dataset`) makes the aggregation equivalent to computing the DVH over the *union of all the organ's voxels across every slice at once* — which is what the true 3D patient-level DVH actually is by definition — rather than an average-of-averages that implicitly treats a thin sliver of organ as equally important as its widest cross-section.

```python
def aggregate_from_batch_outputs(patient_ids: List[str], slice_indices: List[int],
                                  preds: np.ndarray, voxel_counts: np.ndarray
                                  ) -> Dict[str, np.ndarray]:
    grouped_preds: Dict[str, List[np.ndarray]] = defaultdict(list)
    grouped_weights: Dict[str, List[int]] = defaultdict(list)
    for pid, pred, w in zip(patient_ids, preds, voxel_counts):
        grouped_preds[pid].append(pred)
        grouped_weights[pid].append(int(w))
    ...
```

This wrapper exists because a `DataLoader` with `shuffle=True` (used for training, though notably evaluation loaders in `train.py` use `shuffle=False`) can hand back slices from different patients interleaved in arbitrary order across batches — `defaultdict(list)` regroups everything by `patient_id` regardless of what order slices arrived in, before per-patient aggregation runs.

### `metrics.py`

**Purpose**: invert predicted DVH curves back into clinically meaningful dose numbers, and score them against ground truth.

**Curve inversion (D2%/D50%):**

```python
def dose_at_volume(dvh: np.ndarray, volume_percent: float, dose_max_gy: float) -> float:
    num_bins = dvh.shape[-1]
    doses = _bin_edges(num_bins, dose_max_gy)
    target = volume_percent / 100.0

    # Ensure strict monotonic non-increasing curve for a well-posed inversion
    y = np.minimum.accumulate(dvh)

    if target >= y[0]:
        return float(doses[0])
    if target <= y[-1]:
        return float(doses[-1])

    # np.interp requires increasing x; y is decreasing, so flip both arrays
    y_rev = y[::-1]
    doses_rev = doses[::-1]
    return float(np.interp(target, y_rev, doses_rev))
```

Three details matter here. First, `np.minimum.accumulate(dvh)` re-applies the same monotonic-enforcement idea as `model.enforce_monotonic` (running cumulative minimum) — this function is called on both predicted *and* ground-truth curves, so even a ground-truth curve with tiny numerical noise gets a well-posed inversion. Second, the two boundary checks (`target >= y[0]`, `target <= y[-1]`) handle the edge cases where the requested volume fraction is outside the curve's actual range (e.g. asking for D2% when even the *lowest* dose bin still covers less than 2% of the volume) — without them, `np.interp` would silently clamp to the nearest endpoint anyway, but handling it explicitly makes the boundary behavior an intentional, documented choice rather than an implicit side effect of `np.interp`'s clamping. Third, `np.interp` requires its independent-variable array to be *increasing*, but `y` (volume fraction) is *decreasing* in dose by construction — the code explicitly reverses both `y` and `doses` (`[::-1]`) before calling `np.interp`, which is a real gotcha: forgetting this reversal wouldn't crash, it would silently return wrong interpolated values everywhere except the exact array-index boundaries, since `np.interp` doesn't validate that its `xp` argument is sorted.

**Mean dose as integral, not lookup:**

```python
def d_mean(dvh: np.ndarray, dose_max_gy: float) -> float:
    """
    Mean dose = area under the cumulative DVH curve (integral of y(dose) ddose),
    since y(dose) is the survival function of the per-voxel dose distribution
    and E[dose] = integral_0^max P(dose >= d) dd for a nonnegative dose.
    """
    num_bins = dvh.shape[-1]
    doses = _bin_edges(num_bins, dose_max_gy)
    y = np.minimum.accumulate(dvh)
    return float(_trapz(y, doses))
```

This is the concrete implementation of the Part 0 "Dmean = area under the curve" identity, using the trapezoidal rule (`np.trapz`/`np.trapezoid` — the code handles both names since NumPy 2.0 renamed the function: `_trapz = getattr(np, "trapezoid", None) or np.trapz`, a forward-compatibility guard against exactly that rename breaking on newer NumPy).

**Cohort summary and acceptance check:**

```python
def check_acceptance_benchmarks(summary: Dict[str, float],
                                 dmean_threshold_gy: float = 1.0,
                                 d2_threshold_gy: float = 2.5) -> Dict[str, bool]:
    """
    Compare cohort summary stats against the paper's acceptance criteria:
    mean-dose error <= 1.0 Gy, D2% error within 2.0-2.5 Gy.
    """
    return {
        "dmean_within_threshold": summary.get("Dmean_MAE_gy", float("inf")) <= dmean_threshold_gy,
        "d2_within_threshold": summary.get("D2_MAE_gy", float("inf")) <= d2_threshold_gy,
    }
```

`.get(..., float("inf"))` is a deliberate defensive default: if `summarize_results` was ever called with an empty results list (no patients evaluated), `summary` would be `{}`, and comparing a missing key against `inf` guarantees the acceptance check fails closed (reports "not within threshold") rather than raising a `KeyError` or silently defaulting to `True`.

### `train.py`

**Purpose**: end-to-end CLI that wires every module above into one training/evaluation run.

**The evaluation function, showing the full slice→patient pipeline in one place:**

```python
@torch.no_grad()
def evaluate(model, loader, device, dose_max_gy: float):
    model.eval()
    all_preds, all_labels, all_pids, all_slices, all_voxels = [], [], [], [], []

    for batch in loader:
        x = batch["input"].to(device)
        pred = enforce_monotonic(model(x)).cpu().numpy()
        all_preds.append(pred)
        all_labels.append(batch["label"].numpy())
        all_pids.extend(batch["patient_id"])
        all_slices.extend(batch["slice_index"])
        all_voxels.append(batch["voxel_count"].numpy())

    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    voxels = np.concatenate(all_voxels, axis=0)

    pred_dvhs = aggregate_from_batch_outputs(all_pids, all_slices, preds, voxels)
    true_dvhs = aggregate_from_batch_outputs(all_pids, all_slices, labels, voxels)

    results = [
        evaluate_patient(pid, "OAR", pred_dvhs[pid], true_dvhs[pid], dose_max_gy)
        for pid in pred_dvhs
    ]
    summary = summarize_results(results)
    summary["acceptance"] = check_acceptance_benchmarks(summary)
    return summary
```

Note the model is evaluated at the *slice* level (batched forward passes over `DVHSliceDataset` items), but scored at the *patient* level — every raw slice prediction is aggregated back into one 3D DVH per patient (via `aggregate_from_batch_outputs`) before any clinical metric is computed, and `enforce_monotonic` is applied to raw model outputs before aggregation so the weighted-average of already-monotonic curves stays sane. Ground-truth labels go through exactly the same aggregation function as predictions (both calls use `aggregate_from_batch_outputs`), which matters: it ensures pred vs. true are compared on DVH curves built with an identical weighting scheme, not accidentally comparing a weighted-aggregated prediction against, say, a full 3D-volume-derived ground truth computed a different way.

**The train/val split-to-shard pipeline, and the real crash mode documented in `TODO.md`:**

```python
patient_ids = get_patient_ids(args.shard_dir)
splits = split_patients(patient_ids, args.train_frac, args.val_frac, seed=args.seed)

train_examples = load_examples_from_shards(args.shard_dir, splits["train"])
val_examples = load_examples_from_shards(args.shard_dir, splits["val"])
test_examples = load_examples_from_shards(args.shard_dir, splits["test"])
```

With the default `train_frac=0.8, val_frac=0.1` and only 2 real patient shards on disk right now (`dvhnet/shards/Parotids/HNC_001_A4Ac.npz`, `HNC_001_9Ag.npz`), `n_train = round(2*0.8) = 2`, `n_val = round(2*0.1) = 0`, leaving `test` empty too — this is confirmed in `TODO.md`: "the default 80/10/10 split leaves val/test empty with only 2 real patients and crashes the final test evaluation." The actual smoke test that was run used `--train_frac 0.5 --val_frac 0.0` (1 train / 0 val / 1 test) to route around this, and the resulting `val_loss=0.0000` seen in the real run's logs is the *mechanical* consequence of an empty val loader (`run_epoch` returns `0.0` totals divided by `max(n_batches, 1)` when there are zero batches) — not a bug in the loss computation itself, and not evidence the model is actually achieving zero validation loss.

**Real results on disk** (`dvhnet/runs/parotids_smoke/test_summary_Parotids.json`, from the smoke test against the two real GDP-HMM-derived shards):

```json
{
  "n_patients": 1,
  "D2_MAE_gy": 7.47, "D50_MAE_gy": 38.79, "Dmean_MAE_gy": 5.24,
  "MAD_mean": 0.302,
  "acceptance": {"dmean_within_threshold": false, "d2_within_threshold": false}
}
```

These numbers are expected to be poor (5 epochs on a single training patient is nowhere near enough signal for a CNN to learn anything general) — they confirm the *pipeline* runs correctly end-to-end, not that the model is accurate. Both acceptance flags reporting `false` is the correct, expected result here, not a defect.

---

## Part 2 — `dosenet3d/`

Files covered: `synthetic_data.py`, `data_pipeline.py`, `model.py`, `losses.py`, `train.py`, `evaluate.py`.

### `synthetic_data.py`

**Purpose**: fabricate shape- and value-correct synthetic patients so the rest of the pipeline can be exercised end-to-end before real GDP-HMM data access lands. The module docstring is explicit that this "does not read or approximate real DICOM/GDP-HMM data" and exists "to catch shape and wiring bugs, not to validate dose-prediction accuracy."

**Ellipsoid mask/distance-field generation:**

```python
def _ellipsoid_mask(center, radii, shape=(H, W, D)) -> np.ndarray:
    hh, ww, dd = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij")
    ch, cw, cd = center
    rh, rw, rd = radii
    val = ((hh - ch) / rh) ** 2 + ((ww - cw) / rw) ** 2 + ((dd - cd) / rd) ** 2
    return (val <= 1.0).astype(np.uint8)
```

This is the implicit-surface equation of an ellipsoid: a point is inside when the sum of its squared, per-axis-normalized distances from the center is `≤ 1`. `indexing="ij"` (rather than the NumPy default `"xy"`) is used so that `hh, ww, dd` line up directly with array axes 0, 1, 2 — matching, not transposing relative to, the volume's own `[H, W, D]` axis order.

**The important design point: synthetic data calls the *real* channel-construction functions, not a parallel reimplementation:**

```python
from data_pipeline import (
    DEFAULT_OAR_LABEL_MAP, PreprocessConfig, build_oar_channel,
    build_ptv_channel, total_input_channels, window_and_normalize_ct,
)
...
ct_channel = window_and_normalize_ct(ct_hu, cfg)
ptv_channel = build_ptv_channel(ptv_masks, {"PTV": prescription_dose_gy}, cfg)
oar_channel = build_oar_channel(oar_masks, DEFAULT_OAR_LABEL_MAP, cfg)
```

The only thing this file invents is the underlying anatomy (blob positions/shapes) and the dose field's fabrication logic; the actual channel *encoding* — HU windowing, PTV/OAR channel construction — runs through the exact same functions the real-data path (`preprocess_arrays`) will eventually call. This means a smoke test passing here is real evidence that `data_pipeline.py`'s channel-construction code works correctly, not just that the synthetic generator's own bespoke logic is self-consistent.

**Dose fabrication — structured, not random noise, and why that distinction matters for testing the loss:**

```python
dist = _ellipsoid_distance(ptv_center, tuple(r + 6 for r in ptv_radii), shape)
dose = prescription_dose_gy * np.exp(-np.clip(dist - 1.0, 0.0, None) * 0.9)
...
sparing_factor = {"Brainstem": 0.30, "SpinalCord": 0.35, "Parotid_L": 0.55, "Parotid_R": 0.55}
for name, mask in oar_masks.items():
    dose[mask > 0] *= sparing_factor.get(name, 0.5)
```

The dose field is built to be hot and near-uniform inside the PTV, exponentially falling off with distance outside it, and *additionally* suppressed inside each OAR by a sparing factor — serial organs (Brainstem 0.30, SpinalCord 0.35) get a much lower relative dose than parallel organs (Parotids 0.55). This is deliberate: if the synthetic dose field were pure random noise, `dosenet3d/losses.py`'s region-weighted organ loss would have nothing structured to actually validate — a model could fit noise trivially, and the loss's per-structure weighting would never be exercised meaningfully. Baking in an actual serial-vs-parallel sparing pattern gives the region-weighted loss (and the smoke test as a whole) a real, non-trivial signal that mirrors what a real clinical dose distribution looks like.

**`SyntheticDoseDataset.__getitem__` — why raw masks are passed alongside the encoded input tensor:**

```python
def __getitem__(self, idx: int) -> Dict:
    ex = self.examples[idx]
    # `masks` carries each structure's binary mask independently of the
    # input tensor's encoding (integer-label OAR channel, possibly
    # dose-scaled PTV channel, etc.) -- losses.py's region-weighted term
    # consumes these directly rather than reverse-decoding them from the
    # model input, so it stays correct regardless of cfg.oar_one_hot /
    # cfg.ptv_scale_by_dose.
    masks = {name: torch.from_numpy(m.astype("float32")) for name, m in
              {**ex.ptv_masks, **ex.oar_masks}.items()}
    return {"input": torch.from_numpy(ex.input), "target": torch.from_numpy(ex.target),
            "patient_id": ex.patient_id, "masks": masks}
```

A naive design might have `RegionWeightedOrganLoss` decode structure membership back out of the model's *input* tensor (e.g. "voxels where the OAR channel equals label 3 are Parotid_R"). That would break the moment `cfg.oar_one_hot` or `cfg.ptv_scale_by_dose` changes how those channels are encoded — the loss would need to know the current encoding scheme to reverse it correctly. Instead, each structure's plain binary mask is carried through the data item independently of however the input tensor happens to encode it, so the loss function never needs to know or care about the encoding config at all.

### `data_pipeline.py`

**Purpose**: the real (non-synthetic) preprocessing logic — windowing, channel construction, in-plane-only resampling, depth cropping — structured so it's already real-data-ready except for the actual DICOM/GDP-HMM reading step, which is stubbed.

**The module-name collision and its fix — documented as its own concept, since it's a real gotcha:**

```python
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
```

**Why this is a real gotcha, worth understanding fully in case it happens again elsewhere:** the two sibling packages `dvhnet/` and `dosenet3d/` each have their own `model.py`, `losses.py`, and `train.py`, with different contents. Python's import system resolves `import losses` (or `from losses import X`) by searching `sys.path` **in order** and returning the *first* module named `losses` it finds — it has no concept of "the losses.py that belongs to this package" unless the import is written as a proper relative/package-qualified import. The straightforward-looking fix of `sys.path.insert(0, "../dvhnet")` (to reach `dvhnet/preprocessing.py`) has a side effect nobody intends: it also makes `dvhnet/losses.py` and `dvhnet/model.py` importable as bare `losses`/`model` — and if dosenet3d's own `train.py` does `from losses import CompositeDoseLoss` *after* that path insertion happened (or if `dvhnet`'s path entry sorts earlier), it can silently resolve to the wrong file. This is exactly the bug `TODO.md` records as actually hit and fixed: "`dvhnet/losses.py` and `dvhnet/model.py` were shadowing dosenet3d's own same-named files via `sys.path`." Because both files define plausible-looking classes, there was no import error — the wrong module simply loaded successfully and the code appeared to work until behavior diverged from expectations.

The fix — `importlib.util.spec_from_file_location` + `module_from_spec` + `exec_module` — loads a specific file directly by its **path**, not by searching `sys.path` for a name at all, and registers the result in `sys.modules` under a synthetic private key (`"_dvhnet_preprocessing"`, not `"preprocessing"`), so it can never collide with, or be shadowed by, anything else named `preprocessing`, `losses`, or `model` anywhere on `sys.path`. The `sys.modules[module_name] = module` registration *before* `exec_module` runs is itself necessary for a subtle reason called out in the comment: `dataclasses` resolves a decorated class's module at decoration time by looking it up in `sys.modules[cls.__module__]` (for `get_type_hints`-style resolution of string-quoted annotations), and if the module isn't registered yet when the `@dataclass`-decorated code in `preprocessing.py` executes, that lookup fails. `dvhnet/preprocessing.py` uses `@dataclass` for both `PreprocessConfig` and `PatientStudy`, so this ordering isn't optional — it's required for those classes to load correctly via this mechanism at all.

The general lesson, stated as a reusable rule: **when two sibling Python packages both need to be importable and have any same-named files, don't add either one to `sys.path`** — load only the specific file(s) you actually need via `importlib.util.spec_from_file_location`, under a private module-name key that can't collide with anything.

**In-plane-only resampling, and how it delegates to `dvhnet` rather than reimplementing interpolation:**

```python
def resample_inplane_only(volume_zhw: np.ndarray, spacing, target_hw, order):
    """
    CRITICAL per spec: do NOT make the z-axis isotropic like
    dvhnet/preprocessing.py's pipeline does ... Reuses
    dvhnet.preprocessing.resample_volume by pinning the destination
    z-spacing equal to the source (zoom factor 1 on that axis) rather than
    reimplementing the interpolation logic.
    """
    dz, dy, dx = spacing
    z, h, w = volume_zhw.shape
    dst_spacing = (dz, dy * h / target_hw[0], dx * w / target_hw[1])
    dst_shape = (z, target_hw[0], target_hw[1])
    resampled = _dvhnet_resample_volume(volume_zhw, spacing, dst_spacing, dst_shape=dst_shape, order=order)
    return resampled, dst_spacing
```

Rather than writing a second interpolation routine, this calls `dvhnet.preprocessing.resample_volume` (the exact 3-axis zoom function documented in Part 1) but constructs `dst_spacing` with the **z-component copied unchanged from the source spacing** (`dz` on both sides) — since `resample_volume`'s internal `zoom_factors = [s/d for s, d in zip(src_spacing, dst_spacing)]` then computes a zoom factor of exactly `1.0` on that axis, leaving Z untouched, while H/W still get resampled to `target_hw`. This is a clean way to get "resample only these two axes" behavior out of a general 3-axis function without adding a second code path or an `axis` parameter to the underlying routine.

**Depth cropping, centered on the PTV rather than the volume:**

```python
def ptv_center_z_index(ptv_mask_hwd: np.ndarray) -> int:
    """Index (along the last, D/z axis) of the PTV's midpoint; falls back to
    the volume's center if no PTV voxels are present."""
    z_indices = np.nonzero(ptv_mask_hwd.sum(axis=(0, 1)))[0]
    if len(z_indices) == 0:
        return ptv_mask_hwd.shape[-1] // 2
    return int(round((int(z_indices.min()) + int(z_indices.max())) / 2))
```

Since `target_depth = 80` is a fixed window and a real patient's full CT stack is typically much taller, the crop has to be centered *somewhere* — centering on the raw volume's geometric middle would be wrong whenever the tumor isn't anatomically centered in the scan's field of view (which is the common case; scan extents are chosen for coverage margins, not tumor centering). Centering the depth window on the PTV's own midpoint instead guarantees the tumor (and, by the anatomy of typical H&N/thoracic geometry, its nearby OARs) stays inside the cropped window regardless of where it happens to sit in the raw scan.

### `model.py` — `DoseNet3D`

**Purpose**: the asymmetric U-Net: a full-3D encoder (RSEM blocks) paired with a 2D-deformable decoder (RDSEM blocks).

**Confirmed implementation detail: `DeformConv2d` comes from `torchvision.ops`, not a custom implementation:**

```python
from torchvision.ops import DeformConv2d
```

This is the actual import in `model.py` right now — the deformable convolution primitive itself (the core "sample at learned offsets" operation) is torchvision's built-in, battle-tested CUDA/CPU kernel, not a hand-rolled reimplementation. What *is* custom is the wrapper around it that makes it modulated (DCNv2-style):

```python
class DeformConvBlock2D(nn.Module):
    """A single modulated (DCNv2-style) 2D deformable conv: a small offset
    sub-network predicts per-location (dx, dy) sampling offsets, and a
    parallel sigmoid mask sub-network predicts a per-location modulation
    weight, both consumed by torchvision.ops.DeformConv2d."""

    def __init__(self, channels: int, kernel_size: int = 3, deform_groups: int = 1):
        super().__init__()
        offset_channels = 2 * kernel_size * kernel_size * deform_groups
        mask_channels = kernel_size * kernel_size * deform_groups
        self.offset_conv = nn.Conv2d(channels, offset_channels, kernel_size=3, padding=1)
        self.mask_conv = nn.Conv2d(channels, mask_channels, kernel_size=3, padding=1)
        self.deform_conv = DeformConv2d(channels, channels, kernel_size=kernel_size,
                                         padding=kernel_size // 2)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        nn.init.zeros_(self.mask_conv.weight)
        nn.init.zeros_(self.mask_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        return self.deform_conv(x, offset, mask)
```

`torchvision.ops.DeformConv2d` implements the sampling/convolution mechanics but expects the caller to *supply* the per-location offset and modulation-mask tensors — it doesn't predict them itself. `offset_conv` and `mask_conv` are the two small learned sub-networks that produce those tensors: `offset_channels = 2 * k * k * deform_groups` because each of the kernel's `k*k` sampling points needs an `(dx, dy)` pair (hence the factor of 2) per deformable group; `mask_channels = k*k*deform_groups` because each sampling point needs exactly one scalar modulation weight. `mask_conv`'s output is passed through `sigmoid` to constrain the modulation weight to `[0, 1]` (interpretable as "how much to trust/use this sample," where 0 means ignore it entirely).

The zero-initialization of both sub-networks' weights *and* biases is a specific, well-known stabilization trick for deformable convs, and the comment states why directly: with all-zero weights and biases, `offset_conv` outputs all-zero offsets (so `DeformConv2d` initially samples at the exact same fixed grid positions as a normal convolution) and `mask_conv` outputs all-zero pre-sigmoid values, i.e. `sigmoid(0) = 0.5` uniformly (a constant, content-independent modulation). In other words: **at initialization, this block behaves exactly like an ordinary (unmodulated, non-deformed) convolution**, and only starts learning where to actually deform and how to modulate as training progresses. Without this trick, randomly-initialized offsets at the start of training would scatter samples somewhat randomly around each output location, producing a much noisier, harder-to-stabilize training signal from step one.

**RSEM (encoder, 3D) vs. RDSEM (decoder, 2D-deformable) — confirmed actual structure:**

```python
class RSEM(nn.Module):
    """x_out = ReLU(x + SE(Conv-IN-ReLU-Conv-IN(x))), channels unchanged."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(channels, affine=True)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(channels, affine=True)
        self.act = nn.ReLU(inplace=True)
        self.se = SEBlock3D(channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.se(out)
        return self.act(x + out)


class RDSEM(nn.Module):
    """Same residual + SE structure as RSEM, but both internal convs are 2D
    deformable convs. Expects a [N, C, H, W] tensor where N = B*D."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.dconv1 = DeformConvBlock2D(channels)
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.dconv2 = DeformConvBlock2D(channels)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)
        self.act = nn.ReLU(inplace=True)
        self.se = SEBlock2D(channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.norm1(self.dconv1(x)))
        out = self.norm2(self.dconv2(out))
        out = self.se(out)
        return self.act(x + out)
```

Both blocks share the identical residual+SE skeleton (`Conv/DeformConv → IN → ReLU → Conv/DeformConv → IN → SE-gate → residual-add → ReLU`); the only structural difference is that RSEM's two convolutions are ordinary `nn.Conv3d` while RDSEM's are `DeformConvBlock2D` — everything else (channel count preserved, residual shortcut, SE gating, ReLU placement) matches exactly. This confirms the two blocks really are "the same idea applied at two different dimensionalities/operators," not two independently-designed structures that happen to share a name.

**Folding depth into batch — the mechanism behind "2D deformable decoder" operationally:**

```python
def fold_depth_into_batch(x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """[B, C, H, W, D] -> [B*D, C, H, W]"""
    b, c, h, w, d = x.shape
    return x.permute(0, 4, 1, 2, 3).reshape(b * d, c, h, w), b, d

def unfold_batch_to_depth(x: torch.Tensor, b: int, d: int) -> torch.Tensor:
    """[B*D, C, H, W] -> [B, C, H, W, D]"""
    n, c, h, w = x.shape
    return x.reshape(b, d, c, h, w).permute(0, 2, 3, 4, 1)
```

```python
class DecoderStage(nn.Module):
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.norm(self.reduce(x)))
        x2d, b, d = fold_depth_into_batch(x)
        x2d = self.rdsem(x2d)
        return unfold_batch_to_depth(x2d, b, d)
```

This is exactly what "2D deformable decoder" means operationally: after the 3D transposed-conv upsample and skip-connection concatenation (still fully 3D, i.e. cross-slice information is exchanged at that point), the tensor is reshaped so every one of its `D` axial slices becomes an independent item in the *batch* dimension (`permute` + `reshape`, no data duplication — a view-preserving reorganization), the RDSEM block then processes all of those `B*D` slices as ordinary independent 2D images (the deformable conv genuinely only ever sees one 2D slice's worth of neighborhood at a time, with no cross-slice mixing inside RDSEM itself), and finally `unfold_batch_to_depth` reshapes back to `[B, C, H, W, D]` afterward. Cross-slice context is therefore only exchanged in the 3D `ConvTranspose3d`/`Conv3d` layers surrounding each RDSEM call, never within RDSEM itself — the "2.5D" characterization in the module docstring is precise: full 3D awareness at the coarser structural level, per-slice 2D sharpening at the fine-detail level.

**Confirmed shape trace** (from the module docstring and `__main__` self-test, which asserts `out.shape == (batch.shape[0], 1, 256, 256, 80)` and passes with a forward pass on synthetic data):

```
[B,3,256,256,80] -> enc1 -> [B,24,128,128,40] -> enc2 -> [B,48,64,64,20] -> enc3 (bottleneck) -> [B,96,32,32,10]
                 -> dec1 (+enc2 skip) -> [B,48,64,64,20]
                 -> dec2 (+enc1 skip) -> [B,24,128,128,40]
                 -> dec3 (+raw-input skip) -> [B,24,256,256,80]
                 -> 1x1x1 conv + ReLU -> [B,1,256,256,80]
```

Note `dec3`'s skip connection is the **raw model input** itself (`self.dec3 = DecoderStage(c1, skip_ch=in_channels, out_ch=c1)`, called as `self.dec3(d2, skip=x)`) rather than an intermediate encoder feature map — this gives the final deformable refinement stage direct access to the original CT/PTV/OAR channels at full resolution, not just a downsampled-then-upsampled representation of them, which matters for recovering the sharpest possible dose-falloff boundaries right at PTV/OAR edges.

The `__main__` block confirms an actual measured parameter count: `print_model_summary` reports **~981K total parameters** — this was independently confirmed as correct (per `TODO.md`) via a real forward pass on synthetic data, not just a theoretical count.

### `losses.py`

**Purpose**: `CompositeDoseLoss = L_voxel + λ1·L_gradient + λ2·L_organ`.

**Voxel + gradient loss:**

```python
def _grad_l1(pred: torch.Tensor, target: torch.Tensor, dim: int) -> torch.Tensor:
    """Mean L1 distance between finite-difference gradients along `dim`."""
    pred_grad = pred.narrow(dim, 1, pred.shape[dim] - 1) - pred.narrow(dim, 0, pred.shape[dim] - 1)
    target_grad = target.narrow(dim, 1, target.shape[dim] - 1) - target.narrow(dim, 0, target.shape[dim] - 1)
    return (pred_grad - target_grad).abs().mean()

class GradientLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return _grad_l1(pred, target, 2) + _grad_l1(pred, target, 3) + _grad_l1(pred, target, 4)
```

`tensor.narrow(dim, start, length)` is used rather than slicing syntax purely as a style/efficiency choice — `narrow` returns a view without copying, same as slicing would, but reads slightly more explicitly here about which two overlapping windows (`[1:]` and `[:-1]`) are being subtracted to form a finite difference. Dims `2, 3, 4` correspond to `H, W, D` in the `[B, 1, H, W, D]` tensor layout (0=batch, 1=channel) — summing the gradient-difference loss across all three spatial axes means sharp edges are penalized regardless of which anatomical direction they run in.

**The `DEFAULT_SERIAL_ORGANS` casing bug — found and fixed, and why `evaluate.py` importing this constant matters:**

```python
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
```

This was a real, silent bug documented in `TODO.md`: `synthetic_data.py`/`data_pipeline.py` name the structure `"Brainstem"` (lowercase "s"), but the real GDP-HMM data's actual key is `"BrainStem"` (capital "S", confirmed directly against a real downloaded sample file — see Part 3). Membership testing (`name in self.serial_organs`) is exact-string, case-sensitive — so if the set only contained one casing, every mask using the *other* casing would silently fall through to `_weight_for`'s parallel-organ default (`DEFAULT_PARALLEL_WEIGHT = 2.0`) instead of the intended serial weight (`5.0`), with **no error, warning, or shape mismatch** to signal anything was wrong — every synthetic Brainstem mask trained with the wrong loss weight for every run before this was caught. The fix isn't just adding the missing casing to the set (which would fix `synthetic_data.py`'s runs going forward); it also makes `dosenet3d/evaluate.py` **import this exact constant from `losses.py`** (`from losses import DEFAULT_SERIAL_ORGANS`, confirmed in `evaluate.py`'s imports) rather than defining or hardcoding its own copy — so the training-time weighting and the evaluation-time serial/parallel classification are now structurally guaranteed to agree, and the two can no longer silently drift apart the way the original casing mismatch happened in the first place.

**Region-weighted organ loss:**

```python
def forward(self, pred: torch.Tensor, target: torch.Tensor,
            masks: Dict[str, torch.Tensor]) -> torch.Tensor:
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
```

`if voxel_count.item() == 0: continue` guards a genuine divide-by-zero case that isn't hypothetical here: a structure legitimately can have zero voxels in a given cropped/resampled sample (e.g. an OAR entirely outside the depth-cropped window) — without this guard, that structure would contribute `0/0 = NaN` to the loss, and a single `NaN` anywhere in a sum poisons the entire total loss, silently breaking that training step (and, via `.backward()`, corrupting gradients for every parameter, not just ones related to that structure).

### `train.py`

**Purpose**: the smoke-test training loop against `synthetic_data.py`'s fabricated cohort, plus in-plane-only augmentation.

**The lateralized-OAR flip bug class, and why it's dangerous specifically because it's invisible:**

```python
# CRITICAL: a horizontal flip mirrors patient left<->right. Any lateralized
# structure baked into the integer-encoded OAR channel (e.g. Parotid_L=3,
# Parotid_R=4 in DEFAULT_OAR_LABEL_MAP) must have its label value swapped
# post-flip, or the flipped mask keeps the WRONG side's label -- this bug is
# invisible by eye (the flip still "looks right") and only corrupts the
# label channel.
OAR_LABEL_SWAP = {
    DEFAULT_OAR_LABEL_MAP["Parotid_L"]: DEFAULT_OAR_LABEL_MAP["Parotid_R"],
    DEFAULT_OAR_LABEL_MAP["Parotid_R"]: DEFAULT_OAR_LABEL_MAP["Parotid_L"],
}
OAR_CHANNEL_INDEX = 2  # channel 0=CT, 1=PTV, 2=OAR (integer-encoded, default cfg)

def hflip_batch(input_t: torch.Tensor, target_t: torch.Tensor) -> tuple:
    input_flipped = torch.flip(input_t, dims=[3])
    target_flipped = torch.flip(target_t, dims=[3])

    oar = input_flipped[:, OAR_CHANNEL_INDEX].clone()
    swapped = oar.clone()
    for old_label, new_label in OAR_LABEL_SWAP.items():
        swapped[oar == old_label] = new_label
    input_flipped[:, OAR_CHANNEL_INDEX] = swapped

    return input_flipped, target_flipped
```

`torch.flip(dims=[3])` mirrors the geometric arrangement of every channel (the W axis is patient left-right), but each channel's *pixel values* are just copied as-is to their new mirrored position — the geometry flips, but if a pixel encoded "label 3 = Parotid_L" before the flip, that same pixel (now on the anatomically opposite side of the volume) still holds the integer value 3 after the flip, silently mislabeling the parotid that's now really on the physical left as "Parotid_L=3" data sitting on the anatomical right. Because the *shape* of the flipped mask still looks completely correct (an ellipsoid blob still sits where a parotid should be), this class of bug produces no visual or shape-based signal that anything is wrong — only the *identity* of which side is which gets corrupted. `hflip_batch` fixes this by relabeling every `Parotid_L`/`Parotid_R` integer value in the OAR channel to its mirror-image counterpart in the same operation as the geometric flip, so the two always stay consistent. This is exactly the class of bug DVHnet's own mask-flip augmentation (Part 1, `dataset.py`) doesn't need to worry about, because its OAR channel is a single binary per-organ mask with no shared-channel left/right identity baked into the *value* — the difference in how the two pipelines encode organ identity is precisely what makes one augmentation trivially safe and the other require explicit handling.

**Interpolation mode split during affine augmentation — nearest for masks, bilinear for continuous fields (the same principle as `dvhnet/preprocessing.py`, applied here at training-augmentation time instead of at preprocessing time):**

```python
def rotate_translate_batch(input_t, target_t, max_rotation_deg, max_translate_frac) -> tuple:
    """
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
```

The comment gives a concrete and important example of *why* nearest-neighbor is required here beyond the general "hard edges" reasoning from `dvhnet/preprocessing.py`: with an *integer-encoded* OAR channel (label 1 = Brainstem, label 3 = Parotid_L, etc.), bilinearly blending between two adjacent labels during rotation doesn't just soften a boundary — it can produce a value that happens to equal a *third, semantically unrelated* label (interpolating between 1 and 3 can produce 2, which is `SpinalCord`, an organ that was never actually near that boundary). This is strictly worse than the mask-blurring problem in `dvhnet`'s binary-mask case, because there the interpolated value at least stays "partially this one organ" — here it can silently *become a different organ entirely*.

**In-plane-only augmentation applied uniformly across all D slices — not per-slice:**

```python
def _apply_inplane_affine_5d(x: torch.Tensor, theta: torch.Tensor, mode: str) -> torch.Tensor:
    """Apply the SAME (per-patient) in-plane affine to every D slice of a
    [B, C, H, W, D] tensor -- i.e. a single rigid in-plane transform per
    volume, never a different one per slice (that would be an elastic-like
    warp along z, which the spec explicitly excludes)..."""
    x2d, b, d = fold_depth_into_batch(x)
    theta_rep = theta.repeat_interleave(d, dim=0)
    grid = F.affine_grid(theta_rep, x2d.shape, align_corners=False)
    x2d = F.grid_sample(x2d, grid, mode=mode, padding_mode="zeros", align_corners=False)
    return unfold_batch_to_depth(x2d, b, d)
```

This reuses `model.py`'s `fold_depth_into_batch`/`unfold_batch_to_depth` functions directly (not a parallel reimplementation) specifically so the slice ordering this augmentation assumes exactly matches the convention `RDSEM` relies on elsewhere in the model — one rotation/translation angle is computed *per patient* (`theta`, shape `[B, 2, 3]`) and then broadcast to every one of that patient's `D` slices via `repeat_interleave(d, dim=0)`, guaranteeing every slice of one volume gets the identical rigid in-plane transform. Applying a *different* transform per slice would effectively warp the volume non-rigidly along the z-axis (each slice rotated/translated independently relative to its neighbors) — a form of augmentation the design explicitly excludes, consistent with the "no z-axis distortion" principle established in `data_pipeline.py`'s in-plane-only resampling.

**CPU/MPS device handling — a real environment-specific decision, not a placeholder:**

```python
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
```

This isn't a generic "use GPU if available" stub — it's explicit about a real, currently-true PyTorch/torchvision limitation (no MPS backend for `DeformConv2d`) and a deliberate choice to *never* target `mps` at all for this model, rather than let PyTorch throw a confusing mid-forward-pass "operator not implemented for MPS" error the first time a `DeformConvBlock2D` runs. This is directly why the actual smoke test that was run took ~230s/step — it ran on CPU (this development environment has no CUDA GPU), and the print statement in this branch says so explicitly to the user.

**Actual smoke-test results on disk** (`dosenet3d/runs/smoke_test/history.json`, a real 3-epoch run against synthetic data):

```json
[
  {"epoch": 1, "loss": 234.42, "loss_voxel": 17.42, "loss_gradient": 1.71, "loss_organ": 720.49},
  {"epoch": 2, "loss": 161.57, "loss_voxel": 17.46, "loss_gradient": 1.74, "loss_organ": 477.48},
  {"epoch": 3, "loss": 198.87, "loss_voxel": 17.35, "loss_gradient": 1.48, "loss_organ": 602.58}
]
```

All values stayed finite across all three epochs (the `run_epoch` loop's finite-check `assert`/`raise RuntimeError` on any non-finite component never fired), which is exactly and only what this smoke test is designed to confirm — the non-monotonic loss trajectory (234 → 161 → 198) is unsurprising and not evidence of a problem, since 3 epochs on 4 synthetic patients is far too little signal to expect smooth convergence; `TODO.md` and the script's own printed banner both explicitly warn against reading anything about model *quality* into these numbers.

### `evaluate.py`

**Purpose**: score a predicted dose volume against ground truth — voxel MAE plus per-structure clinical metrics (PTV coverage, serial-organ max-dose, parallel-organ mean-dose) with an acceptance tolerance check.

**Confirmed reuse of `dvhnet`'s DVH math, via the same importlib mechanism as `data_pipeline.py`:**

```python
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
```

This confirms, in actual code, the cross-reference claim from Part 1: `compute_slice_cumulative_dvh` genuinely is called here on a **full 3D volume**, not a single slice, and it works unmodified because — as explained in Part 1 — the function's implementation only ever does `dose_slice[oar_mask_slice > 0]`, a boolean-mask flatten that has no dependency on the input's dimensionality. `dvhnet.metrics.dose_at_volume`/`d2`/`d50`/`d_mean` are likewise used directly (`_dvhnet_metrics.d2(...)`, `_dvhnet_metrics.d_mean(...)`) — **this cross-pipeline reuse is real and confirmed in the current code**, not merely planned.

**Per-structure-kind metric dispatch:**

```python
def evaluate_structure(name, pred_dose, true_dose, mask, dose_max_gy, prescription_dose_gy) -> List[StructureReport]:
    if mask.sum() == 0:
        return []
    tol = _tolerance(prescription_dose_gy)
    reports = []

    if name.upper().startswith("PTV"):
        ...  # D95%, D98%
    elif name in DEFAULT_SERIAL_ORGANS:
        ...  # D2%, Dmax
    else:
        ...  # Dmean
    return reports
```

This directly encodes the Part 0 serial/parallel distinction into evaluation policy: PTVs are scored on target *coverage* (D95%/D98% — how much of the tumor got at least the prescribed dose), serial organs are scored on their worst-case dose (D2% and the raw voxel max, `Dmax`, computed directly rather than via curve inversion since it's just `pred_dose[mask > 0].max()`), and everything else (implicitly, parallel organs) falls to `Dmean`. Because `DEFAULT_SERIAL_ORGANS` is imported from `losses.py` (not redefined here), the fix described earlier for the casing bug applies transitively to evaluation as well — a structure classified as serial for loss-weighting purposes is now guaranteed to also be classified as serial for evaluation-metric purposes.

**The acceptance-tolerance formula:**

```python
def _tolerance(prescription_dose_gy: float, abs_threshold_gy: float = 2.0, pct_threshold: float = 0.03) -> float:
    """err <= max(abs_threshold, pct_threshold * Rx) is exactly equivalent
    to 'err <= abs_threshold OR err <= pct_threshold * Rx'."""
    return max(abs_threshold_gy, pct_threshold * prescription_dose_gy)
```

This is a clean algebraic simplification worth spelling out: "pass if the error is within 2 Gy, OR within 3% of the prescription dose" is a disjunction of two separate `err <= X` conditions — and `(err <= A) OR (err <= B)` is logically identical to the single condition `err <= max(A, B)`, since satisfying the looser of the two bounds is exactly what "at least one of them holds" means. Rather than write and maintain two separate comparisons with an `or`, the code precomputes the single effective threshold once and does one comparison against it.

---

## Part 3 — GDP-HMM Adapter (`dvhnet/gdp_hmm_adapter.py`)

**Purpose**: convert a single GDP-HMM AAPM Challenge patient `.npz` file into a dvhnet training shard — i.e., produce exactly the same `{patient_id}.npz` shard format `dataset.load_examples_from_shards` already expects, so no downstream dvhnet code needs to know or care that a given shard originated from GDP-HMM data rather than a locally-run DICOM pipeline.

Despite living physically inside `dvhnet/` (not a separate top-level folder), this adapter is documented as its own part here because it's conceptually a distinct pipeline stage — the bridge between GDP-HMM's raw distribution format and dvhnet's training format.

### Why this reuses `compute_slice_cumulative_dvh` rather than reimplementing DVH math

The adapter's own docstring states this directly: "This is NOT a key-rename... This adapter does that derivation for real, by building a `preprocessing.PatientStudy` from the GDP-HMM arrays and handing it to `preprocessing.build_slice_dataset` — the same function dvhnet's own DICOM-based path uses — rather than reimplementing the DVH math."

Concretely, the adapter never calls `compute_slice_cumulative_dvh` directly — it calls `build_slice_dataset`, which internally calls `compute_slice_cumulative_dvh` per slice (see Part 1):

```python
from preprocessing import PatientStudy, PreprocessConfig, build_slice_dataset  # noqa: E402
from dataset import save_patient_shard  # noqa: E402
...
study = PatientStudy(
    patient_id=patient_id,
    ct_volume=ct_volume,
    dose_volume=dose_volume,
    target_mask=target_mask,
    oar_masks={oar_name: oar_mask},
    spacing=spacing,
)

examples = build_slice_dataset(study, cfg, oar_name)
```

This is a meaningful design choice: it guarantees that a GDP-HMM-derived shard's DVH labels are computed with **the exact same numerical procedure** as a DICOM-derived shard's — same bin edges, same `>=` threshold convention, same monotonic-by-construction curve — so a model trained on a mix of both sources sees genuinely identical label semantics, not two independently-implemented approximations of "the same" quantity that might subtly disagree.

### Confirmed differences from the DICOM path, and why each one is necessary

**No dose/CT origin alignment needed:**

```python
# `img` and `dose` are ALREADY on the same [Z,H,W] grid (GDP-HMM's own
# DICOM2NPZ.py resamples dose onto the CT grid before packaging) -- unlike
# dvhnet's own DICOM path, this adapter does NOT need
# preprocessing.align_dose_to_ct at all.
```

Because GDP-HMM pre-resamples dose onto the CT grid as part of its own packaging pipeline (confirmed by the adapter's docstring, which cites the upstream `DICOM2NPZ.py` script by name), the adapter can build a `PatientStudy` directly from `img`/`dose` without ever calling `align_dose_to_ct` — a genuine simplification enabled by upstream work already having been done, not an oversight.

**Dose scaling — raw integer grid, not physical Gy directly:**

```python
def dose_in_gy(patient_dict: Dict) -> np.ndarray:
    """Real dose (Gy) = raw dose grid * dose_scale (DoseGridScaling)."""
    raw = patient_dict["dose"].astype(np.float64)
    scale = float(patient_dict["dose_scale"])
    return (raw * scale).astype(np.float32)
```

This mirrors the exact same pattern as DICOM RTDOSE's `DoseGridScaling` field in `preprocessing.load_rtdose` (Part 1) — GDP-HMM stores dose as an integer grid plus a separate scale factor rather than storing floating-point Gy values directly (likely for storage efficiency), and `dose_scale` is confirmed against the real sample data to be a very small number (`7.13e-05` in the inspected `HNC_001+A4Ac+MOS_25934.npz` file), consistent with a large-integer-range raw grid being scaled down into a physically small Gy range.

**Spacing axis-order conversion:**

```python
def gdp_hmm_spacing_to_dvhnet(spacing_xyz: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """GDP-HMM stores SimpleITK (x, y, z) spacing; dvhnet.preprocessing wants
    (dz, dy, dx) matching its [Z, H, W] array axis order."""
    sx, sy, sz = spacing_xyz
    return (sz, sy, sx)
```

SimpleITK's convention orders spacing as `(x, y, z)`, but `PatientStudy.spacing` is documented (Part 1) as `(dz, dy, dx)` to match the array's own `[Z, H, W]` axis order — a plain pass-through without this reversal would silently swap the z-spacing and x-spacing values used throughout resampling, which wouldn't crash (both are just floats) but would produce a geometrically wrong understanding of the volume's real-world dimensions.

**PTV/target resolution — no single canonical key exists:**

```python
PTV_KEY_CANDIDATES = ["PTV_Total", "PTV70", "PTVHighOPT", "PTV", "CTV"]

def resolve_target_mask(patient_dict: Dict,
                         candidates: List[str] = PTV_KEY_CANDIDATES) -> Tuple[np.ndarray, str]:
    for key in candidates:
        mask = patient_dict.get(key)
        if isinstance(mask, np.ndarray) and mask.sum() > 0:
            return (mask > 0).astype(np.uint8), key
    raise KeyError(...)
```

Confirmed against the real sample data: the file does contain `PTV_Total` (a union-across-dose-levels key), plus several dose-level-specific keys (`PTV70`, `PTVHighOPT`, `PTVLow`, `PTVLowOPT`, `RingPTVHigh`, `RingPTVLow`, etc.) rather than one single "PTV" key — this candidate-list-with-fallback approach handles that heterogeneity directly, always preferring the broadest union (`PTV_Total`) when it's present and non-empty, falling back progressively toward more specific single-level keys otherwise. `convert_patient`'s returned summary dict includes `target_key_used`, so which candidate actually resolved for a given patient is recorded and auditable rather than silently assumed.

### `dataset.save_patient_shard` reuse — matching the shard format exactly

```python
save_patient_shard(out_shard_dir, patient_id, examples)
shard_path = Path(out_shard_dir) / f"{patient_id}.npz"
```

The adapter imports and calls `dataset.save_patient_shard` directly (Part 1) rather than writing its own `.npz` serialization — this is what guarantees byte-for-byte structural compatibility with what `dataset.load_examples_from_shards` expects to read back (`target_masks`, `oar_masks`, `dvh_labels`, `voxel_counts`, `slice_indices` arrays, one file per patient), with zero risk of the two independently drifting in field names, array shapes, or dtypes. This was confirmed to actually work, not just assumed to work: `TODO.md` records that the two real shards produced by this adapter (`dvhnet/shards/Parotids/HNC_001_A4Ac.npz`, `HNC_001_9Ag.npz`) were "format-verified against what `dataset.load_examples_from_shards` expects (monotonic DVH curves, correct field names/shapes)" and were then actually consumed successfully by `train.py` end-to-end, including a full test-set evaluation.

### The real OAR naming scheme — confirmed directly against sample data, and the mismatch with `PreprocessConfig.oar_names`

This is a genuine, currently-unresolved naming inconsistency in the codebase, and it's worth documenting precisely rather than glossing over.

**What the real GDP-HMM data actually contains** (confirmed directly against `GDP-HMM_AAPMChallenge/data/HNC_001+A4Ac+MOS_25934.npz` while writing this documentation):

```
BrainStem, BrainStem_03, SpinalCord, SpinalCord_05, Parotids, Parotids-PTV,
ParotidCon-PTV, ParotidIps-PTV, Mandible, Mandible-PTV, Larynx, Larynx-PTV,
Esophagus, OralCavity, OCavity-PTV, Lens_L, Lens_R, OpticNerve_L, OpticNerve_R,
Chiasm, Cochlea_L, Cochlea_R, Brain, BrachialPlexus, Eyes, LacrimalGlands,
Lips, Lungs, Pituitary, Posterior_Neck, Shoulders, Submandibular,
Submand-PTV, SubmandL-PTV, SubmandR-PTV, Thyroid, Thyroid-PTV, Trachea,
Body, PTV_Total, PTV70, PTVHighOPT, PTVLow, PTVLowOPT, PTVLow-PTVMid,
RingPTVHigh, RingPTVLow
```

Note in particular: **both** `BrainStem` and `BrainStem_03` exist as separate keys in the same file (two distinct mask arrays, not a naming variant of the same one), likewise **both** `SpinalCord` and `SpinalCord_05`, and there is a single combined **`Parotids`** key rather than separate left/right parotid masks.

**What `dvhnet/preprocessing.py`'s `PreprocessConfig.oar_names` currently lists** (Part 1):

```python
oar_names: List[str] = field(default_factory=lambda: [
    "Brainstem", "SpinalCord", "ParotidL", "ParotidR", "Mandible",
    "Larynx", "Esophagus", "OralCavity", "Lens_L", "Lens_R",
    "OpticNerve_L", "OpticNerve_R", "Chiasm", "TemporalLobe_L",
    "TemporalLobe_R", "InnerEar_L",
])
```

**The mismatches, itemized:**
- `"Brainstem"` (lowercase "s") vs. real data's `"BrainStem"`/`"BrainStem_03"` (capital "S", and two variants).
- `"SpinalCord"` matches one of the two real variants, but not `"SpinalCord_05"`.
- `"ParotidL"`/`"ParotidR"` (separate laterality) vs. real data's single combined `"Parotids"` key — there is no left/right parotid split in the real data at all.
- `"TemporalLobe_L"`, `"TemporalLobe_R"`, `"InnerEar_L"` don't appear anywhere in the real sample's key list at all (the real data instead has `Cochlea_L`/`Cochlea_R`, `Brain`, and others not in this config list).

**How this is actually handled today — this is a resolved-by-bypass situation, not a silent bug, but it is a real gap worth flagging:** `gdp_hmm_adapter.convert_patient` takes `oar_name` as a direct CLI/function argument and validates it only against the specific npz file's own keys (`if oar_name not in d: raise KeyError(...)`) — it **never reads or consults `cfg.oar_names` at all** for OAR selection. `PreprocessConfig` is still constructed and passed through to `build_slice_dataset` (for `dose_max_gy`/`num_bins`/`target_matrix`/etc.), but its `oar_names` field is simply inert dead weight along this code path — the actual OAR key used end-to-end is whatever string the caller supplies (confirmed: `--oar Parotids` was the value actually used to produce the real shards on disk, matching the real data's key exactly, not any name from `PreprocessConfig.oar_names`). So the adapter itself has no bug here: it was written to use whatever the caller says, and the caller (per `TODO.md`, "OAR selection — **picked `Parotids` (combined)**") was set to the real key. **The actual open issue is that `PreprocessConfig.oar_names`'s default list is misleading/stale relative to GDP-HMM data** — anyone following `dvhnet/README.md`'s documented usage pattern (`for oar_name in cfg.oar_names: examples = build_slice_dataset(study, cfg, oar_name)`) against a GDP-HMM-sourced `PatientStudy` would silently get **zero shards for every OAR** (since none of those names match any of the real data's mask keys, and `build_slice_dataset` just returns an empty list when `study.oar_masks.get(oar_name)` is `None`) with no error raised anywhere. This is documented here as a known, currently-unresolved gap — see Open Items.

---

## Cross-Reference: What Reuses What

The following reuse relationships were confirmed by reading the actual current import statements and call sites in each file — nothing here is inferred from a plan or from memory of an earlier conversation.

| Consumer | Reuses | From | Mechanism |
|---|---|---|---|
| `dosenet3d/data_pipeline.py` | `resample_volume` | `dvhnet/preprocessing.py` | `importlib.util.spec_from_file_location` under private name `_dvhnet_preprocessing` (avoids the `losses.py`/`model.py` name collision — see Part 2) |
| `dosenet3d/evaluate.py` | `compute_slice_cumulative_dvh` | `dvhnet/preprocessing.py` | same `importlib` mechanism, private name `_dvhnet_preprocessing`; called on full 3D volumes, works unmodified because the function only does a boolean-mask flatten |
| `dosenet3d/evaluate.py` | `dose_at_volume`, `d2`, `d50`, `d_mean` | `dvhnet/metrics.py` | same `importlib` mechanism, private name `_dvhnet_metrics` — **confirmed actually implemented in current code**, not just planned |
| `dosenet3d/evaluate.py` | `DEFAULT_SERIAL_ORGANS` | `dosenet3d/losses.py` | plain `from losses import DEFAULT_SERIAL_ORGANS` (same package, no collision risk) — keeps loss-weighting and evaluation-classification from drifting apart after the casing-bug fix |
| `dosenet3d/synthetic_data.py` | `window_and_normalize_ct`, `build_ptv_channel`, `build_oar_channel`, `total_input_channels`, `DEFAULT_OAR_LABEL_MAP`, `normalize_dose` | `dosenet3d/data_pipeline.py` | plain same-package `import` — synthetic data exercises real channel-construction code, not a parallel reimplementation |
| `dosenet3d/train.py` | `fold_depth_into_batch`, `unfold_batch_to_depth` | `dosenet3d/model.py` | plain same-package `import` — augmentation's per-slice affine transform reuses the model's own batch/depth-folding convention so slice ordering can't drift between the two |
| `dvhnet/gdp_hmm_adapter.py` | `PatientStudy`, `PreprocessConfig`, `build_slice_dataset` | `dvhnet/preprocessing.py` | plain same-package `import` (adapter lives inside `dvhnet/`, no collision) — reuses the *exact* DVH-derivation function (`compute_slice_cumulative_dvh`, called transitively via `build_slice_dataset`), not a reimplementation |
| `dvhnet/gdp_hmm_adapter.py` | `save_patient_shard` | `dvhnet/dataset.py` | plain same-package `import` — guarantees the GDP-HMM-derived shard format is byte-structurally identical to the DICOM-derived one |
| `dvhnet/train.py` | `DVHSliceDataset`, `collate_fn`, `load_examples_from_shards`, `DVHNet`, `enforce_monotonic`, `DVHLoss`, `split_patients`, `aggregate_from_batch_outputs`, `evaluate_patient`, `summarize_results`, `check_acceptance_benchmarks` | `dvhnet/{dataset,model,losses,preprocessing,aggregate,metrics}.py` | plain same-package imports |
| **Colab notebook** | `dvhnet/train.py`, `dosenet3d/train.py` | GitHub clones of both repos | shells out via `subprocess`/`!` cell magics; does not import Python code directly — see Colab section below |

**Not (yet) implemented, despite being planned:** `dosenet3d/data_pipeline.py`'s `load_and_preprocess_patient` explicitly documents that it *should eventually* call `dvhnet.preprocessing.load_ct_series`, `load_rtdose`, `load_rtstruct_masks`, and `align_dose_to_ct` once GDP-HMM Hugging Face access lands — but today it only `raise NotImplementedError(...)`. This is a planned, not actual, cross-reference; see Open Items.

**A one-way dependency, not circular:** `dvhnet/` never imports anything from `dosenet3d/` — all reuse flows in the direction `dosenet3d → dvhnet`. This matches the project's own structure: `dvhnet` is the older, more complete pipeline that `dosenet3d` was built to reuse rather than duplicate.

---

## The Colab Pipeline (`Data Requirements and Preparation.ipynb`)

This notebook is the third pipeline. Despite its filename, its actual content (confirmed by reading every cell) is **not** about data requirements or preparation — it is titled internally "DVHnet / DoseNet3D — Colab GPU training runner" and its sole job is to run training remotely on a Colab GPU. See Open Items for this naming discrepancy.

**What it actually does, cell by cell:**

1. **Mounts Google Drive** at `/content/drive`, and creates a fixed directory layout under `DRIVE_ROOT = '/content/drive/MyDrive/dose_prediction_project'` (`checkpoints/`, `shards/`, `logs/`) — everything under Drive survives a Colab runtime recycle/disconnect, which free-tier Colab does aggressively (idle timeout, ~12h hard cap).
2. **Confirms a GPU is actually attached**, raising a `RuntimeError` with explicit remediation instructions if not — this fails loudly and immediately rather than letting training silently run on CPU for hours before someone notices.
3. **Clones (or pulls) `dvhnet` and `dosenet3d` from GitHub** into the ephemeral `/content/work/` scratch space:
   ```python
   GIT_REMOTE_DVHNET = 'https://github.com/krank-09/dvhnet.git'
   GIT_REMOTE_DOSENET3D = 'https://github.com/krank-09/dosenet3d.git'
   ```
   This is the load-bearing assumption behind the whole notebook: **it only ever runs whatever is currently pushed to GitHub**, never anything edited locally-only. Its own Notes section states this explicitly: "Keep repos as the source of truth, not this notebook... don't hand-edit `train.py` inside Colab, or local and remote will drift." Since neither `dvhnet/` nor `dosenet3d/` currently has a configured `origin` remote confirmed to point at those exact URLs (they are independent local git repos — see Open Items), this notebook cannot actually run successfully as-is until that's set up.
4. **Installs dependencies** (`pydicom`, `rt-utils`, `huggingface_hub`, `scipy`) — deliberately does *not* pin `torch`/`torchvision`/`numpy` versions, since Colab's runtime image ships its own compatible versions of those already.
5. **Syncs shards from Drive to local `/content/scratch`** via `rsync`, because reading/writing many small `.npz` files directly on a mounted Drive filesystem is slow — training reads from fast local scratch, while checkpoints (few, large files) are written straight back to Drive.
6. **Runs `dvhnet/train.py`** with `--shard_dir`, `--oar`, `--epochs 100`, `--batch_size 32`, `--out_dir` pointed directly at a Drive path — so `train.py`'s own checkpoint/history/test-summary saves land in persistent storage with no separate copy step needed.
7. **Runs `dosenet3d/train.py`** similarly, though the cell's own comment admits this call is speculative ("Adjust flags to whatever `train.py`'s actual CLI ends up being"). **This is now stale**: the notebook's cell invokes `dosenet3d/train.py --data_dir ... --epochs 50 --batch_size 1 --out_dir ...`, but the actual current `dosenet3d/train.py` CLI (confirmed in Part 2) has **no `--data_dir` argument at all** — its actual arguments are `--epochs`, `--batch_size`, `--lr`, `--weight_decay`, `--lambda_gradient`, `--lambda_organ`, `--n_synthetic_patients`, `--seed`, `--out_dir`, and it only ever trains against `synthetic_data.SyntheticDoseDataset` (real-data loading isn't wired up yet — see Part 2 and Open Items). This is a genuine drift between the notebook and the current script, documented here rather than papered over.
8. **Verifies checkpoints actually landed on Drive** with an `ls`/`find -newermt` sanity check, because the notebook's own Notes flag that "Drive writes can silently lag under heavy I/O."

**Design rationale documented in the notebook's own markdown cells:** local Claude Code stays the development environment for writing/debugging code and running CPU smoke tests (exactly the smoke tests documented in Parts 1–2); this notebook exists purely as a remote GPU executor for full-scale training runs once the code is validated locally — it is explicitly not meant to be a place code gets written or edited.

---

## Open Items / Known Gaps

Everything below was found while writing this documentation by reading the actual current files and comparing them against each other, against `TODO.md`, and against real sample data — none of it is smoothed over to make the documentation look cleaner than the code currently is.

1. **`PreprocessConfig.oar_names` (dvhnet/preprocessing.py) does not match real GDP-HMM naming, and nothing currently enforces or reconciles that.** Real data uses `BrainStem`/`BrainStem_03`, `SpinalCord`/`SpinalCord_05`, and a single combined `Parotids` key; the config lists `Brainstem`, `SpinalCord`, `ParotidL`, `ParotidR`, plus several names (`TemporalLobe_L/R`, `InnerEar_L`) that don't appear in the real sample at all. `gdp_hmm_adapter.py` bypasses this entirely by taking `--oar` as a direct argument, so today's actual shard production isn't broken by it — but the config's default value is actively misleading if anyone follows `dvhnet/README.md`'s own documented `for oar_name in cfg.oar_names: ...` loop pattern against GDP-HMM-sourced data: it would silently produce zero shards, for every OAR, with no error. This is a resolved-by-bypass situation for the adapter specifically, but an unresolved config/documentation mismatch overall.

2. **The `dosenet3d`-side Colab training cell is stale relative to the actual `train.py` CLI.** The notebook invokes `python train.py --data_dir {LOCAL_SHARD_DIR}_3d ...`, but the current script has no `--data_dir` flag and only trains against synthetic data (`--n_synthetic_patients`). The notebook's own comment ("Adjust flags to whatever `train.py`'s actual CLI ends up being") acknowledges this was written speculatively; it has not been updated since `dosenet3d/train.py` was actually finalized.

3. **GDP-HMM real-data loading for `dosenet3d` is explicitly not implemented.** `data_pipeline.load_and_preprocess_patient` raises `NotImplementedError` and `TODO.md` marks "extend the GDP-HMM adapter to also emit `dosenet3d`'s tensor format" as explicitly deferred pending Hugging Face access. Everything in Part 2 that isn't `synthetic_data.py` itself is real, production logic — but it has never yet been exercised against real data end-to-end, only synthetic.

4. **The dvhnet-side default train/val/test split silently breaks with very small cohorts.** `split_patients`'s default `train_frac=0.8, val_frac=0.1` rounds to zero val/test patients when the shard directory has only 1–2 patients (today's actual state: 2 real shards), and `train.py`'s final test-set evaluation then operates on an empty loader. This was worked around for the one real smoke test that's been run (`--train_frac 0.5 --val_frac 0.0`), but the underlying crash mode for small cohorts is not otherwise guarded against in `train.py` itself.

5. **The Colab notebook's own repo-remote URLs may not be live/configured.** `GIT_REMOTE_DVHNET`/`GIT_REMOTE_DOSENET3D` point at specific GitHub URLs the notebook expects to already exist and be up to date; whether `dvhnet/` and `dosenet3d/`'s local git repos actually have a matching `origin` configured and pushed was not verified as part of writing this documentation and should be checked before relying on the Colab notebook to work as-is.

6. **The Colab notebook's filename doesn't match its content.** It's named `Data Requirements and Preparation.ipynb`, but its actual first markdown cell titles it "DVHnet / DoseNet3D — Colab GPU training runner," and every cell is about cloning repos and driving remote GPU training — nothing in the notebook addresses data requirements or data preparation. This is a naming/content mismatch worth fixing (rename the file) rather than a functional bug.

7. **`train.py`'s `--resume_from` checkpoint-resume flag doesn't exist yet.** The Colab notebook's own Notes section flags this directly: "add a `--resume_from <checkpoint.pt>` flag to `train.py` if it doesn't already support one... `train.py` as currently written always starts fresh." Confirmed: neither `dvhnet/train.py` nor `dosenet3d/train.py` has any such argument today. For a free-tier Colab session that can disconnect mid-run, this means a dropped session cannot currently resume from its last checkpoint without code changes.

8. **`gdp_hmm_adapter.py` has only been run against substitute patients, not the originally-targeted ones.** `TODO.md` records that HF access for the originally-targeted patients (`0522c0001`/`0522c0003`) is still pending, so all real-adapter validation so far (`HNC_001`'s two plan variants) used locally-available substitute patients instead. The adapter's logic has been exercised and format-verified end-to-end, but not yet against the specific patients originally scoped.

9. **DVHnet's real-data test-set metrics are not meaningful yet, by design.** The one real end-to-end run produced D2 MAE 7.5 Gy / D50 MAE 38.8 Gy / Dmean MAE 5.2 Gy against the acceptance thresholds (≤1.0 Gy Dmean, ≤2.5 Gy D2) — both flagged `false`, as expected for 5 epochs on 1 training patient. This is correctly documented in `TODO.md` and the run's own JSON as a pipeline-correctness check, not an accuracy result, but it's worth restating here so these numbers are never later mistaken for a real accuracy benchmark.

10. **"After both tracks are checked out" (TODO.md) — a further pipeline is planned but not yet scoped.** The TODO file's own final section states this directly: a next stage is planned on top of dvhnet + dosenet3d, but its shape hasn't been defined. Nothing in the current codebase implements or anticipates it beyond that one-line note.
