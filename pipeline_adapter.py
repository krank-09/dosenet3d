"""
gdp_hmm_adapter.py
===================
Real-data GDP-HMM npz -> dosenet3d's [C,H,W,D] input / [1,H,W,D] target
tensors, replacing synthetic_data.py as the training data source. Mirrors
dvhnet/gdp_hmm_adapter.py's npz-reading conventions (same raw arr_0-pickle
format, same PTV_DICT.json/Pat_Obj_DICT.json metadata resolution) but feeds
data_pipeline.preprocess_arrays instead of deriving DVH curves.

Key departure from data_pipeline.py's DEFAULT_OAR_LABEL_MAP: that map was
written for the synthetic smoke test and assumes separate Parotid_L/
Parotid_R structures. Real GDP-HMM head-and-neck patients only ship a single
combined "Parotids" mask (confirmed against real files -- see
dvhnet/gdp_hmm_adapter.py's docstring and TODO.md), so this adapter uses its
own GDP_HMM_OAR_LABEL_MAP = {"BrainStem": 1, "SpinalCord": 2, "Parotids": 3}.

PTV masks/doses are resolved for every available level (PTV_High/Mid/Low)
via PTV_DICT.json when the patient is present there, falling back to a
single guessed key (unknown prescription dose) otherwise -- same pattern as
dvhnet's adapter. Mask dict keys returned by __getitem__ are canonicalized
to a FIXED set (PTV_High/PTV_Mid/PTV_Low + the OAR label map's names) with
zero-filled entries for structures absent in a given patient, so the
default DataLoader collate_fn can batch mismatched patients (a mask that's
all-zero contributes nothing to RegionWeightedOrganLoss -- guarded there,
not a divide-by-zero -- so zero-filling is a no-op on the loss, not a
correctness compromise).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object

from data_pipeline import PreprocessConfig, preprocess_arrays

GDP_HMM_OAR_LABEL_MAP: Dict[str, int] = {"BrainStem": 1, "SpinalCord": 2, "Parotids": 3}
PTV_LEVELS = ("PTV_High", "PTV_Mid", "PTV_Low")
CANONICAL_MASK_KEYS = list(PTV_LEVELS) + list(GDP_HMM_OAR_LABEL_MAP.keys())

# Fallback preference order when a patient isn't in PTV_DICT.json -- no
# per-level split possible, so it's always treated as PTV_High with an
# unknown (guessed) prescription dose.
PTV_KEY_CANDIDATES = ["PTV_Total", "PTV70", "PTVHighOPT", "PTV", "CTV"]
FALLBACK_PRESCRIPTION_GY = 70.0  # only used when metadata doesn't have this patient

DEFAULT_META_DIR = Path(__file__).resolve().parent.parent.parent / "pipeline" / "meta_files"


def load_gdp_hmm_npz(path: str) -> Dict:
    npz = np.load(path, allow_pickle=True)
    return dict(npz)["arr_0"].item()


def load_ptv_dict(meta_dir: Optional[str]) -> Optional[Dict]:
    if meta_dir is None:
        return None
    ptv_path = Path(meta_dir) / "PTV_DICT.json"
    if not ptv_path.exists():
        return None
    return json.loads(ptv_path.read_text())


def meta_patient_id_from_npz_path(npz_path: str) -> str:
    return Path(npz_path).stem.split("+")[0]


def dose_in_gy(patient_dict: Dict) -> np.ndarray:
    raw = patient_dict["dose"].astype(np.float64)
    scale = float(patient_dict["dose_scale"])
    return (raw * scale).astype(np.float32)


def gdp_hmm_spacing_to_zyx(spacing_xyz: Tuple[float, float, float]) -> Tuple[float, float, float]:
    sx, sy, sz = spacing_xyz
    return (sz, sy, sx)


def resolve_ptv_masks_and_doses(d: Dict, meta_patient_id: str,
                                 ptv_dict: Optional[Dict]) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Returns ({level_name: binary mask}, {level_name: prescription_dose_gy})
    keyed by canonical level name (PTV_High/PTV_Mid/PTV_Low), not the raw
    per-patient OPTName -- so dict keys are uniform across patients."""
    masks: Dict[str, np.ndarray] = {}
    doses: Dict[str, float] = {}

    if ptv_dict is not None and meta_patient_id in ptv_dict:
        entry = ptv_dict[meta_patient_id]
        for level in PTV_LEVELS:
            if level not in entry:
                continue
            key = entry[level]["OPTName"]
            m = d.get(key)
            if isinstance(m, np.ndarray) and m.sum() > 0:
                masks[level] = (m > 0).astype(np.uint8)
                doses[level] = float(entry[level]["PDose"])

    if not masks:
        for cand in PTV_KEY_CANDIDATES:
            m = d.get(cand)
            if isinstance(m, np.ndarray) and m.sum() > 0:
                masks["PTV_High"] = (m > 0).astype(np.uint8)
                doses["PTV_High"] = FALLBACK_PRESCRIPTION_GY
                break

    if not masks:
        raise KeyError(f"no PTV/target mask resolvable (tried metadata for "
                        f"{meta_patient_id!r} and fallback candidates {PTV_KEY_CANDIDATES})")

    return masks, doses


def convert_patient(npz_path: str, cfg: Optional[PreprocessConfig] = None,
                     meta_dir: Optional[str] = DEFAULT_META_DIR,
                     oar_label_map: Optional[Dict[str, int]] = None,
                     ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict]:
    """
    Convert one GDP-HMM patient .npz -> (input [C,H,W,D], target [1,H,W,D],
    masks {canonical_name: [H,W,D]}, summary dict).
    """
    cfg = cfg or PreprocessConfig()
    oar_label_map = oar_label_map or GDP_HMM_OAR_LABEL_MAP
    d = load_gdp_hmm_npz(npz_path)

    meta_patient_id = meta_patient_id_from_npz_path(npz_path)
    ptv_dict = load_ptv_dict(meta_dir)
    ptv_masks, prescription_doses = resolve_ptv_masks_and_doses(d, meta_patient_id, ptv_dict)

    oar_masks = {name: (d[name] > 0).astype(np.uint8) for name in oar_label_map
                 if name in d and isinstance(d[name], np.ndarray)}
    if not oar_masks:
        raise KeyError(f"none of the OAR label map's structures "
                        f"{list(oar_label_map)} present in {npz_path!r}")

    ct_volume = d["img"].astype(np.float32)
    dose_volume = dose_in_gy(d)
    spacing = gdp_hmm_spacing_to_zyx(tuple(d["spacing"]))
    body_raw = d.get("Body")
    body_mask = (body_raw > 0).astype(np.uint8) if isinstance(body_raw, np.ndarray) else None

    # preprocess_arrays keys masks by the dict keys passed in (ptv_masks are
    # already canonicalized to PTV_High/Mid/Low above; oar_masks by their
    # real structure name, which already matches oar_label_map's keys).
    input_tensor, target_tensor, masks_hwd = preprocess_arrays(
        ct_volume, ptv_masks, oar_masks, dose_volume, spacing, cfg,
        prescription_doses, oar_label_map, body_mask_zhw=body_mask,
    )

    # Canonicalize to the FIXED key set for safe cross-patient collation:
    # zero-fill anything this patient doesn't have (RegionWeightedOrganLoss
    # skips all-zero masks, so this is a no-op on the loss, not a fudge).
    ref_shape = target_tensor.shape[1:]  # [H, W, D]
    canonical_masks = {
        name: masks_hwd.get(name, np.zeros(ref_shape, dtype=np.float32))
        for name in CANONICAL_MASK_KEYS
    }

    summary = {
        "meta_patient_id": meta_patient_id,
        "npz_source": npz_path,
        "ptv_levels_used": list(ptv_masks.keys()),
        "prescription_doses_gy": prescription_doses,
        "oars_present": list(oar_masks.keys()),
        "input_shape": tuple(input_tensor.shape),
        "target_shape": tuple(target_tensor.shape),
    }
    return input_tensor, target_tensor, canonical_masks, summary


class GDPHMMDoseDataset(Dataset):
    """torch Dataset over real GDP-HMM patient .npz files -- replaces
    SyntheticDoseDataset as dosenet3d's training data source. Each item is
    preprocessed on the fly (no separate shard-building step); this is a
    real, non-trivial per-item cost ([Z,H,W] resample + [3,256,256,80]
    tensor construction), so a real training run should use num_workers>0."""

    def __init__(self, npz_paths: List[str], cfg: Optional[PreprocessConfig] = None,
                 meta_dir: Optional[str] = DEFAULT_META_DIR,
                 oar_label_map: Optional[Dict[str, int]] = None):
        if torch is None:  # pragma: no cover
            raise ImportError("torch is required for GDPHMMDoseDataset")
        self.npz_paths = list(npz_paths)
        self.cfg = cfg or PreprocessConfig()
        self.meta_dir = meta_dir
        self.oar_label_map = oar_label_map or GDP_HMM_OAR_LABEL_MAP

    def __len__(self) -> int:
        return len(self.npz_paths)

    def __getitem__(self, idx: int) -> Dict:
        # Some .npz files in the GDP-HMM train_01/train_02 archives can be
        # corrupt (observed: zlib "invalid block type" mid-decompress) or
        # missing an expected structure. A single bad file must not kill a
        # multi-hour run, so on any failure we log it and fall through to the
        # next index (wrapping), bounded so an all-bad dataset still raises.
        n = len(self.npz_paths)
        last_err = None
        for offset in range(min(n, 25)):
            j = (idx + offset) % n
            path = self.npz_paths[j]
            try:
                patient_id = meta_patient_id_from_npz_path(path)
                input_arr, target_arr, masks, _summary = convert_patient(
                    path, self.cfg, self.meta_dir, self.oar_label_map)
                masks_t = {name: torch.from_numpy(m) for name, m in masks.items()}
                return {
                    "input": torch.from_numpy(input_arr),
                    "target": torch.from_numpy(target_arr),
                    "patient_id": patient_id,
                    "masks": masks_t,
                }
            except Exception as e:  # noqa: BLE001 -- deliberately broad
                last_err = e
                print(f"[dataset skip] {path}: {type(e).__name__}: {e}", flush=True)
        raise RuntimeError(f"25 consecutive dataset items failed near idx {idx}; "
                           f"last error: {last_err}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz_path", required=True)
    parser.add_argument("--meta_dir", default=str(DEFAULT_META_DIR))
    args = parser.parse_args()

    inp, tgt, masks, summary = convert_patient(args.npz_path, meta_dir=args.meta_dir or None)
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("mask keys:", list(masks.keys()))
    print("input dose channel range check -- CT[0,1]:", inp[0].min(), inp[0].max())
    print("target dose range (Gy):", tgt.min(), tgt.max())
