"""Real-data path on a fake patient packed the way GDP-HMM .npz files are."""

import numpy as np

from evaluate import evaluate_patient
from pipeline_adapter import CANONICAL_MASK_KEYS, convert_patient


def _fake_patient(path):
    z, h, w = 24, 64, 64
    zz, yy, xx = np.mgrid[:z, :h, :w]
    ptv = ((zz - 12) ** 2 / 25 + (yy - 32) ** 2 / 64 + (xx - 32) ** 2 / 64 < 1).astype(np.uint8)
    parotids = ((yy - 32) ** 2 / 16 + (xx - 52) ** 2 / 16 < 1) & (abs(zz - 12) < 4)
    dist = np.sqrt((yy - 32) ** 2 + (xx - 32) ** 2 + ((zz - 12) * 3) ** 2)
    dose = np.clip(70 - dist, 0, None)
    d = {"img": np.random.default_rng(0).normal(0, 300, (z, h, w)).astype(np.int16),
         "dose": (dose / 0.01).astype(np.int32), "dose_scale": 0.01,
         "spacing": (3.0, 3.0, 3.0), "PTV_Total": ptv, "Parotids": parotids.astype(np.uint8),
         "Body": np.ones((z, h, w), np.uint8)}
    np.savez(path, d)


def test_convert_patient_shapes_and_masks(tmp_path):
    f = tmp_path / "P001+plan.npz"
    _fake_patient(f)
    inp, tgt, masks, summary = convert_patient(str(f), meta_dir=str(tmp_path))  # no PTV_DICT → fallback
    assert inp.shape == (3, 256, 256, 80) and tgt.shape == (1, 256, 256, 80)
    assert set(masks) == set(CANONICAL_MASK_KEYS)
    assert masks["Parotids"].sum() > 0 and masks["SpinalCord"].sum() == 0   # absent → zero-filled
    assert summary["ptv_levels_used"] == ["PTV_High"]
    assert 60 < tgt.max() <= 70.01


def test_perfect_prediction_passes_every_check(tmp_path):
    f = tmp_path / "P001+plan.npz"
    _fake_patient(f)
    _, tgt, masks, _ = convert_patient(str(f), meta_dir=str(tmp_path))
    rep = evaluate_patient(tgt, tgt, {k: v for k, v in masks.items() if v.sum()}, 80.0, 70.0)
    assert rep["voxel_mae_gy"] == 0
    assert rep["structures"] and all(s.within_tolerance for s in rep["structures"])
