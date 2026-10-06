import numpy as np
import pytest

from dvh_utils import center_crop_or_pad, cumulative_dvh, d2, d50, d_mean, resample_volume

BIN = 80 / 255


def test_uniform_dose_metrics_are_analytic():
    dose = np.linspace(0, 60, 20001)
    y = cumulative_dvh(dose, np.ones_like(dose))
    assert d50(y, 80) == pytest.approx(30, abs=BIN)
    assert d2(y, 80) == pytest.approx(58.8, abs=BIN)
    assert d_mean(y, 80) == pytest.approx(30, abs=BIN)


def test_empty_mask_gives_none():
    assert cumulative_dvh(np.ones((4, 4)), np.zeros((4, 4))) is None


def test_resample_halves_voxel_count_per_axis():
    vol = np.random.default_rng(0).random((10, 20, 20))
    out = resample_volume(vol, (2, 2, 2), (4, 4, 4), order=1)
    assert out.shape == (5, 10, 10)


def test_crop_or_pad_is_centered():
    vol = np.ones((2, 4, 4))
    out = center_crop_or_pad(vol, (4, 2, 6))
    assert out.shape == (4, 2, 6)
    assert out[1:3, :, 1:5].all() and out.sum() == 2 * 2 * 4
