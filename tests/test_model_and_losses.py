import torch

from losses import DEFAULT_PARALLEL_WEIGHT, DEFAULT_SERIAL_WEIGHT, CompositeDoseLoss, RegionWeightedOrganLoss
from model import DoseNet3D, fold_depth_into_batch, unfold_batch_to_depth
from train import build_oar_label_swap


def test_forward_keeps_shape_and_dose_is_non_negative():
    torch.manual_seed(0)
    x = torch.rand(1, 3, 32, 32, 16)          # any H, W, D divisible by 8
    out = DoseNet3D(in_channels=3).eval()(x)
    assert out.shape == (1, 1, 32, 32, 16)
    assert (out >= 0).all()


def test_fold_unfold_round_trip():
    x = torch.rand(2, 3, 8, 8, 5)
    folded, b, d = fold_depth_into_batch(x)
    assert folded.shape == (10, 3, 8, 8)
    assert torch.equal(unfold_batch_to_depth(folded, b, d), x)


def test_loss_is_zero_for_perfect_prediction():
    dose = torch.rand(1, 1, 16, 16, 8) * 70
    masks = {"PTV_High": (dose[:, 0] > 50).float(), "SpinalCord": (dose[:, 0] < 10).float()}
    out = CompositeDoseLoss()(dose, dose, masks)
    assert out["loss"].item() == 0


def test_gradient_term_penalises_blurring():
    target = torch.zeros(1, 1, 16, 16, 8)
    target[..., 8:, :, :] = 60                 # sharp dose edge
    blurred = torch.nn.functional.avg_pool3d(target, 3, stride=1, padding=1)
    out = CompositeDoseLoss()(blurred, target, {})
    assert out["loss_gradient"] > 0


def test_serial_organs_weigh_more_than_parallel():
    loss = RegionWeightedOrganLoss()
    assert loss._weight_for("SpinalCord") == DEFAULT_SERIAL_WEIGHT
    assert loss._weight_for("BrainStem") == DEFAULT_SERIAL_WEIGHT
    assert loss._weight_for("Parotids") == DEFAULT_PARALLEL_WEIGHT


def test_left_right_labels_swap_on_flip():
    assert build_oar_label_swap({"Parotid_L": 1, "Parotid_R": 2, "SpinalCord": 3}) == {1: 2, 2: 1}
    assert build_oar_label_swap({"BrainStem": 1, "SpinalCord": 2, "Parotids": 3}) == {}
