"""Standalone soft dice must not treat raw logits as probabilities."""

import torch

from vesuvius.models.training.loss.losses import _create_loss
from vesuvius.models.training.loss.nnunet_losses import softmax_helper_dim1


def _logits_and_target():
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 32, 32, 32) * 0.3 - 0.15
    target = (torch.rand(2, 1, 32, 32, 32) < 0.15).float()
    return logits, target


def test_omitted_nonlin_keeps_negative_logits_finite():
    logits, target = _logits_and_target()
    loss = _create_loss("MemoryEfficientSoftDiceLoss", {}, 1.0, None, None)

    assert loss.apply_nonlin is torch.sigmoid
    value = float(loss(logits, target))
    assert abs(value) < 2.0


def test_explicit_none_still_sees_raw_logits():
    logits, target = _logits_and_target()
    loss = _create_loss(
        "MemoryEfficientSoftDiceLoss",
        {"apply_nonlin": "none"},
        1.0,
        None,
        None,
    )

    assert loss.apply_nonlin is None
    value = float(loss(logits, target))
    assert abs(value) > 1.0e6


def test_yaml_strings_select_sigmoid_or_softmax():
    from vesuvius.models.training.loss.losses import _resolve_dice_nonlin

    assert _resolve_dice_nonlin("sigmoid") is torch.sigmoid
    assert _resolve_dice_nonlin("softmax") is softmax_helper_dim1
    assert _resolve_dice_nonlin(None) is torch.sigmoid
