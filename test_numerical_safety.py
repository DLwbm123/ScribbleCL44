import numpy as np
import pytest
import torch
import torch.nn.functional as F

from numerical_safety import (
    NumericalAudit,
    NumericalFailure,
    checked_int32_cost,
    log_probability_resize,
    normalized_unary,
    rms_saliency,
    sparse_pce_from_log_probs,
    tensor_summary,
    validate_graph_labels,
)


torch.set_num_threads(1)


@pytest.mark.parametrize("value", [0.0, 1e-30, 1e20])
def test_rms_finite_for_extreme_finite_gradients(value):
    gradient = torch.full((2, 1, 8, 8), value)
    assert torch.equal(rms_saliency(gradient), gradient[:, 0].abs())


@pytest.mark.parametrize("value", [0.0, 1e-30, 1e20])
def test_unary_normalization_extreme_and_zero(value):
    unary, diagnostics = normalized_unary(torch.full((2, 8, 8), value), 4)
    assert torch.isfinite(unary).all()
    assert torch.allclose(unary.sum((-2, -1)), torch.ones(2))
    assert torch.allclose(unary, torch.full_like(unary, 0.25))
    assert diagnostics["zero_saliency_samples"] == (2 if value == 0 else 0)


def test_normal_unary_path_matches_original():
    torch.manual_seed(42)
    gradient = torch.rand(3, 16, 16)
    pooled = F.avg_pool2d(gradient, 4)
    original = pooled / pooled.sum((-2, -1), keepdim=True)
    safe, _ = normalized_unary(gradient, 4)
    assert torch.allclose(original, safe, atol=1e-7, rtol=1e-6)


def test_batch_with_one_empty_sample_has_only_one_fallback():
    gradient = torch.ones(3, 8, 8)
    gradient[1].zero_()
    result, diagnostics = normalized_unary(gradient, 4)
    assert diagnostics["zero_saliency_sample_indices"] == [1]
    assert torch.allclose(result.sum((-1, -2)), torch.ones(3))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_saliency_fails_instead_of_sanitizing(bad):
    with pytest.raises(FloatingPointError):
        normalized_unary(torch.full((1, 8, 8), bad), 4)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_graph_cost_rejects_nonfinite(bad):
    with pytest.raises(FloatingPointError):
        checked_int32_cost("unary", np.array([bad]))


def test_graph_cost_rejects_overflow():
    with pytest.raises(OverflowError):
        checked_int32_cost("unary", np.array([1e30]))


def test_normal_cost_preserves_original_integer_conversion():
    values = np.array([0.0, 1.25, -1.25, 1500.5])
    assert np.array_equal(checked_int32_cost("unary", values), values.astype(np.int32))


@pytest.mark.parametrize("bad", [[-1, 0], [0, 2], [0, 0.5]])
def test_graph_labels_must_be_valid_ids(bad):
    with pytest.raises(ValueError):
        validate_graph_labels(np.array(bad), 2)


@pytest.mark.parametrize("size", [(5, 7), (11, 13), (3, 4)])
def test_log_resize_matches_probability_interpolation_and_gradients(size):
    torch.manual_seed(42)
    first = torch.randn(2, 3, 5, 7, requires_grad=True)
    second = first.detach().clone().requires_grad_(True)
    old = F.interpolate(first.softmax(1), size=size, mode="bilinear", align_corners=False)
    old = old / old.sum(1, keepdim=True).clamp_min(1e-12)
    new = log_probability_resize(second, size)
    assert torch.allclose(old, new.exp(), atol=2e-7, rtol=2e-6)
    labels = torch.randint(0, 3, (2, *size), dtype=torch.long)
    labels[:, 0] = -100
    target = torch.stack([labels.eq(index) for index in range(3)], 1).float()
    old_loss = -(target * (old + 1e-12).log()).sum() / target.sum().clamp_min(1)
    new_loss = sparse_pce_from_log_probs(new, labels)
    old_gradient = torch.autograd.grad(old_loss, first)[0]
    new_gradient = torch.autograd.grad(new_loss, second)[0]
    assert torch.allclose(old_loss, new_loss, atol=1e-6, rtol=1e-6)
    assert torch.allclose(old_gradient, new_gradient, atol=2e-7, rtol=1e-4)


def test_saturated_wrong_prediction_keeps_corrective_pce_gradient():
    logits = torch.tensor([[[[100.]], [[-100.]]]], requires_grad=True)
    target = torch.ones((1, 1, 1), dtype=torch.long)
    loss = sparse_pce_from_log_probs(log_probability_resize(logits, (1, 1)), target)
    gradient = torch.autograd.grad(loss, logits)[0]
    assert loss.item() == 200.0
    assert torch.equal(gradient, torch.tensor([[[[1.]], [[-1.]]]]))


def test_all_ignore_pce_is_zero_with_zero_gradient():
    logits = torch.randn(1, 2, 4, 4, requires_grad=True)
    loss = sparse_pce_from_log_probs(
        log_probability_resize(logits, (4, 4)), torch.full((1, 4, 4), -100, dtype=torch.long)
    )
    gradient = torch.autograd.grad(loss, logits)[0]
    assert loss.item() == 0 and torch.equal(gradient, torch.zeros_like(gradient))


def test_multichannel_rms_stays_finite_and_matches_scale():
    gradient = torch.full((1, 3, 8, 8), 1e20)
    result = rms_saliency(gradient)
    assert torch.isfinite(result).all()
    assert torch.allclose(result, torch.full_like(result, 1e20))


def test_log_resize_for_native_244_to_256():
    torch.manual_seed(42)
    logits = torch.randn(1, 2, 244, 244, requires_grad=True)
    old = F.interpolate(logits.softmax(1), size=(256, 256), mode="bilinear", align_corners=False)
    new = log_probability_resize(logits, (256, 256)).exp()
    assert torch.allclose(old, new, atol=3e-5, rtol=1e-4)
    loss = sparse_pce_from_log_probs(
        log_probability_resize(logits, (256, 256)), torch.zeros(1, 256, 256, dtype=torch.long)
    )
    assert torch.isfinite(torch.autograd.grad(loss, logits)[0]).all()


def test_deferred_audit_checks_gradients_state_and_first_nan(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
    image = torch.ones(2, 1, 1, 1)
    label = torch.zeros(2, 1, 1, dtype=torch.long)
    audit = NumericalAudit(tmp_path, enabled=True)
    audit.begin(stage=0, epoch=0, iteration=1, task_id=0, image=image, label=label)
    loss = model(torch.ones(2, 2)).sum()
    audit.check("current_global", "loss", loss)
    audit.flush()
    loss.backward()
    audit.check_gradients(model)
    optimizer.step()
    audit.check_model_state(model, optimizer)
    audit.begin(stage=0, epoch=0, iteration=2, task_id=0, image=image, label=label)
    audit.check("replay_global", "injected_nan", torch.tensor(float("nan")))
    with pytest.raises(NumericalFailure, match="replay_global/injected_nan"):
        audit.flush()


def test_nonfinite_summary_keeps_scalar_metadata_json_safe():
    value = torch.tensor([torch.finfo(torch.float32).max, float("nan")])
    summary = tensor_summary(value)
    assert not summary["finite"]
    assert np.isfinite(summary["mean"])
