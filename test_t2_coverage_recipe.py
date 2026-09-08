"""Small checks for nested masks and finite clipped updates."""
import numpy as np
import torch
from types import SimpleNamespace
from prepare_t2_coverage import expand


def test_expansion():
    dense = np.zeros((2, 12, 12), dtype=np.int64)
    dense[0, 2:10, 2:10] = 1
    sparse = np.full_like(dense, -100)
    sparse[:, 0, :] = 0
    sparse[0, 5, 4:8] = 1
    expanded = expand(sparse, dense, .4)
    assert (expanded == 1).sum() == 26
    assert np.array_equal(expanded == 0, sparse == 0)
    assert np.array_equal(expanded[sparse != -100], sparse[sparse != -100])
    assert np.array_equal(expanded, expand(sparse, dense, .4))


def test_clipped_update():
    p = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.SGD([p], lr=.003, momentum=.9)
    for _ in range(4):
        optimizer.zero_grad()
        (p * torch.tensor([1e6, -1e6])).sum().backward()
        old = p.detach().clone()
        torch.nn.utils.clip_grad_norm_([p], 1., error_if_nonfinite=True)
        optimizer.step()
        assert torch.isfinite(p).all() and (p - old).norm() <= .003 * 4
    p.grad[:] = float("inf")
    try:
        torch.nn.utils.clip_grad_norm_([p], 1., error_if_nonfinite=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("non-finite update was accepted")


def test_disabled_replay_global_is_not_executed():
    import runner_core
    from zs_derpp_smoke import Tiny
    model = Tiny().train()
    image = torch.randn(2, 1, 8, 8)
    labels = torch.zeros(2, 8, 8, dtype=torch.long)
    replay = (image, image, labels, torch.zeros(2, dtype=torch.long), torch.full((2,), 2))
    original = runner_core.zs_cutout_invariance
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled global branch executed")
    runner_core.zs_cutout_invariance = forbidden
    try:
        pce, global_loss = runner_core.derpp_replay_losses(
            model, replay, "organ", SimpleNamespace(zs_global_weight=0.), torch.device("cpu"),
        )
        assert torch.isfinite(pce) and global_loss.item() == 0
        pce.backward()
        assert model.head.weight.grad is not None
    finally:
        runner_core.zs_cutout_invariance = original


if __name__ == "__main__":
    test_expansion()
    test_clipped_update()
    print("coverage and clipping checks PASS")
