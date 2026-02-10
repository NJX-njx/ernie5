import pytest

torch = pytest.importorskip("torch")

from ernie5.training.scheduler import WSDScheduler


def test_wsd_scheduler_phases():
    param = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([param], lr=1.0)
    scheduler = WSDScheduler(
        optimizer,
        num_warmup_steps=2,
        num_stable_steps=2,
        num_decay_steps=4,
        min_lr_ratio=0.2,
    )

    lrs = []
    for _ in range(9):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    # warmup
    assert lrs[0] < lrs[1]
    # stable plateau
    assert abs(lrs[2] - lrs[3]) < 1e-8
    # decay then floor
    assert lrs[4] >= lrs[5] >= lrs[6]
    assert lrs[-1] == 0.2
