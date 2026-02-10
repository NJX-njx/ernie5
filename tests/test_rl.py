import pytest

torch = pytest.importorskip("torch")

from ernie5.training.rl import UnifiedMultiModalRL


def test_wpsm_filters_high_success_samples_and_keeps_unknowns():
    rl = UnifiedMultiModalRL()
    samples = [
        {"id": 1, "success_rate": 0.95},
        {"id": 2, "success_rate": 0.5},
        {"id": 3},
    ]
    kept = rl.wpsm_masking(samples, threshold=0.8)
    kept_ids = {x["id"] for x in kept}
    assert kept_ids == {2, 3}


def test_wpsm_keeps_at_least_one_sample_when_all_filtered():
    rl = UnifiedMultiModalRL()
    samples = [{"id": 1, "success_rate": 0.99}]
    kept = rl.wpsm_masking(samples, threshold=0.8)
    assert len(kept) == 1
    assert kept[0]["id"] == 1


def test_wpsm_threshold_validation():
    rl = UnifiedMultiModalRL()
    with pytest.raises(ValueError):
        rl.wpsm_masking([{"id": 1}], threshold=1.1)
