import pytest

torch = pytest.importorskip("torch")

from ernie5.configs import ERNIE5Config, ModelScale
from ernie5.models import ERNIE5ForCausalLM


def test_causal_lm_loss_computes_without_tensor_truthiness_error():
    config = ERNIE5Config.from_scale(ModelScale.MINI)
    config.use_mtp = False

    model = ERNIE5ForCausalLM(config)
    model.eval()

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, seq_len))
    labels = input_ids.clone()
    text_mask = torch.ones_like(labels, dtype=torch.bool)

    loss, logits = model(input_ids=input_ids, labels=labels, text_mask=text_mask)

    assert loss is not None
    assert loss.ndim == 0
    assert logits.shape[:2] == (batch_size, seq_len)
