import pytest

torch = pytest.importorskip("torch")

from ernie5.configs import ERNIE5Config, ModelScale, TokenizerConfig
from ernie5.data import MultiModalCollator
from ernie5.tokenizers import TextTokenizer


class _FakeEncoding:
    def __init__(self, ids):
        self.ids = ids


class _FakeTokenizerBackend:
    def token_to_id(self, token):
        return {"<s>": 1, "</s>": 2}.get(token)

    def encode(self, text):
        return _FakeEncoding([10, 11, 12])


def test_text_attention_mask_is_causal_unmasked_false():
    cfg = ERNIE5Config.from_scale(ModelScale.MINI)
    tk = TextTokenizer(TokenizerConfig(vocab_file=None))
    tk._tokenizer = _FakeTokenizerBackend()

    collator = MultiModalCollator(config=cfg, tokenizer=tk)
    out = collator([{"type": "text", "content": "hello"}])

    attn = out["attention_mask"][0]
    seq_len = out["input_ids"].shape[1]

    for i in range(seq_len):
        for j in range(seq_len):
            if j <= i:
                assert attn[i, j].item() is False
            else:
                assert attn[i, j].item() is True
