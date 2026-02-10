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
        return {
            "<s>": 1,
            "</s>": 2,
        }.get(token)

    def encode(self, text):
        if text == "short":
            return _FakeEncoding([10, 11])
        return _FakeEncoding([10, 11, 12, 13, 14])


def test_collator_masks_padding_labels_to_ignore_index():
    cfg = ERNIE5Config.from_scale(ModelScale.MINI)
    tk = TextTokenizer(TokenizerConfig(vocab_file=None, utf16be_fallback=False))
    # 避免依赖真实 tokenizers 训练流程，直接替换 backend
    tk._tokenizer = _FakeTokenizerBackend()

    collator = MultiModalCollator(config=cfg, tokenizer=tk)
    batch = [
        {"type": "text", "content": "short"},
        {"type": "text", "content": "long"},
    ]

    out = collator(batch)
    labels = out["labels"]
    input_ids = out["input_ids"]

    pad_positions = input_ids.eq(cfg.pad_token_id)
    assert pad_positions.any()
    assert torch.all(labels[pad_positions].eq(-100))
