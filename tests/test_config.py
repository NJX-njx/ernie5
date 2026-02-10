from ernie5.configs import ERNIE5Config, ModelScale


def test_unified_vocab_size_matches_offsets():
    cfg = ERNIE5Config.from_scale(ModelScale.MINI)
    visual_size = 2**cfg.visual.tokenizer_bits
    assert cfg.visual_vocab_offset == cfg.vocab_size
    assert cfg.audio_vocab_offset == cfg.vocab_size + visual_size
    assert (
        cfg.get_unified_vocab_size() == cfg.audio_vocab_offset + cfg.audio.codebook_size
    )
