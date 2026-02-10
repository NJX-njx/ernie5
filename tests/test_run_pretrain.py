import pytest

from run_pretrain import parse_args, resolve_scale
from ernie5.configs import ModelScale


def test_parse_args_default_scale_is_mini():
    args = parse_args([])
    assert args.scale == "mini"


def test_parse_args_rejects_invalid_scale():
    with pytest.raises(SystemExit):
        parse_args(["--scale", "invalid"])


def test_resolve_scale_mapping():
    assert resolve_scale("mini") == ModelScale.MINI
    assert resolve_scale("small") == ModelScale.SMALL
    assert resolve_scale("medium") == ModelScale.MEDIUM
    assert resolve_scale("full") == ModelScale.FULL
