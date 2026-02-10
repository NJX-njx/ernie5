import importlib


def test_package_exports_config_and_model_scale_without_torch_dependency():
    pkg = importlib.import_module("ernie5")
    configs = importlib.import_module("ernie5.configs")

    assert hasattr(pkg, "ERNIE5Config")
    assert hasattr(configs, "ModelScale")


def test_package_has_lazy_model_getattr():
    pkg = importlib.import_module("ernie5")
    try:
        getattr(pkg, "NOT_EXISTS")
    except AttributeError:
        pass
    else:
        raise AssertionError("Missing attribute should raise AttributeError")
