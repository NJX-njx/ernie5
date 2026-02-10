import importlib


def test_package_exports_config_and_model_scale_without_torch_dependency():
    pkg = importlib.import_module("ernie5")
    configs = importlib.import_module("ernie5.configs")

    assert hasattr(pkg, "ERNIE5Config")
    assert hasattr(configs, "ModelScale")


def test_package_has_lazy_model_getattr():
    import sys

    # Remove torch from sys.modules if present to verify lazy loading
    torch_module_keys = [
        k for k in sys.modules if k == "torch" or k.startswith("torch.")
    ]
    torch_modules_backup = {k: sys.modules[k] for k in torch_module_keys}
    for module_name in torch_module_keys:
        del sys.modules[module_name]

    # Import ernie5 - should not import torch
    pkg = importlib.import_module("ernie5")
    assert "torch" not in sys.modules, "Importing ernie5 should not import torch"

    # Verify __getattr__ is defined
    assert hasattr(
        pkg, "__getattr__"
    ), "ernie5 should define __getattr__ for lazy loading"

    # Verify unknown attribute raises AttributeError
    try:
        getattr(pkg, "NOT_EXISTS")
    except AttributeError:
        pass
    else:
        raise AssertionError("Missing attribute should raise AttributeError")

    # Restore torch modules
    for module_name, module in torch_modules_backup.items():
        sys.modules[module_name] = module
