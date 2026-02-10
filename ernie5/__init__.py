"""
ERNIE 5.0 - 统一多模态基础模型简化实现
基于百度文心5.0技术报告的PyTorch实现。

设计说明:
- 顶层包默认只暴露轻量配置对象，避免在未安装 torch 时导入失败。
- 对模型类使用懒加载（__getattr__），兼顾易用性与可移植性。
"""

from ernie5.configs import ERNIE5Config

__version__ = "0.1.0"
__author__ = "ERNIE 5.0 Implementation"

__all__ = [
    "ERNIE5Config",
    "ERNIE5Model",
    "ERNIE5ForCausalLM",
]


def __getattr__(name: str):
    """懒加载模型类，避免 import ernie5 时强依赖 torch。

    Raises:
        AttributeError: 当请求的属性不在支持列表中时抛出。
    """
    if name in {"ERNIE5Model", "ERNIE5ForCausalLM"}:
        from ernie5.models import ERNIE5Model, ERNIE5ForCausalLM

        return {
            "ERNIE5Model": ERNIE5Model,
            "ERNIE5ForCausalLM": ERNIE5ForCausalLM,
        }[name]
    raise AttributeError(f"module 'ernie5' has no attribute '{name}'")


def __dir__():
    """支持 dir() 和 IDE 自动补全。"""
    return __all__
