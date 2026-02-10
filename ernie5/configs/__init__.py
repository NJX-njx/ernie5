"""
ERNIE 5.0 配置模块

包含:
- 模型配置
- 训练配置
- Tokenizer配置
- 预设配置（简化版/完整版）
"""

from ernie5.configs.model_config import ERNIE5Config, ModelScale
from ernie5.configs.tokenizer_config import TokenizerConfig
from ernie5.configs.training_config import TrainingConfig

__all__ = [
    "ERNIE5Config",
    "ModelScale",
    "TrainingConfig",
    "TokenizerConfig",
]
