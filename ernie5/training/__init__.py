"""
ERNIE 5.0 训练模块

包含:
- 预训练循环
- 多阶段学习率调度
- 弹性训练策略
- 分布式训练支持
- 统一多模态RL
"""

from ernie5.training.trainer import ERNIE5Trainer
from ernie5.training.scheduler import WSDScheduler, CosineScheduler
from ernie5.training.elastic import ElasticTrainingManager
from ernie5.training.losses import NextGroupTokenLoss
from ernie5.training.rl import UnifiedMultiModalRL

__all__ = [
    "ERNIE5Trainer",
    "WSDScheduler",
    "CosineScheduler",
    "ElasticTrainingManager",
    "NextGroupTokenLoss",
    "UnifiedMultiModalRL",
]
