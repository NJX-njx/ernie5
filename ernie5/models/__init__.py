"""
ERNIE 5.0 模型架构模块

包含:
- Transformer Backbone
- 超稀疏MoE层
- 弹性训练支持
- 多模态理解与生成模块
"""

from ernie5.models.backbone import ERNIE5Model
from ernie5.models.transformer import TransformerBlock
from ernie5.models.moe import SparseMoELayer
from ernie5.models.attention import MultiHeadAttention
from ernie5.models.embeddings import UniRoPE, MultiModalEmbedding

__all__ = [
    "ERNIE5Model",
    "TransformerBlock",
    "SparseMoELayer",
    "MultiHeadAttention",
    "UniRoPE",
    "MultiModalEmbedding",
]
