"""
ERNIE 5.0 - 统一多模态基础模型简化实现
基于百度文心5.0技术报告的PyTorch实现

核心特性:
- 超稀疏混合专家架构（Ultra-Sparse MoE）
- 统一多模态Token空间
- Next-Group-of-Tokens预测目标
- 弹性训练支持
"""

__version__ = "0.1.0"
__author__ = "ERNIE 5.0 Implementation"

# 仅导出轻量配置对象，避免在导入包时强依赖 torch。
from ernie5.configs import ERNIE5Config

__all__ = [
    "ERNIE5Config",
]
