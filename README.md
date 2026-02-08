# ERNIE 5.0 PyTorch Implementation

这是一个基于 ERNIE 5.0 (2025) 技术报告的简化版 PyTorch 实现。它包含了一个统一多模态自回归模型的核心组件，支持混合专家 (MoE) 架构和弹性训练。

## 核心特性

- **Ultra-Sparse MoE**: 实现了超稀疏路由与共享专家机制。
- **Unified Multimodal**: 包含了视觉、音频和文本的统一 Token 化与生成流程。
- **Elastic Training**: 支持深度、宽度和稀疏度的动态弹性训练。
- **Next-Group-of-Tokens Prediction**: 实现了统一的多模态预测目标。

## 目录结构

```
ernie5/
├── configs/       # 模型、训练和 Tokenizer 配置
├── data/          # 数据集、采样器和 Collator
├── models/        # 核心模型架构 (Transformer, MoE, Encoders, Heads)
├── tokenizers/    # 多模态 Tokenizer (Text, Visual, Audio)
└── training/      # 训练循环、Loss、Scheduler 和 RL
```

## 快速开始

### 1. 安装依赖

```bash
pip install torch numpy tqdm tokenizers
```

### 2. 运行预训练模拟

我们提供了一个脚本来演示如何加载配置并启动训练循环。

```bash
python run_pretrain.py --scale mini
```

这将初始化一个 "mini" 规模的模型 (约 100M 参数) 并准备好 Trainer。

### 3. 配置模型

你可以通过修改 `ernie5/configs/model_config.py` 中的 `ERNIE5Config` 来调整模型架构，例如层数、专家数、隐藏维度等。

```python
from ernie5.configs import ERNIE5Config, ModelScale

# 加载预设配置 (Mini, Small, Medium, Large, Full)
config = ERNIE5Config.from_scale(ModelScale.SMALL)

# 自定义配置
config.moe.num_experts = 32
config.moe.top_k = 4
```

## 实现细节

### MoE 路由
位于 `ernie5/models/moe.py`。实现了无辅助损失的 Top-K 路由策略。

### 视觉编码
位于 `ernie5/models/visual_encoder.py`。采用了双路径设计：一条路径使用 CNN 提取细粒度特征，另一条路径使用 ViT 提取语义特征。

### 音频生成
位于 `ernie5/models/audio_generator.py`。实现了深度方向自回归循环，逐层预测残差 RVQ 码。

## 引用

Based on: *ERNIE 5.0: A Unified Autoregressive Foundation Model for Understanding and Generation*
