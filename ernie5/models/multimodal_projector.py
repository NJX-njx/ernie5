from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from ernie5.configs.model_config import ERNIE5Config
from ernie5.models.audio_encoder import AudioDepthwiseEncoder
from ernie5.models.visual_encoder import DualPathVisualEncoder


class MultiModalProjector(nn.Module):
    """
    统一多模态投影层

    负责将各模态的原始输入或中间特征，映射到 LLM 的 Unified Token Space。

    输入:
    - Text: Input IDs -> Embeddings (Done in Backbone)
    - Image/Video: 像素 -> VisualEncoder -> Projection -> LLM Embedding
    - Audio: Codes -> AudioEncoder -> Projection -> LLM Embedding

    在 ERNIE 5.0 中，"Token Space" 是统一的。
    这意味着我们需要将通过Encoder提取的连续特征 (Continuous Embeddings)
    对齐到 Transformer 的 Hidden Size。
    """

    def __init__(self, config: ERNIE5Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        # 1. Visual Projector (Encoder + Linear)
        # Visual Encoder 输出通常维度较小或特定，需要投影到 hidden_size
        self.visual_encoder = DualPathVisualEncoder(
            config.visual, out_dim=self.hidden_size
        )

        # 2. Audio Projector (Encoder + Linear)
        self.audio_encoder = AudioDepthwiseEncoder(
            config.audio, out_dim=self.hidden_size
        )

        # 模态类型标识 (用于内部逻辑，实际输入是混合序列)

    def forward_visual(self, images: torch.Tensor) -> torch.Tensor:
        """
        处理图像/视频输入
        images: [B, C, H, W] or [B, T, C, H, W]
        """
        # 如果是视频 [B, T, C, H, W]，此处简化处理
        if images.dim() == 5:
            b, t, c, h, w = images.shape
            images = images.view(b * t, c, h, w)
            features = self.visual_encoder(images)  # [B*T, Seq, D]
            features = features.view(b, t * features.size(1), -1)
        else:
            features = self.visual_encoder(images)

        return features

    def forward_audio(self, audio_codes: torch.Tensor) -> torch.Tensor:
        """
        处理音频Code输入
        audio_codes: [B, Layers, T]
        """
        return self.audio_encoder(audio_codes)
