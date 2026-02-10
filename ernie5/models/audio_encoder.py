from typing import List, Optional

import torch
import torch.nn as nn

from ernie5.configs.model_config import AudioConfig


class AudioDepthwiseEncoder(nn.Module):
    """
    音频理解的深度方向嵌入 (Audio Depthwise Embedding)

    原理：
    音频由多个残差量化层 (RVQ layers) 表示。
    我们将每一层 (Level) 的Code分别Embedding，然后以"加性"方式融合。
    Embedding = Sum(Embed(Code_level_i))

    这与将多个Code flatten成序列不同，这里是在Embedding维度上进行融合，
    保持了时间维度的长度不变 (Token rate = 12.5Hz)。
    """

    def __init__(self, config: AudioConfig, out_dim: int):
        super().__init__()
        self.config = config
        self.num_layers = config.rvq_layers
        self.codebook_size = config.codebook_size

        # 每层一个Embedding表
        # 或者共享Embedding表? 论文通常建议每层独立
        self.embeddings = nn.ModuleList(
            [nn.Embedding(self.codebook_size, out_dim) for _ in range(self.num_layers)]
        )

        # 可选：Layer weights (学习每层的重要性)
        self.layer_weights = nn.Parameter(torch.ones(self.num_layers))

        self.out_proj = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, audio_codes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_codes: [Batch, Num_Layers, Seq_Len]
                RVQ生成的Discrete Codes
        Returns:
            audio_embeddings: [Batch, Seq_Len, Out_Dim]
        """
        batch_size, num_layers, seq_len = audio_codes.shape

        # 确保输入层数不超过模型支持层数
        assert num_layers <= self.num_layers

        total_embedding = 0.0

        # 深度方向求和
        for i in range(num_layers):
            code_level = audio_codes[:, i, :]  # [B, T]
            emb = self.embeddings[i](code_level)  # [B, T, D]

            # 加权求和
            total_embedding = total_embedding + emb * self.layer_weights[i]

        total_embedding = self.norm(total_embedding)
        return self.out_proj(total_embedding)
