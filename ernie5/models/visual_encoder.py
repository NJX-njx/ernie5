import torch
import torch.nn as nn
import torch.nn.functional as F

from ernie5.configs.model_config import VisualConfig


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: [B, C, H, W] -> [B, Embed, Grid, Grid] -> [B, Embed, NumPatches] -> [B, NumPatches, Embed]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class DualPathVisualEncoder(nn.Module):
    """
    ERNIE 5.0 双路径视觉编码器 (Dual-Path Visual Encoder)

    结合了:
    1. CNN路径: 提取细粒度感知特征
    2. ViT路径: 提取高层语义特征
    3. Attention Fusion: 融合两者特征
    """

    def __init__(self, config: VisualConfig, out_dim: int):
        super().__init__()
        self.config = config

        # --- CNN Branch ---
        # 简单实现：使用ResNet-like结构或直接堆叠Conv
        cnn_channels = config.cnn_channels
        layers = []
        in_c = 3
        for out_c in cnn_channels:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=2),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(),
                )
            )
            in_c = out_c
        self.cnn_encoder = nn.Sequential(*layers)
        self.cnn_out_dim = cnn_channels[-1]

        # --- ViT Branch ---
        # 简单实现：一个微型ViT
        self.vit_patch_embed = PatchEmbed(
            img_size=config.image_size,
            patch_size=config.patch_size,
            embed_dim=config.vit_hidden_dim,
        )
        # ViT Blocks (简化: 4层)
        self.vit_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.vit_hidden_dim,
                    nhead=8,
                    dim_feedforward=config.vit_hidden_dim * 4,
                    batch_first=True,
                )
                for _ in range(4)
            ]
        )

        # --- Fusion ---
        # 将CNN特征对齐到ViT维度
        self.cnn_proj = nn.Linear(self.cnn_out_dim, config.vit_hidden_dim)

        # 融合注意力: Self-Attention over combined tokens
        self.fusion_attn = nn.MultiheadAttention(
            embed_dim=config.vit_hidden_dim, num_heads=8, batch_first=True
        )

        # --- Output Projection ---
        self.output_proj = nn.Linear(config.vit_hidden_dim, out_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [Batch, 3, H, W]
        Returns:
            visual_features: [Batch, Seq_Len, Out_Dim]
        """
        batch_size = images.size(0)

        # 1. ViT Path
        vit_tokens = self.vit_patch_embed(images)  # [B, N_patches, D_vit]
        # 添加简单位置编码 (Fixed absolute PE for simplicity)
        # 实际应使用 Learnable PE or RoPE

        for block in self.vit_blocks:
            vit_tokens = block(vit_tokens)

        # 2. CNN Path
        cnn_feat = self.cnn_encoder(images)  # [B, C_last, H_small, W_small]
        # Flatten CNN features
        cnn_tokens = cnn_feat.flatten(2).transpose(1, 2)  # [B, N_cnn, C_last]
        cnn_tokens = self.cnn_proj(cnn_tokens)  # [B, N_cnn, D_vit]

        # 3. Fusion
        # Concatenate tokens: [ViT; CNN]
        combined_tokens = torch.cat(
            [vit_tokens, cnn_tokens], dim=1
        )  # [B, N_total, D_vit]

        # Fusion Attention
        # 这里的Fusion在论文中是一个"Patch Merger with Attention"
        # 简单起见，我们对拼接后的序列做一次Self-Attention
        fused_tokens, _ = self.fusion_attn(
            combined_tokens, combined_tokens, combined_tokens
        )

        # 4. Pooling or Selection
        # ERNIE 5.0 可能保留序列结构以保持空间信息，或者做Pool
        # 假设我们需要将其作为 prefix 输入到 LLM，我们通常保留序列
        output = self.output_proj(fused_tokens)

        return output
