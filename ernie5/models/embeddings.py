import torch
import torch.nn as nn
import math
from typing import Tuple, Optional
from ernie5.configs.model_config import ERNIE5Config

class UniRoPE(nn.Module):
    """
    Unified Spatiotemporal RoPE (Uni-RoPE)
    
    能够同时处理1D(文本)、2D(图像)和3D(视频)的位置编码。
    实现思路：
    将Head Dimension分组，分别用于编码不同的空间/时间维度。
    例如：对于视频(t, h, w)，我们可以将dim分为三部分分别编码t, h, w。
    
    Args:
        config: ERNIE 5.0配置对象
    """
    
    def __init__(self, config: ERNIE5Config):
        super().__init__()
        self.config = config
        self.dim = config.hidden_size // config.attention.num_attention_heads
        self.base = config.attention.rope_base
        self.max_position_embeddings = config.max_position_embeddings
        
        # 缓存这里的频率以避免重复计算
        self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq_buffer", self.inv_freq, persistent=False)
        
    def forward(self, position_ids: torch.Tensor, seq_len: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成Rotary Embeddings (cos, sin)
        
        Args:
            position_ids: [batch_size, seq_len, 3] 
                最后一维包含 (time_pos, height_pos, width_pos)
                对于文本，time_pos为token index, h=0, w=0
                对于图像，time_pos=0, h=row, w=col
                对于视频，time_pos=frame, h=row, w=col
            seq_len: 序列长度 (可选)
            
        Returns:
            cos, sin: [batch_size, seq_len, head_dim] (或者广播兼容的形状)
        """
        # 如果没有传入position_ids，假设是1D文本
        if position_ids is None:
            raise ValueError("position_ids must be provided for Uni-RoPE")
            
        device = position_ids.device
        dtype = self.inv_freq_buffer.dtype
        
        # position_ids shape: [bs, seq, 3] -> (t, h, w)
        t_pos = position_ids[..., 0]
        h_pos = position_ids[..., 1]
        w_pos = position_ids[..., 2]
        
        # 策略：将特征维度三等分（简化的实现策略）
        # 实际实现可能更复杂，比如交织，或者根据模态类型动态调整
        # 这里采用固定分配：D/3给t，D/3给h，D/3给w
        # 注意：dim必须能被6整除以保证 splits 是偶数（cos/sin成对）
        
        # 为了通用性，我们这里采用一种加权叠加或者分段策略。
        # 常见Video RoPE做法：分别计算emb，然后concat
        
        alloc = self.config.attention.rope_axis_dim_allocation
        alloc_sum = sum(alloc)
        dim_t = int(self.dim * alloc[0] / alloc_sum)
        dim_h = int(self.dim * alloc[1] / alloc_sum)
        dim_w = self.dim - dim_t - dim_h
        
        # 保证偶数
        dim_t = dim_t - (dim_t % 2)
        dim_h = dim_h - (dim_h % 2)
        dim_w = self.dim - dim_t - dim_h
        
        inv_freq_t = self.inv_freq_buffer[:dim_t//2]
        inv_freq_h = self.inv_freq_buffer[dim_t//2 : (dim_t//2 + dim_h//2)]
        inv_freq_w = self.inv_freq_buffer[(dim_t//2 + dim_h//2):]
        
        # 计算freqs
        # einsum: [bs, seq], [dim/2] -> [bs, seq, dim/2]
        freqs_t = torch.einsum("bs,d->bsd", t_pos.to(dtype), inv_freq_t)
        freqs_h = torch.einsum("bs,d->bsd", h_pos.to(dtype), inv_freq_h)
        freqs_w = torch.einsum("bs,d->bsd", w_pos.to(dtype), inv_freq_w)
        
        # 拼接回完整维度 [bs, seq, dim/2]
        freqs = torch.cat((freqs_t, freqs_h, freqs_w), dim=-1)
        
        # 再次拼接生成 [bs, seq, dim] -> cos, sin
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos()
        sin = emb.sin()
        
        return cos, sin

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """
    应用RoPE旋转到Q和K
    """
    # q, k: [bs, num_heads, seq_len, head_dim]
    # cos, sin: [bs, seq_len, head_dim] -> 需要调整形状以广播
    
    # 调整 cos, sin 到 [bs, 1, seq_len, head_dim]
    if cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class MultiModalEmbedding(nn.Module):
    """
    统一多模态Embedding层
    所有模态共享同一个Token Embedding空间（或者通过投影对齐到该空间）
    """
    
    def __init__(self, config: ERNIE5Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        # 简单的统一Embedding：所有模态共用一个词表
        # 在实际复杂实现中，可能会有分别的 Embedding + Projector，然后 sum
        # 但ERNIE 5.0提到"Unified Token Space"，这里我们实现一个巨大的共享Embedding
        self.token_embedding = nn.Embedding(
            config.get_unified_vocab_size(),
            config.hidden_size,
            padding_idx=config.pad_token_id
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
        """
        embeddings = self.token_embedding(input_ids)
        embeddings = self.dropout(embeddings)
        return embeddings
