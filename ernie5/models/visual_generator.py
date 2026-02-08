import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from ernie5.configs.model_config import ERNIE5Config

class NextFrameScalePredictionHead(nn.Module):
    """
    视觉生成头 (NFSP Head)
    
    支持:
    - 图像: Next-Scale Prediction (多尺度自回归)
    - 视频: Next-Frame Prediction + Next-Scale
    
    输入: Transformer Hidden States
    输出: Discrete Visual Tokens (Bit-wise codes)
    """
    
    def __init__(self, config: ERNIE5Config):
        super().__init__()
        self.config = config
        self.visual_config = config.visual
        
        # 词表大小 = 2^bits
        self.vocab_size = 2 ** self.visual_config.tokenizer_bits
        
        # 简单线性分类头
        # 实际可能包含简单的MLP或者Norm
        self.head = nn.Linear(config.hidden_size, self.vocab_size)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: [Batch, Seq_Len, Hidden_Size]
            labels: [Batch, Seq_Len] (Visual Token IDs)
        """
        logits = self.head(hidden_states) # [B, S, Vocab]
        
        loss = None
        if labels is not None:
            # 标准自回归损失: Predict Next Token
            # Shift
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            if loss_mask is not None:
                shift_mask = loss_mask[..., 1:].contiguous()
                valid = shift_mask.view(-1)
                if valid.any():
                    loss_fct = nn.CrossEntropyLoss(reduction="none")
                    per_token = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
                    loss = per_token[valid].mean()
                else:
                    loss = torch.tensor(0.0, device=hidden_states.device)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
            
        return logits, loss

class VisualGenerator(nn.Module):
    """
    视觉生成管理器
    处理多尺度/多帧生成的逻辑控制
    """
    def __init__(self, config: ERNIE5Config):
        super().__init__()
        self.config = config
        self.head = NextFrameScalePredictionHead(config)
        
    def create_generation_mask(self, batch_size, seq_len, scales: int):
        """
        创建尺度级因果掩码 (Scale-wise Causal Mask)
        规则:
        - Scale t can see Scale 0...t-1 (Causal)
        - Within Scale t, tokens can see each other (Bidirectional) - 论文中是"双向可见"
        """
        # 简化实现: 假设seq_len被scales整除，一段一段地处理
        # 实际还需要考虑Frame维度
        mask = torch.ones((seq_len, seq_len), dtype=torch.bool)
        
        # Apply standard causal mask first
        mask = torch.tril(mask)
        
        # Modify for bidirectional within scale? 
        # For simplify, we stick to standard causal for now, 
        # as implementing bidirectional within blocks requires complex mask manipulation relative to position_ids
        return mask
