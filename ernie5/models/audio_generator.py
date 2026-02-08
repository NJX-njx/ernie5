import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from ernie5.configs.model_config import ERNIE5Config

class NextCodecPredictionHead(nn.Module):
    """
    音频生成头 (NCP Head)
    
    实现 Next-Codec Prediction:
    - 粗到细分层预测 (Coarse-to-Fine)
    - 多音频头 (Multi-Head for different RVQ levels)
    - 深度方向自回归 (Depthwise AR)
    """
    
    def __init__(self, config: ERNIE5Config):
        super().__init__()
        self.config = config
        self.audio_config = config.audio
        
        self.num_layers = self.audio_config.rvq_layers
        self.codebook_size = self.audio_config.codebook_size
        
        # 每一个RVQ层级都有一个预测头
        self.heads = nn.ModuleList([
            nn.Linear(config.hidden_size, self.codebook_size)
            for _ in range(self.num_layers)
        ])
        
        # 反馈嵌入 (Feedback Embeddings) - 将预测出的较粗层级Embedding加回到hidden state
        self.feedback_embeddings = nn.ModuleList([
            nn.Embedding(self.codebook_size, config.hidden_size)
            for _ in range(self.num_layers)
        ])
        
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: [Batch, Seq_Len, Hidden_Size] (from Backbone)
            labels: [Batch, Seq_Len, Num_Layers] (RVQ Codes)
            
        Returns:
            all_logits: List of [Batch, Seq_Len, Vocab]
            loss: scalar
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # 深度方向自回归循环
        # h_0 = h_backbone
        # For l = 0 to L-1:
        #   p_l = Head_l(h_l)
        #   code_l = sample(p_l) (or use ground truth in training)
        #   h_{l+1} = h_l + Embed(code_l)
        
        current_hidden = hidden_states
        all_logits = []
        total_loss = 0.0
        
        for i in range(self.num_layers):
            # 1. Predict current level
            logits = self.heads[i](current_hidden) # [B, S, Vocab]
            all_logits.append(logits)
            
            # 2. Calculate Loss (if labels provided)
            # 训练时强制使用 Teacher Forcing
            if labels is not None:
                # labels: [B, S, L] -> take layer i
                target = labels[:, :, i] # [B, S]
                
                # Shift? 
                # 这里要注意：NCP通常是预测"当前时间步"的所有深度层级
                # 还是预测"下一个时间步"?
                # 论文中：深度方向自回归。
                # 通常：Backbone预测 t 的 Level 0。Level 0 embedding 帮助预测 Level 1 ...
                # 所以我们还是需要在时间维度上 Shift Backbone input。
                # 但在这里的深度循环中，是对齐的。
                
                if loss_mask is not None:
                    valid = loss_mask.view(-1)
                    if valid.any():
                        loss_fct = nn.CrossEntropyLoss(reduction="none")
                        per_token = loss_fct(logits.view(-1, self.codebook_size), target.view(-1))
                        layer_loss = per_token[valid].mean()
                    else:
                        layer_loss = torch.tensor(0.0, device=hidden_states.device)
                else:
                    loss_fct = nn.CrossEntropyLoss()
                    layer_loss = loss_fct(logits.view(-1, self.codebook_size), target.view(-1))
                total_loss += layer_loss
                
                # Teacher Forcing for next level
                code_input = target
            else:
                # Inference: sample or argmax
                code_input = torch.argmax(logits, dim=-1)
                
            # 3. Update hidden state for next level
            # h_{l+1} = h_l + Embed(code)
            if i < self.num_layers - 1:
                emb = self.feedback_embeddings[i](code_input)
                current_hidden = current_hidden + emb
                
        # Average loss over layers
        if labels is not None:
            total_loss = total_loss / self.num_layers
            return all_logits, total_loss
            
        return all_logits, None
