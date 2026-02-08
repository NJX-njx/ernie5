import torch
import torch.nn as nn
from typing import Optional, Tuple
from ernie5.configs.model_config import ERNIE5Config
from ernie5.models.attention import MultiHeadAttention
from ernie5.models.moe import SparseMoELayer, SwiGLUExpert

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class TransformerBlock(nn.Module):
    """
    ERNIE 5.0 Transformer Layer
    包含:
    - RMSNorm
    - MultiHead Attention (with Uni-RoPE)
    - Sparse MoE or Dense MLP
    - Residual connections
    """
    
    def __init__(self, config: ERNIE5Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        
        self.hidden_size = config.hidden_size
        
        # Pre-normalization
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Attention
        self.self_attn = MultiHeadAttention(config)
        
        # MoE or Dense
        # 频率控制: e.g. freq=2 => layer 1(MoE), 2(Dense), 3(MoE)... or 0, 2
        # Start from 0-indexed. usually every N layers
        self.is_moe_layer = (layer_idx % config.moe_layer_frequency == 0) and (layer_idx > 0) 
        # Note: ERNIE 5.0 report might specify layer range. Assuming standard frequency.
        # Often first few layers are dense. Let's assume simpler: every Nth layer is MoE.
        
        if self.is_moe_layer:
            self.mlp = SparseMoELayer(config)
        else:
            self.mlp = SwiGLUExpert(config.hidden_size, config.intermediate_size)
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        top_k: Optional[int] = None,
        active_experts: Optional[list] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        
        # 1. Attention Block
        residual = hidden_states
        
        hidden_states = self.input_layernorm(hidden_states)
        
        attn_output, current_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        
        hidden_states = residual + attn_output
        
        # 2. MLP/MoE Block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        router_logits = None
        
        if self.is_moe_layer:
            hidden_states, router_logits = self.mlp(hidden_states, top_k=top_k, active_experts=active_experts)
        else:
            hidden_states = self.mlp(hidden_states)
            
        hidden_states = residual + hidden_states
        
        return hidden_states, current_key_value, router_logits
