import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from ernie5.configs.model_config import ERNIE5Config
from ernie5.models.embeddings import apply_rotary_pos_emb

class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制 (GQA + RoPE + FlashAttention)
    """
    
    def __init__(self, config: ERNIE5Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.attention.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        # 分组查询注意力 Grouped Query Attention (GQA)
        self.num_kv_heads = config.attention.num_kv_heads if config.attention.num_kv_heads else self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.dropout = config.attention.attention_dropout
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head
        # [BS, Seq, Heads, Dim] -> [BS, Heads, Seq, Dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        if rotary_pos_emb is not None:
            cos, sin = rotary_pos_emb
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            
        # KV Cache handling
        if past_key_value is not None:
            # past_key_value: (past_k, past_v)
            past_k, past_v = past_key_value
            key_states = torch.cat([past_k, key_states], dim=2)
            value_states = torch.cat([past_v, value_states], dim=2)
            
        if use_cache:
            current_key_value = (key_states, value_states)
        else:
            current_key_value = None
            
        # GQA: Repeat KV heads if needed to match Q heads
        if self.num_kv_groups > 1:
            # [BS, KV_Heads, Seq, Dim] -> [BS, KV_Heads, 1, Seq, Dim] -> [BS, KV_Heads, Groups, Seq, Dim] -> [BS, Heads, Seq, Dim]
            key_states = key_states[:, :, None, :, :].expand(batch_size, self.num_kv_heads, self.num_kv_groups, -1, self.head_dim).reshape(batch_size, self.num_heads, -1, self.head_dim)
            value_states = value_states[:, :, None, :, :].expand(batch_size, self.num_kv_heads, self.num_kv_groups, -1, self.head_dim).reshape(batch_size, self.num_heads, -1, self.head_dim)
            
        # FlashAttention (via SDPA)
        # dropout is handled inside SDPA
        # attention_mask: SDPA expects mask for positions to IGNORE/MASK OUT as True (or -inf in float mask)
        # But wait, pytorch documentation says:
        # "attn_mask: binary mask ... or float mask"
        # If is_causal=True, attn_mask should likely be None or compatible.
        
        # Determine causality implies strict lower triangular.
        # But in ERNIE 5.0, image/video parts might be bidirectional.
        # So we supply explicit `attention_mask`.
        
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False # We handle patterns via attn_mask
        )
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        
        output = self.o_proj(attn_output)
        
        return output, current_key_value
