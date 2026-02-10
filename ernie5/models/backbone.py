import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union
from ernie5.configs.model_config import ERNIE5Config
from ernie5.models.embeddings import MultiModalEmbedding, UniRoPE
from ernie5.models.transformer import TransformerBlock, RMSNorm
from ernie5.models.multimodal_projector import MultiModalProjector
from ernie5.training.elastic import ElasticTrainingManager, ElasticContext
from ernie5.models.visual_generator import VisualGenerator
from ernie5.models.audio_generator import NextCodecPredictionHead

class ERNIE5Model(nn.Module):
    """
    ERNIE 5.0 主干网络 (Backbone)
    
    由多层TransformerBlock堆叠而成，支持:
    - 统一多模态Embedding
    - Uni-RoPE 位置编码
    - 弹性训练 (Elastic Training)
    """
    
    def __init__(self, config: ERNIE5Config):
        super().__init__()
        self.config = config
        
        # 1. Embeddings
        self.embeddings = MultiModalEmbedding(config)

        # Optional projector for raw multimodal features
        self.projector = MultiModalProjector(config)
        
        # 2. Positional Encoding
        self.uni_rope = UniRoPE(config)
        
        # 3. Layers
        self.layers = nn.ModuleList([
            TransformerBlock(config, i)
            for i in range(config.num_hidden_layers)
        ])
        
        # 4. Final Norm
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # 5. Elastic Manager
        self.elastic_manager = ElasticTrainingManager(config)
        
        self.gradient_checkpointing = False
        
    def forward(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        media_positions: Optional[List[List[int]]] = None,
        media_embeddings: Optional[List[torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        output_router_logits: bool = False,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, ...]]:
        
        # 1. Embedding
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embeddings(input_ids)

        if media_positions is not None and media_embeddings is not None:
            # Replace token embeddings at specified positions with projected media embeddings
            for b, positions in enumerate(media_positions):
                if len(positions) == 0:
                    continue
                embeds = media_embeddings[b]
                length = min(len(positions), embeds.size(0))
                hidden_states[b, positions[:length], :] = embeds[:length]
        batch_size, seq_len, _ = hidden_states.shape
        
        # 2. Uni-RoPE
        # 如果没有提供position_ids，创建一个默认的 (假设1D)
        if position_ids is None:
            # [Batch, Seq, 3] -> (t, 0, 0)
            position_ids = torch.zeros(
                (batch_size, seq_len, 3), 
                dtype=torch.long, 
                device=hidden_states.device
            )
            position_ids[..., 0] = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
            
        cos, sin = self.uni_rope(position_ids, seq_len=seq_len)
        rotary_pos_emb = (cos, sin)
        
        # 3. Elastic Context Sampling
        elastic_ctx = ElasticContext(
            active_layers=list(range(len(self.layers))),
            active_experts=None,
            current_top_k=self.config.moe.top_k
        )
        
        if self.training and self.config.elastic.enabled:
            elastic_ctx = self.elastic_manager.sample_context()
            
        # 4. Layers Loop
        all_router_logits = []
        next_decoder_cache = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            # Depth Elasticity: Skip layer if not in active_layers
            if i not in elastic_ctx.active_layers:
                continue
                
            past_layer_kv = past_key_values[i] if past_key_values is not None else None
            
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                    
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    rotary_pos_emb,
                    past_layer_kv,
                    use_cache,
                    elastic_ctx.current_top_k,
                    elastic_ctx.active_experts,
                )
            else:
                layer_outputs = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    rotary_pos_emb=rotary_pos_emb,
                    past_key_value=past_layer_kv,
                    use_cache=use_cache,
                    top_k=elastic_ctx.current_top_k,
                    active_experts=elastic_ctx.active_experts
                )
                
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache.append(layer_outputs[1])
                
            if output_router_logits and layer_outputs[2] is not None:
                all_router_logits.append(layer_outputs[2])
                
        hidden_states = self.norm(hidden_states)
        
        if output_router_logits:
            return hidden_states, next_decoder_cache, all_router_logits
            
        return hidden_states, next_decoder_cache

class ERNIE5ForCausalLM(nn.Module):
    """
    ERNIE 5.0 用于生成任务 (Next-Token Prediction)
    """
    def __init__(self, config: ERNIE5Config):
        super().__init__()
        self.config = config
        self.model = ERNIE5Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.get_unified_vocab_size(), bias=False)
        self.visual_head = VisualGenerator(config)
        self.audio_head = NextCodecPredictionHead(config)
        if config.use_mtp:
            self.mtp_heads = nn.ModuleList([
                nn.Linear(config.hidden_size, config.get_unified_vocab_size(), bias=False)
                for _ in range(config.mtp_group_size)
            ])
        else:
            self.mtp_heads = None
        
        # Share weights
        self.lm_head.weight = self.model.embeddings.token_embedding.weight
        if self.mtp_heads is not None:
            for head in self.mtp_heads:
                head.weight = self.model.embeddings.token_embedding.weight
        
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        visual_mask: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        visual_labels: Optional[torch.Tensor] = None,
        audio_labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        outputs = self.model(input_ids, **kwargs)
        hidden_states = outputs[0]
        
        logits = self.lm_head(hidden_states)
        
        loss = None
        text_loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            if text_mask is not None:
                shift_mask = text_mask[..., 1:].contiguous().view(-1)
                if shift_mask.any():
                    loss_fct = nn.CrossEntropyLoss(reduction="none")
                    per_token = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    text_loss = per_token[shift_mask].mean()
                else:
                    text_loss = torch.tensor(0.0, device=hidden_states.device)
            else:
                loss_fct = nn.CrossEntropyLoss()
                text_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Visual loss (NFSP)
        visual_loss = None
        if visual_labels is not None:
            visual_logits, visual_loss = self.visual_head(
                hidden_states,
                labels=visual_labels,
                loss_mask=visual_mask
            )

        # Audio loss (NCP)
        audio_loss = None
        if audio_labels is not None:
            _, audio_loss = self.audio_head(
                hidden_states,
                labels=audio_labels,
                loss_mask=audio_mask
            )

        # MTP loss
        mtp_loss = None
        if self.mtp_heads is not None and labels is not None:
            mtp_losses = []
            for k, head in enumerate(self.mtp_heads, start=1):
                mtp_logits = head(hidden_states[..., :-k, :])
                mtp_labels = labels[..., k:]
                if text_mask is not None:
                    mtp_mask = text_mask[..., k:].contiguous().view(-1)
                    if mtp_mask.any():
                        loss_fct = nn.CrossEntropyLoss(reduction="none")
                        per_token = loss_fct(mtp_logits.view(-1, mtp_logits.size(-1)), mtp_labels.view(-1))
                        mtp_losses.append(per_token[mtp_mask].mean())
                else:
                    loss_fct = nn.CrossEntropyLoss()
                    mtp_losses.append(loss_fct(mtp_logits.view(-1, mtp_logits.size(-1)), mtp_labels.view(-1)))
            if len(mtp_losses) > 0:
                mtp_loss = sum(mtp_losses) / len(mtp_losses)

        loss = None
        if text_loss is not None or visual_loss is not None or audio_loss is not None or mtp_loss is not None:
            # NOTE:
            # 不要使用 `(tensor or 0.0)` 聚合损失，这会触发
            # `RuntimeError: Boolean value of Tensor with more than one value is ambiguous`。
            # 这里显式地按 None 判断并逐项累加，保证训练稳定。
            loss_terms = [
                term for term in (text_loss, visual_loss, audio_loss)
                if term is not None
            ]
            loss = sum(loss_terms) if loss_terms else torch.tensor(0.0, device=hidden_states.device)
            if mtp_loss is not None:
                loss = loss + self.config.mtp_loss_weight * mtp_loss

        return (loss, logits) if loss is not None else logits
