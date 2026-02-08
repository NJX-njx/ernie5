import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from ernie5.configs.model_config import ERNIE5Config, MoEConfig

class SwiGLUExpert(nn.Module):
    """
    SwiGLU Expert FFN
    """
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class SparseMoELayer(nn.Module):
    """
    超稀疏混合专家层 (Ultra-Sparse MoE)
    支持:
    - 模态无关路由 (Modality-Agnostic Routing)
    - 共享专家 (Shared Experts)
    - Top-K 稀疏激活
    """
    
    def __init__(self, config: ERNIE5Config):
        super().__init__()
        self.config = config
        self.moe_config = config.moe
        self.hidden_size = config.hidden_size
        
        # 专家FFN维度
        self.expert_intermediate_size = int(config.intermediate_size * self.moe_config.expert_ffn_multiplier)
        
        # 1. 独占专家 (Routed Experts)
        self.num_experts = self.moe_config.num_experts
        self.experts = nn.ModuleList([
            SwiGLUExpert(self.hidden_size, self.expert_intermediate_size)
            for _ in range(self.num_experts)
        ])
        
        # 2. 共享专家 (Shared Experts) - 总是激活
        self.use_shared = self.moe_config.use_shared_expert
        if self.use_shared:
            self.num_shared = self.moe_config.num_shared_experts
            self.shared_experts = nn.ModuleList([
                SwiGLUExpert(self.hidden_size, self.expert_intermediate_size)
                for _ in range(self.num_shared)
            ])
        
        # 3. 路由器 (Router / Gate)
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.top_k = self.moe_config.top_k
        
        # 负载均衡Bias (如果使用无Loss负载均衡，这个Bias需要动态更新)
        # 这里仅定义为Parameter
        if self.moe_config.auxiliary_loss_free:
            self.register_buffer("router_bias", torch.zeros(self.num_experts))
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k: Optional[int] = None,
        active_experts: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
        Returns:
            final_hidden_states: [batch_size, seq_len, hidden_size]
            router_logits: 用计算辅助loss (如果需要)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        flat_hidden_states = hidden_states.view(-1, hidden_dim)
        
        final_hidden_states = 0.0
        
        # 1. 处理共享专家
        if self.use_shared:
            shared_output = 0.0
            for expert in self.shared_experts:
                shared_output = shared_output + expert(flat_hidden_states)
            final_hidden_states = final_hidden_states + shared_output  # 简单求和或平均? 通常是加法
            
        # 2. 计算路由
        router_logits = self.router(flat_hidden_states) # [num_tokens, num_experts]
        
        # 添加Bias (用于负载均衡)
        if hasattr(self, "router_bias"):
            router_logits = router_logits + self.router_bias
            
        router_probs = F.softmax(router_logits, dim=-1)

        # Apply expert subset mask if elastic width is enabled
        if active_experts is not None:
            mask = torch.full_like(router_probs, float("-inf"))
            mask[:, active_experts] = 0.0
            router_probs = F.softmax(router_logits + mask, dim=-1)
        
        # Top-K
        # weights: [num_tokens, k], indices: [num_tokens, k]
        k = top_k if top_k is not None else self.top_k
        weights, indices = torch.topk(router_probs, k, dim=-1)
        
        # Normalize weights
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        # 3. 专家计算 (Naive Loop implementation for simplicity)
        # 优化实现通常使用 permute + group conv 或 scatter/gather
        
        # 初始化 routed output
        routed_output = torch.zeros_like(flat_hidden_states)
        
        # 遍历每个专家 (这在Python循环中可能慢，但在小规模下可读性最好)
        # 实际高性能实现应使用 torch.compile 或专门的kernel
        for i, expert in enumerate(self.experts):
            if active_experts is not None and i not in active_experts:
                continue
            # 找出哪些token选中了该专家 (indices包含专家ID)
            # indices: [num_tokens, k]
            # mask: [num_tokens, k]
            batch_indices, k_indices = torch.where(indices == i)
            
            if len(batch_indices) == 0:
                continue
                
            # 获取这些token的输入
            # batch_indices 是平铺后的token索引
            selected_tokens = flat_hidden_states[batch_indices]
            
            # 计算专家输出
            expert_out = expert(selected_tokens)
            
            # 获取对应的路由权重
            scaling = weights[batch_indices, k_indices].unsqueeze(-1)
            
            # 加权并累加回输出
            # scatter_add_ 需要 index 匹配维度
            # 这里简单做: routed_output[batch_indices] += expert_out * scaling
            # 注意: 如果同一个token多次选中同一个专家(理论上topk不会)，这里会重复加。Topk返回唯一indices。
            
            # 由于可能存在并发写入问题（如果同一个位置被多个线程写），PyTorch的 index_add_ 是原子的
            routed_output.index_add_(0, batch_indices, expert_out * scaling)
            
        final_hidden_states = final_hidden_states + routed_output
        
        # Auxiliary-loss-free load balancing via router bias update
        if self.moe_config.auxiliary_loss_free and self.training:
            with torch.no_grad():
                # Usage per expert
                usage = torch.bincount(indices.view(-1), minlength=self.num_experts).float()
                usage = usage / max(1.0, usage.sum())
                target = torch.full_like(usage, 1.0 / self.num_experts)
                delta = target - usage
                self.router_bias.add_(self.moe_config.load_balance_update_speed * delta)

        return final_hidden_states.view(batch_size, seq_len, hidden_dim), router_logits
