import torch
import random
from dataclasses import dataclass
from typing import List, Optional, Dict
from ernie5.configs.model_config import ERNIE5Config

@dataclass
class ElasticContext:
    """
    当前训练步的弹性配置上下文
    """
    active_layers: List[int]  # 激活的层索引
    active_experts: Optional[List[int]] = None # 激活的专家索引 (Width elastic)
    current_top_k: Optional[int] = None # 当前Top-K (Sparsity elastic)

class ElasticTrainingManager:
    """
    弹性训练管理器
    
    负责在训练过程中动态采样子网络配置：
    - 深度弹性 (Depth)
    - 宽度弹性 (Width)
    - 稀疏度弹性 (Sparsity)
    """
    
    def __init__(self, config: ERNIE5Config):
        self.config = config
        self.elastic_config = config.elastic
        self.num_layers = config.num_hidden_layers
        self.num_experts = config.moe.num_experts
        
    def sample_context(self) -> ElasticContext:
        """
        采样当前步的配置
        """
        if not self.elastic_config.enabled:
            return ElasticContext(
                active_layers=list(range(self.num_layers)),
                active_experts=None, # All
                current_top_k=self.config.moe.top_k
            )
            
        # 1. 深度弹性
        # 总是保留第一层和最后一层，中间随机采样
        active_layers = [0]
        middle_layers = list(range(1, self.num_layers - 1))
        
        if random.random() < self.elastic_config.depth_elastic_prob:
            # 随机丢弃部分层
            # 简单策略：按概率保留，或者采样固定比例
            min_layers = int(self.num_layers * self.elastic_config.min_depth_ratio)
            num_keep = random.randint(min_layers, len(middle_layers))
            kept_middle = sorted(random.sample(middle_layers, num_keep))
            active_layers.extend(kept_middle)
        else:
            active_layers.extend(middle_layers)
            
        active_layers.append(self.num_layers - 1)
        
        # 2. 宽度弹性 (专家子集)
        active_experts = None
        if random.random() < self.elastic_config.width_elastic_prob:
            min_experts = int(self.num_experts * self.elastic_config.min_width_ratio)
            # 保证至少有top_k个专家，否则无法路由
            min_experts = max(min_experts, self.config.moe.top_k)
            
            num_keep = random.randint(min_experts, self.num_experts)
            active_experts = sorted(random.sample(range(self.num_experts), num_keep))
            
        # 3. 稀疏度弹性 (Top-K)
        current_top_k = self.config.moe.top_k
        if random.random() < self.elastic_config.sparsity_elastic_prob:
            # 随机减少top_k
            min_k = self.config.moe.min_top_k
            if min_k < current_top_k:
                current_top_k = random.randint(min_k, current_top_k)
                
        return ElasticContext(
            active_layers=active_layers,
            active_experts=active_experts,
            current_top_k=current_top_k
        )
