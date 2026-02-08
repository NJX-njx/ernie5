import torch
from torch.optim.lr_scheduler import LambdaLR
import math

class WSDScheduler(LambdaLR):
    """
    Warmup-Stable-Decay (WSD) Scheduler
    ERNIE 5.0 Stage 1 使用的学习率调度策略
    
    Curve:
    1. Warmup: 0 -> lr_max (steps: warmup_steps)
    2. Stable: lr_max (steps: stable_steps)
    3. Decay: lr_max -> 0 (steps: decay_steps, style: cosine or linear)
    """
    def __init__(
        self, 
        optimizer, 
        num_warmup_steps: int, 
        num_stable_steps: int, 
        num_decay_steps: int, 
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_stable_steps = num_stable_steps
        self.num_decay_steps = num_decay_steps
        self.min_lr_ratio = min_lr_ratio
        
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            
            if current_step < num_warmup_steps + num_stable_steps:
                return 1.0
                
            # Decay phase
            decay_step = current_step - num_warmup_steps - num_stable_steps
            if decay_step >= num_decay_steps:
                return min_lr_ratio
            
            # 简化版: 线性衰减
            # progress = float(decay_step) / float(max(1, num_decay_steps))
            # return max(min_lr_ratio, 1.0 - (1.0 - min_lr_ratio) * progress)
            
            # ERNIE 5.0 mentions Cosine Annealing in Stage 2, but WSD in Stage 1 usually implies specific decay
            # Let's use Cosine for decay phase
            progress = float(decay_step) / float(max(1, num_decay_steps))
            return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
            
        super().__init__(optimizer, lr_lambda, last_epoch)

class CosineScheduler(LambdaLR):
    """标准的Cosine Annealing with Warmup"""
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1, last_epoch=-1):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
        super().__init__(optimizer, lr_lambda, last_epoch)
