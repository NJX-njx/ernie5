from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TrainingConfig:
    """
    训练配置
    """
    output_dir: str = "./output"
    overwrite_output_dir: bool = False
    
    do_train: bool = True
    do_eval: bool = True
    do_predict: bool = False
    
    # 批次大小
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    
    # 学习率调度
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    
    # Pre-training schedules
    max_steps: int = 100000
    warmup_steps: int = 2000
    
    # ERNIE 5.0 Stages
    stage1_steps: int = 20000  # 8K context
    stage2_steps: int = 50000  # 32K/128K context
    
    # 精度
    fp16: bool = False
    bf16: bool = True
    
    # 分布式
    local_rank: int = -1
    ddp_backend: str = "nccl"
    
    # 梯度累积
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # 日志
    logging_steps: int = 100
    save_steps: int = 1000
    
    # 随机种子
    seed: int = 42
