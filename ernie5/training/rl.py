import random
from collections import deque
from typing import Any, Deque, Dict, List

import torch


class UnifiedMultiModalRL:
    """统一多模态强化学习框架（简化实现）。

    组件：
    1. U-RB (Unbiased Replay Buffer)
    2. MISC (Multi-Granularity Importance Sampling Clipping)
    3. WPSM (Well-Performed Sample Masking)
    """

    def __init__(self, capacity: int = 10000):
        # U-RB: 实际上需要复杂的分布式Buffer，这里模拟单机版本。
        self.replay_buffer: Deque[Dict[str, Any]] = deque(maxlen=capacity)

    def add_experience(self, experience: Dict[str, Any]) -> None:
        """添加一条轨迹样本。"""
        self.replay_buffer.append(experience)

    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """从回放池中采样一个 mini-batch。"""
        return random.sample(self.replay_buffer, min(len(self.replay_buffer), batch_size))

    def compute_policy_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        epsilon_clip: float = 0.2,
    ) -> torch.Tensor:
        """PPO-style 策略损失（含 clipping）。"""
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - epsilon_clip, 1.0 + epsilon_clip) * advantages
        return -torch.min(surr1, surr2).mean()

    def wpsm_masking(
        self,
        samples: List[Dict[str, Any]],
        threshold: float = 0.8,
        metric_key: str = "success_rate",
    ) -> List[Dict[str, Any]]:
        """WPSM: 过滤掉已掌握样本。

        约定：
        - 若样本中存在 `metric_key`，且其值 >= threshold，则视为“已掌握”并过滤。
        - 若缺失该字段，则保留（默认保守策略）。
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")

        filtered: List[Dict[str, Any]] = []
        for sample in samples:
            score = sample.get(metric_key)
            if score is None:
                filtered.append(sample)
                continue
            if score < threshold:
                filtered.append(sample)

        # 避免极端情况下全过滤导致训练中断，至少保留一个样本。
        if not filtered and samples:
            filtered.append(samples[0])
        return filtered
