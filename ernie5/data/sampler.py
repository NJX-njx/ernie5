import torch
from torch.utils.data import Sampler
from typing import List, Iterator, Optional
import numpy as np

class MultiModalSampler(Sampler):
    """
    多模态混合采样器
    
    支持按比例从不同数据源采样 (Text, Image-Text, Video-Text, etc.)
    """
    def __init__(
        self, 
        dataset_sizes: List[int], 
        weights: List[float], 
        batch_size: int,
        drop_last: bool = False
    ):
        self.dataset_sizes = dataset_sizes
        self.weights = torch.tensor(weights, dtype=torch.float)
        self.weights = self.weights / self.weights.sum() # Normalize
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        self.total_size = sum(dataset_sizes)
        
    def __iter__(self):
        # 简单策略：每一批次根据权重选择一个 Source，然后从该 Source 采样 Batch
        # 或者：更加细粒度，混合 Batch
        
        # 模拟无限流采样逻辑，或者基于 Epoch 的逻辑
        # 这里实现基于 Epoch 的逻辑：生成所有索引
        
        # 实际大规模训练通常是 Infinite Loop + Step limit
        # 返回全局索引，适配 MultiSourceDataset / ConcatDataset
        offsets = [0]
        for size in self.dataset_sizes[:-1]:
            offsets.append(offsets[-1] + size)

        num_batches = self.total_size // self.batch_size
        if not self.drop_last and self.total_size % self.batch_size != 0:
            num_batches += 1

        for _ in range(num_batches):
            # 选择一个数据源
            ds_idx = int(torch.multinomial(self.weights, 1).item())
            size = self.dataset_sizes[ds_idx]
            start = offsets[ds_idx]

            for _ in range(self.batch_size):
                sample_idx = torch.randint(0, size, (1,)).item()
                yield start + sample_idx

class WeightedMixingSampler:
    """
    流式混合采样器逻辑 (用于 IterableDataset)
    """
    def __init__(self, datasets: List, weights: List[float]):
        self.datasets = datasets
        self.weights = weights
        self.iterators = [iter(d) for d in datasets]
        
    def __iter__(self):
        while True:
            # 根据权重随机选择一个数据集
            choice = np.random.choice(len(self.datasets), p=self.weights)
            try:
                yield next(self.iterators[choice])
            except StopIteration:
                # 重新初始化该迭代器? 或者停止
                # 对于无限流，通常不会 Stop
                self.iterators[choice] = iter(self.datasets[choice])
                yield next(self.iterators[choice])
