import random
from typing import Dict, Iterator, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


class MultiModalDataset(Dataset):
    """
    通用多模态对齐数据集 (Map-style)
    用于 SFT 或具有明确索引的数据
    """

    def __init__(self, data_source: List[Dict]):
        """
        data_source example:
        [
            {
                "type": "image_text",
                "text": "A cat sitting on a mat",
                "image_path": "/path/to/cat.jpg"
            },
            ...
        ]
        """
        self.data_source = data_source

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        # 实际实现需加载图像/音频文件
        # 这里仅返回元数据
        return self.data_source[idx]


class MultiModalIterableDataset(IterableDataset):
    """
    流式多模态数据集 (Iterable-style)
    用于大规模预训练，支持无限流
    """

    def __init__(self, file_paths: List[str], modality: str = "text"):
        self.file_paths = file_paths
        self.modality = modality

    def __iter__(self):
        # 模拟数据生成
        while True:
            # 随机生成一些 Mock 数据
            if self.modality == "text":
                yield {
                    "type": "text",
                    "content": "This is a sample text document "
                    * random.randint(1, 10),
                }
            elif self.modality == "image_text":
                yield {
                    "type": "image_text",
                    "text": "An image description",
                    "image": torch.randn(3, 256, 256),  # Mock image
                }
            elif self.modality == "interleaved":
                # 交织数据
                yield {
                    "type": "interleaved",
                    "content": [
                        {"type": "text", "value": "Look at this:"},
                        {"type": "image", "value": torch.randn(3, 256, 256)},
                        {"type": "text", "value": "It is amazing!"},
                    ],
                }


class MultiSourceDataset(Dataset):
    """
    将多个Dataset拼接，并支持通过全局索引访问。
    """

    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets
        self.offsets = []
        total = 0
        for ds in datasets:
            self.offsets.append(total)
            total += len(ds)
        self.total_size = total

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        for i in range(len(self.datasets) - 1, -1, -1):
            if idx >= self.offsets[i]:
                return self.datasets[i][idx - self.offsets[i]]
        raise IndexError("Index out of range")
