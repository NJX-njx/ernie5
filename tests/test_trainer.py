import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader

from ernie5.training.trainer import ERNIE5Trainer
from ernie5.configs.training_config import TrainingConfig


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)

    def forward(self, input_ids, labels=None, **kwargs):
        x = self.proj(input_ids.float())
        loss = x.mean()
        return loss, x


class _DummyDataset(torch.utils.data.IterableDataset):
    def __iter__(self):
        while True:
            yield {
                "input_ids": torch.ones(2, 4),
                "labels": torch.ones(2, 4),
                "position_ids": None,
                "attention_mask": None,
                "text_mask": None,
                "visual_mask": None,
                "audio_mask": None,
                "visual_labels": None,
                "audio_labels": None,
            }


def test_trainer_runs_with_optional_none_fields(tmp_path):
    model = _DummyModel()
    cfg = TrainingConfig(
        output_dir=str(tmp_path),
        max_steps=1,
        save_steps=100,
        stage1_steps=1,
        stage2_steps=1,
        warmup_steps=0,
        per_device_train_batch_size=1,
    )
    dataloader = DataLoader(_DummyDataset(), batch_size=None)
    trainer = ERNIE5Trainer(model=model, config=cfg, train_dataloader=dataloader)
    trainer.train()
