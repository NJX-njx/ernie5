import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict
from tqdm import tqdm
import os

from ernie5.models.backbone import ERNIE5ForCausalLM
from ernie5.configs.training_config import TrainingConfig
from ernie5.training.scheduler import WSDScheduler, CosineScheduler
from ernie5.training.losses import NextGroupTokenLoss

class ERNIE5Trainer:
    """
    ERNIE 5.0 Trainer
    """
    def __init__(
        self,
        model: ERNIE5ForCausalLM,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay
        )
        
        # Scheduler: Stage1 WSD, Stage2 Cosine
        self.stage1_scheduler = WSDScheduler(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_stable_steps=max(0, config.stage1_steps - config.warmup_steps - 2000),
            num_decay_steps=2000
        )
        self.stage2_scheduler = CosineScheduler(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=max(1, config.stage2_steps)
        )
        
        # Loss Function
        self.criterion = NextGroupTokenLoss()
        
    def train(self):
        self.model.train()
        global_step = 0
        total_loss = 0.0
        
        # Progress Bar
        progress_bar = tqdm(total=self.config.max_steps, desc="Training")
        
        train_iterator = iter(self.train_dataloader)
        
        while global_step < self.config.max_steps:
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(self.train_dataloader)
                batch = next(train_iterator)
                
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            def maybe_to_device(key: str):
                value = batch.get(key)
                return value.to(self.device) if value is not None else None

            position_ids = maybe_to_device("position_ids")
            attention_mask = maybe_to_device("attention_mask")
            text_mask = maybe_to_device("text_mask")
            visual_mask = maybe_to_device("visual_mask")
            audio_mask = maybe_to_device("audio_mask")
            visual_labels = maybe_to_device("visual_labels")
            audio_labels = maybe_to_device("audio_labels")
            
            # Forward
            # 对于简化版，Backbone forward returns (loss, logits) if labels provided
            # 但我们需要多模态 Loss，所以 Backbone 的 forward 需要更灵活
            # 这里简化：假设 ERNIE5ForCausalLM 主要计算 Text Loss
            # Visual/Audio Heads 需要单独调用
            
            # 我们直接调用 model(input_ids) 得到 hidden_states
            # 然后手动调各 Head? 或者集成在 model forward 里?
            # 简单起见，假设 ERNIE5ForCausalLM.forward 已经集成了所有Head计算并返回各自 Loss (如果实现很完整)
            # 但目前 backbone.py 里只返回了 Text Loss。
            
            # 正确做法：修改 ERNIE5ForCausalLM 以包含 VisualGenerator 等
            
            loss, logits = self.model(
                input_ids,
                labels=labels,
                position_ids=position_ids,
                attention_mask=attention_mask,
                text_mask=text_mask,
                visual_mask=visual_mask,
                audio_mask=audio_mask,
                visual_labels=visual_labels,
                audio_labels=audio_labels
            )
            
            # Backward
            loss.backward()
            
            # Clip Grad
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            # Update
            self.optimizer.step()
            # Stage switch
            if global_step < self.config.stage1_steps:
                self.stage1_scheduler.step()
            else:
                self.stage2_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            
            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})
            
            if global_step % self.config.save_steps == 0:
                self.save_checkpoint(global_step)
                
    def save_checkpoint(self, step):
        save_path = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
