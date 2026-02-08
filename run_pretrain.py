import argparse
import logging
import torch
from torch.utils.data import DataLoader

from ernie5.configs import ERNIE5Config, TrainingConfig, TokenizerConfig, ModelScale
from ernie5.models import ERNIE5Model, ERNIE5ForCausalLM
from ernie5.tokenizers import TextTokenizer, VisualTokenizer, AudioTokenizer
from ernie5.data import MultiModalIterableDataset, MultiModalCollator
from ernie5.training import ERNIE5Trainer

def main():
    parser = argparse.ArgumentParser(description="Run ERNIE 5.0 Pre-training")
    parser.add_argument("--scale", type=str, default="mini", choices=["mini", "small", "medium", "full"], help="Model scale")
    args = parser.parse_args()
    
    # 1. 配置
    scale_map = {
        "mini": ModelScale.MINI,
        "small": ModelScale.SMALL,
        "medium": ModelScale.MEDIUM,
        "full": ModelScale.FULL
    }
    model_config = ERNIE5Config.from_scale(scale_map[args.scale])
    training_config = TrainingConfig(output_dir=f"./output_ernie5_{args.scale}")
    tokenizer_config = TokenizerConfig(vocab_file=None) # Train from scratch or load
    
    print(f"Initializing ERNIE 5.0 [{args.scale}]...")
    print(f"Num Parameters: {model_config.get_num_parameters() / 1e6:.2f}M")
    print(f"Activated Params: {model_config.get_activated_parameters() / 1e6:.2f}M")
    
    # 2. 模型与Tokenizer
    # 实际应用中需先训练 Tokenizer
    tokenizer = TextTokenizer(tokenizer_config)
    visual_tokenizer = VisualTokenizer(model_config.visual)
    audio_tokenizer = AudioTokenizer(model_config.audio)
    
    model = ERNIE5ForCausalLM(model_config)
    
    # 3. 数据准备
    # 使用 Mock 无限流数据
    dataset = MultiModalIterableDataset(file_paths=[], modality="text") 
    collator = MultiModalCollator(model_config, tokenizer, visual_tokenizer, audio_tokenizer)
    
    dataloader = DataLoader(
        dataset,
        batch_size=training_config.per_device_train_batch_size,
        collate_fn=collator
    )
    
    # 4. 训练器
    trainer = ERNIE5Trainer(
        model=model,
        config=training_config,
        train_dataloader=dataloader
    )
    
    print("Starting Training...")
    # trainer.train() # Uncomment to run
    print("Training simulation ready. Run `trainer.train()` to start.")

if __name__ == "__main__":
    main()
