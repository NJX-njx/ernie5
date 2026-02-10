import argparse
from typing import Optional

from ernie5.configs import ERNIE5Config, TrainingConfig, TokenizerConfig, ModelScale


def build_parser() -> argparse.ArgumentParser:
    """构建预训练脚本 CLI 参数。"""
    parser = argparse.ArgumentParser(description="Run ERNIE 5.0 Pre-training")
    parser.add_argument(
        "--scale",
        type=str,
        default="mini",
        choices=["mini", "small", "medium", "large", "full"],
        help="Model scale",
    )
    return parser


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """解析命令行参数，便于测试注入 argv。"""
    return build_parser().parse_args(argv)


def resolve_scale(scale: str) -> ModelScale:
    """将 CLI scale 字符串映射到 ModelScale 枚举。"""
    scale_map = {
        "mini": ModelScale.MINI,
        "small": ModelScale.SMALL,
        "medium": ModelScale.MEDIUM,
        "large": ModelScale.LARGE,
        "full": ModelScale.FULL,
    }
    try:
        return scale_map[scale]
    except KeyError:
        valid_scales = ", ".join(scale_map.keys())
        raise ValueError(f"Invalid scale '{scale}'. Expected one of: {valid_scales}.") from None


def main(argv: Optional[list[str]] = None) -> None:
    # 延迟导入重依赖，使 parser 测试不依赖 torch。
    from torch.utils.data import DataLoader

    from ernie5.models import ERNIE5ForCausalLM
    from ernie5.tokenizers import TextTokenizer, VisualTokenizer, AudioTokenizer
    from ernie5.data import MultiModalIterableDataset, MultiModalCollator
    from ernie5.training import ERNIE5Trainer

    args = parse_args(argv)

    # 1. 配置
    model_config = ERNIE5Config.from_scale(resolve_scale(args.scale))
    training_config = TrainingConfig(output_dir=f"./output_ernie5_{args.scale}")
    tokenizer_config = TokenizerConfig(vocab_file=None)  # Train from scratch or load

    print(f"Initializing ERNIE 5.0 [{args.scale}]...")
    print(f"Num Parameters: {model_config.get_num_parameters() / 1e6:.2f}M")
    print(f"Activated Params: {model_config.get_activated_parameters() / 1e6:.2f}M")

    # 2. 模型与Tokenizer
    tokenizer = TextTokenizer(tokenizer_config)
    visual_tokenizer = VisualTokenizer(model_config.visual)
    audio_tokenizer = AudioTokenizer(model_config.audio)
    model = ERNIE5ForCausalLM(model_config)

    # 3. 数据准备
    dataset = MultiModalIterableDataset(file_paths=[], modality="text")
    collator = MultiModalCollator(model_config, tokenizer, visual_tokenizer, audio_tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=training_config.per_device_train_batch_size,
        collate_fn=collator,
    )

    # 4. 训练器
    trainer = ERNIE5Trainer(
        model=model,
        config=training_config,
        train_dataloader=dataloader,
    )

    print("Starting Training...")
    # trainer.train()  # Uncomment to run
    print("Training simulation ready. Run `trainer.train()` to start.")


if __name__ == "__main__":
    main()
