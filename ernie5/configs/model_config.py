"""
ERNIE 5.0 模型配置

定义模型架构的所有超参数，支持简化版和完整版配置切换。
基于技术报告中的设计：
- 超稀疏MoE架构（激活率<3%）
- 统一多模态Token空间
- 弹性训练支持
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Literal, Optional


class ModelScale(Enum):
    """模型规模枚举"""

    MINI = "mini"  # 用于调试：~100M参数
    SMALL = "small"  # 简化版：~1B参数
    MEDIUM = "medium"  # 中等版：~7B参数
    LARGE = "large"  # 大型版：~70B参数
    FULL = "full"  # 完整版：万亿参数


@dataclass
class MoEConfig:
    """混合专家层配置"""

    # 专家数量
    num_experts: int = 16

    # Top-K专家选择（激活的专家数）
    # ERNIE 5.0使用<3%激活率，对于16个专家约为1-2个
    top_k: int = 2

    # 专家隐藏层维度倍数
    expert_ffn_multiplier: float = 2.0

    # 是否使用模态无关路由（ERNIE 5.0核心创新）
    modality_agnostic_routing: bool = True

    # 无辅助损失负载均衡
    auxiliary_loss_free: bool = True

    # 负载均衡bias更新速度
    load_balance_update_speed: float = 1e-4

    # 弹性稀疏度支持
    elastic_sparsity: bool = True

    # 弹性稀疏度的最小Top-K
    min_top_k: int = 1

    # 专家容量因子（每个专家最多处理的token比例）
    capacity_factor: float = 1.25

    # 是否使用共享专家（部分专家被所有token共享）
    use_shared_expert: bool = True
    num_shared_experts: int = 2


@dataclass
class AttentionConfig:
    """注意力机制配置"""

    # 注意力头数
    num_attention_heads: int = 16

    # KV头数（用于分组查询注意力GQA）
    num_kv_heads: Optional[int] = None  # None表示不使用GQA

    # 注意力dropout
    attention_dropout: float = 0.0

    # 是否使用FlashAttention
    use_flash_attention: bool = True

    # 是否使用FlashMask（支持异构注意力模式）
    use_flash_mask: bool = True

    # RoPE基数（用于长序列外推）
    rope_base: float = 1_000_000.0

    # 是否使用Uni-RoPE（统一时空位置编码）
    use_uni_rope: bool = True

    # Uni-RoPE维度分配比例 (t, h, w)
    rope_axis_dim_allocation: List[int] = field(default_factory=lambda: [1, 1, 1])


@dataclass
class ElasticConfig:
    """弹性训练配置"""

    # 是否启用弹性训练
    enabled: bool = True

    # 深度弹性：随机采样浅层子网络的概率
    depth_elastic_prob: float = 0.25

    # 深度弹性：最小保留层比例
    min_depth_ratio: float = 0.5

    # 宽度弹性：随机采样专家子集的概率
    width_elastic_prob: float = 0.20

    # 宽度弹性：最小保留专家比例
    min_width_ratio: float = 0.5

    # 稀疏度弹性：降低Top-K的概率
    sparsity_elastic_prob: float = 0.20


@dataclass
class VisualConfig:
    """视觉模态配置"""

    # 图像分辨率
    image_size: int = 256

    # Patch大小
    patch_size: int = 16

    # 视觉Tokenizer比特数（用于比特量化）
    tokenizer_bits: int = 8

    # 渐进式Tokenizer比特调度（低->高）
    tokenizer_bit_schedule: List[int] = field(default_factory=lambda: [4, 6, 8])
    use_progressive_tokenizer: bool = True

    # 最大尺度数（用于多尺度生成）
    max_scales: int = 4

    # 视频最大帧数
    max_video_frames: int = 16

    # 是否使用双路径编码器
    use_dual_path_encoder: bool = True

    # CNN分支配置
    cnn_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])

    # ViT分支隐藏维度
    vit_hidden_dim: int = 768


@dataclass
class AudioConfig:
    """音频模态配置"""

    # 采样率
    sample_rate: int = 16000

    # Token率（Hz）
    token_rate: float = 12.5

    # RVQ层数（残差向量量化）
    rvq_layers: int = 8

    # 每层码本大小
    codebook_size: int = 1024

    # 最大音频长度（秒）
    max_audio_seconds: float = 30.0

    # 是否使用Whisper蒸馏
    use_whisper_distill: bool = True

    # 是否支持Speaker Embedding
    use_speaker_embedding: bool = True
    speaker_embedding_dim: int = 256


@dataclass
class ERNIE5Config:
    """
    ERNIE 5.0 完整模型配置

    基于技术报告的关键设计：
    1. 超稀疏MoE架构（<3%激活率）
    2. 统一多模态Token空间
    3. Next-Group-of-Tokens预测
    4. 弹性训练支持
    """

    # ========== 基础架构 ==========

    # 隐藏层维度
    hidden_size: int = 2048

    # Transformer层数
    num_hidden_layers: int = 24

    # 中间层维度（FFN）
    intermediate_size: int = 8192

    # 文本词表大小
    vocab_size: int = 100000

    # 最大序列长度
    max_position_embeddings: int = 8192

    # 隐藏层dropout
    hidden_dropout: float = 0.0

    # 层归一化epsilon
    layer_norm_eps: float = 1e-6

    # 初始化范围
    initializer_range: float = 0.02

    # 是否使用RMSNorm（替代LayerNorm）
    use_rms_norm: bool = True

    # ========== MoE配置 ==========

    # 每隔多少层使用MoE
    moe_layer_frequency: int = 2

    # MoE详细配置
    moe: MoEConfig = field(default_factory=MoEConfig)

    # ========== 注意力配置 ==========

    attention: AttentionConfig = field(default_factory=AttentionConfig)

    # ========== 弹性训练配置 ==========

    elastic: ElasticConfig = field(default_factory=ElasticConfig)

    # ========== 多模态配置 ==========

    # 支持的模态
    modalities: List[str] = field(
        default_factory=lambda: ["text", "image", "video", "audio"]
    )

    visual: VisualConfig = field(default_factory=VisualConfig)

    audio: AudioConfig = field(default_factory=AudioConfig)

    # ========== 生成配置 ==========

    # 是否使用多Token预测（MTP）
    use_mtp: bool = True

    # MTP预测的Token组大小
    mtp_group_size: int = 4

    # MTP损失权重
    mtp_loss_weight: float = 0.3

    # 是否使用历史Token损坏训练
    use_history_corruption: bool = True

    # 历史Token损坏比例
    history_corruption_rate: float = 0.1

    # ========== 特殊Token ID ==========

    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # 模态分隔符Token ID
    image_token_id: int = 3
    video_token_id: int = 4
    audio_token_id: int = 5

    # 统一词表偏移 (运行时计算)
    visual_vocab_offset: int = 0
    audio_vocab_offset: int = 0

    @classmethod
    def from_scale(cls, scale: ModelScale) -> "ERNIE5Config":
        """根据模型规模创建预设配置"""

        if scale == ModelScale.MINI:
            # 调试用迷你版本~100M
            return cls(
                hidden_size=512,
                num_hidden_layers=6,
                intermediate_size=2048,
                vocab_size=32000,
                max_position_embeddings=2048,
                moe=MoEConfig(num_experts=4, top_k=1),
                attention=AttentionConfig(num_attention_heads=8),
                visual=VisualConfig(image_size=128, max_scales=2),
                audio=AudioConfig(rvq_layers=4),
            )

        elif scale == ModelScale.SMALL:
            # 简化版~1B参数
            return cls(
                hidden_size=1024,
                num_hidden_layers=12,
                intermediate_size=4096,
                vocab_size=50000,
                max_position_embeddings=4096,
                moe=MoEConfig(num_experts=8, top_k=2),
                attention=AttentionConfig(num_attention_heads=16),
            )

        elif scale == ModelScale.MEDIUM:
            # 中等版~7B参数
            return cls(
                hidden_size=2048,
                num_hidden_layers=24,
                intermediate_size=8192,
                vocab_size=100000,
                max_position_embeddings=8192,
                moe=MoEConfig(num_experts=16, top_k=2),
                attention=AttentionConfig(num_attention_heads=32, num_kv_heads=8),
            )

        elif scale == ModelScale.LARGE:
            # 大型版~70B参数
            return cls(
                hidden_size=4096,
                num_hidden_layers=48,
                intermediate_size=16384,
                vocab_size=150000,
                max_position_embeddings=32768,
                moe=MoEConfig(num_experts=64, top_k=4),
                attention=AttentionConfig(num_attention_heads=64, num_kv_heads=8),
            )

        elif scale == ModelScale.FULL:
            # 完整版（需要大规模集群）
            return cls(
                hidden_size=8192,
                num_hidden_layers=96,
                intermediate_size=32768,
                vocab_size=200000,
                max_position_embeddings=131072,
                moe=MoEConfig(num_experts=256, top_k=8),
                attention=AttentionConfig(num_attention_heads=128, num_kv_heads=16),
            )

        else:
            raise ValueError(f"Unknown model scale: {scale}")

    def get_num_parameters(self, include_moe: bool = True) -> int:
        """估算模型参数量"""

        # 嵌入层参数
        embed_params = self.vocab_size * self.hidden_size

        # 每层Transformer参数（不含MoE）
        attn_params = 4 * self.hidden_size * self.hidden_size  # Q, K, V, O
        ffn_params = 2 * self.hidden_size * self.intermediate_size

        # MoE层参数
        moe_layers = self.num_hidden_layers // self.moe_layer_frequency
        dense_layers = self.num_hidden_layers - moe_layers

        total_params = embed_params
        total_params += dense_layers * (attn_params + ffn_params)

        if include_moe:
            expert_params = (
                2
                * self.hidden_size
                * int(self.intermediate_size * self.moe.expert_ffn_multiplier)
            )
            total_params += moe_layers * (
                attn_params + self.moe.num_experts * expert_params
            )
        else:
            total_params += moe_layers * (attn_params + ffn_params)

        return total_params

    def get_activated_parameters(self) -> int:
        """估算激活参数量（推理时实际使用的参数）"""

        embed_params = self.vocab_size * self.hidden_size
        attn_params = 4 * self.hidden_size * self.hidden_size
        ffn_params = 2 * self.hidden_size * self.intermediate_size

        moe_layers = self.num_hidden_layers // self.moe_layer_frequency
        dense_layers = self.num_hidden_layers - moe_layers

        total_params = embed_params
        total_params += dense_layers * (attn_params + ffn_params)

        # MoE层只激活top_k个专家
        expert_params = (
            2
            * self.hidden_size
            * int(self.intermediate_size * self.moe.expert_ffn_multiplier)
        )
        activated_experts = self.moe.top_k
        if self.moe.use_shared_expert:
            activated_experts += self.moe.num_shared_experts

        total_params += moe_layers * (attn_params + activated_experts * expert_params)

        return total_params

    def __post_init__(self):
        """配置验证"""

        # 确保KV头数能整除注意力头数
        if self.attention.num_kv_heads is not None:
            assert (
                self.attention.num_attention_heads % self.attention.num_kv_heads == 0
            ), "num_attention_heads must be divisible by num_kv_heads"

        # 确保隐藏维度能被注意力头数整除
        assert (
            self.hidden_size % self.attention.num_attention_heads == 0
        ), "hidden_size must be divisible by num_attention_heads"

        # 确保MoE层频率合理
        assert self.moe_layer_frequency > 0, "moe_layer_frequency must be positive"

        # 统一词表偏移
        visual_vocab_size = 2**self.visual.tokenizer_bits
        audio_vocab_size = self.audio.codebook_size
        self.visual_vocab_offset = self.vocab_size
        self.audio_vocab_offset = self.vocab_size + visual_vocab_size

    def get_unified_vocab_size(self) -> int:
        """文本+视觉+音频统一词表大小"""
        visual_vocab_size = 2**self.visual.tokenizer_bits
        audio_vocab_size = self.audio.codebook_size
        return self.vocab_size + visual_vocab_size + audio_vocab_size
