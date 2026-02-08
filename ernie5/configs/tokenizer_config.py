from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class TokenizerConfig:
    """
    Tokenizer配置
    """
    vocab_file: Optional[str] = None
    merges_file: Optional[str] = None
    
    # 特殊Token
    unk_token: str = "<unk>"
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    pad_token: str = "<pad>"
    mask_token: str = "<mask>"
    
    # 多模态特殊Token
    image_token: str = "<image>"
    video_token: str = "<video>"
    audio_token: str = "<audio>"
    
    # 其他控制参数
    do_lower_case: bool = False
    utf16be_fallback: bool = True
    bpe_dropout: float = 0.1
    max_no_space_length: int = 64
