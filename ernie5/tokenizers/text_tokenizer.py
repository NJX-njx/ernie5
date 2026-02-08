import os
import re
from typing import List, Union, Dict, Iterable
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from ernie5.configs.tokenizer_config import TokenizerConfig

class TextTokenizer:
    """
    ERNIE 5.0 文本Tokenizer
    
    基于BPE，支持：
    - UTF-16BE编码 (模拟)
    - 多模态特殊Token
    - 训练与加载
    """
    
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self._tokenizer = None
        
        # 如果提供了词表文件，则加载
        if config.vocab_file and os.path.exists(config.vocab_file):
            self.load(config.vocab_file)
        else:
            # 否则初始化一个新的Tokenizer用于训练
            self._init_new_tokenizer()
            
    def _init_new_tokenizer(self):
        # 使用BPE模型
        self._tokenizer = Tokenizer(models.BPE(dropout=self.config.bpe_dropout))
        
        # 预分词：字节级
        self._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        
        # 解码器
        self._tokenizer.decoder = decoders.ByteLevel()
        
        # 后处理 (TemplateProcessing) 留空，手动处理bos/eos
        
    def _iter_filtered_lines(self, files: List[str]) -> Iterable[str]:
        """按报告要求过滤长无空格短语，并进行UTF-16BE预处理。"""
        no_space_re = re.compile(r"\S{%d,}" % self.config.max_no_space_length)
        for path in files:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip("\n")
                    if not line:
                        continue
                    if no_space_re.search(line):
                        continue
                    yield self._preprocess_text(line)

    def train(self, files: List[str], vocab_size: int = 50000):
        """训练Tokenizer"""
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=[
                self.config.unk_token,
                self.config.bos_token,
                self.config.eos_token,
                self.config.pad_token,
                self.config.mask_token,
                self.config.image_token,
                self.config.video_token,
                self.config.audio_token,
            ],
            show_progress=True
        )
        
        # 使用 iterator 以支持过滤与预处理
        self._tokenizer.train_from_iterator(self._iter_filtered_lines(files), trainer)
        
    def save(self, path: str):
        """保存Tokenizer"""
        self._tokenizer.save(path)
        
    def load(self, path: str):
        """加载Tokenizer"""
        self._tokenizer = Tokenizer.from_file(path)
        
    def _preprocess_text(self, text: str) -> str:
        if self.config.do_lower_case:
            text = text.lower()
        if self.config.utf16be_fallback:
            # 将UTF-16BE字节映射为latin-1字符串，保持字节值不变
            text = text.encode("utf-16-be", errors="ignore").decode("latin-1", errors="ignore")
        return text

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """编码文本"""
        text = self._preprocess_text(text)
        encoded = self._tokenizer.encode(text)
        ids = encoded.ids
        
        if add_special_tokens:
            # 这里需要获取token id，临时用magic number或者查找
            # 实际应从tokenizer获取id
            bos_id = self._tokenizer.token_to_id(self.config.bos_token)
            eos_id = self._tokenizer.token_to_id(self.config.eos_token)
            
            if bos_id is not None:
                ids = [bos_id] + ids
            if eos_id is not None:
                ids = ids + [eos_id]
                
        return ids
        
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """解码ID序列"""
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
    
    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()
    
    def token_to_id(self, token: str) -> int:
        return self._tokenizer.token_to_id(token)
    
    def id_to_token(self, id: int) -> str:
        return self._tokenizer.id_to_token(id)
