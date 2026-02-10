from typing import Any, Dict, List, Optional, Union

import torch

from ernie5.configs.model_config import ERNIE5Config
from ernie5.tokenizers.audio_tokenizer import AudioTokenizer
from ernie5.tokenizers.text_tokenizer import TextTokenizer
from ernie5.tokenizers.visual_tokenizer import VisualTokenizer


class MultiModalCollator:
    """
    多模态数据整理器 (Collator)

    核心功能：
    1. 将文本 Token 化
    2. 将图像/音频通过各自 Tokenizer 转换为 Discrete Codes (或保留像素供模型内部处理)
    3. 构建统一的 input_ids 序列 (Text IDs + Special IDs + Media Placeholders)
    4. 构建 Attention Mask (处理 Uni-RoPE 和 FlashMask)
    5. 构建 Labels (用于计算 Loss)

    注意：为了简化，这里假设模型内部处理 Raw Images/Audio 到底层 Code，
    或者我们在 Collator 这里离线处理。
    技术报告中提到 Tokenizer 是在训练前/中使用的。
    如果是端到端训练，Input 通常是 Raw Pixel。
    如果是两阶段，Input 可能是 Pre-tokenized Codes。

    ERNIE 5.0 是 "Unified Autoregressive"，Input 是 Unified Tokens。
    因此，Collator 应该负责把多模态数据平铺成一长串 ID。
    """

    def __init__(
        self,
        config: ERNIE5Config,
        tokenizer: TextTokenizer,
        visual_tokenizer: Optional[VisualTokenizer] = None,
        audio_tokenizer: Optional[AudioTokenizer] = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.visual_tokenizer = visual_tokenizer
        self.audio_tokenizer = audio_tokenizer

        # 特殊 ID
        self.pad_id = config.pad_token_id
        self.img_token_id = config.image_token_id
        # ...

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        batch: List of samples from Dataset
        """
        input_ids_list = []
        labels_list = []
        visual_labels_list = []
        audio_labels_list = []
        attention_mask_list = []
        position_ids_list = []

        # modality masks
        text_mask_list = []
        visual_mask_list = []
        audio_mask_list = []

        for sample in batch:
            sample_ids = []
            sample_pos = []
            token_groups = []  # group id for FlashMask
            token_modalities = []  # "text" | "visual" | "audio"
            visual_label_tokens = []
            audio_label_tokens = []

            group_counter = 0
            t_cursor = 0

            # Normalize to interleaved format
            if sample.get("type") == "text":
                content = [{"type": "text", "value": sample["content"]}]
            elif sample.get("type") == "image_text":
                content = [
                    {"type": "text", "value": sample.get("text", "")},
                    {"type": "image", "value": sample.get("image")},
                ]
            elif sample.get("type") == "audio_text":
                content = [
                    {"type": "text", "value": sample.get("text", "")},
                    {"type": "audio", "value": sample.get("audio")},
                ]
            else:
                content = sample.get("content", [])

            for item in content:
                if item["type"] == "text":
                    ids = self.tokenizer.encode(item["value"])
                    sample_ids.extend(ids)
                    seq_len = len(ids)
                    pos = torch.zeros((seq_len, 3), dtype=torch.long)
                    pos[:, 0] = torch.arange(t_cursor, t_cursor + seq_len)
                    sample_pos.append(pos)
                    token_groups.extend([-1] * seq_len)
                    token_modalities.extend(["text"] * seq_len)
                    visual_label_tokens.extend([-100] * seq_len)
                    audio_label_tokens.extend(
                        [[-100] * self.config.audio.rvq_layers] * seq_len
                    )
                    t_cursor += seq_len

                elif item["type"] == "image":
                    if self.visual_tokenizer is None:
                        raise ValueError(
                            "visual_tokenizer is required for image inputs"
                        )
                    # add image special token
                    sample_ids.append(self.config.image_token_id)
                    token_groups.append(-1)
                    token_modalities.append("text")
                    visual_label_tokens.append(-100)
                    audio_label_tokens.append([-100] * self.config.audio.rvq_layers)
                    sample_pos.append(
                        torch.tensor([[t_cursor, 0, 0]], dtype=torch.long)
                    )
                    t_cursor += 1

                    tokens_per_scale = self.visual_tokenizer.encode_multiscale(
                        item["value"].unsqueeze(0)
                    )
                    for scale_tokens in tokens_per_scale:
                        # scale_tokens: [B, H, W]
                        h, w = scale_tokens.shape[-2], scale_tokens.shape[-1]
                        flat = scale_tokens.view(-1)
                        # offset into unified vocab
                        flat_ids = flat + self.config.visual_vocab_offset
                        sample_ids.extend(flat_ids.tolist())

                        # position ids (t=0 for image), h,w grid
                        grid_h = torch.arange(h).repeat_interleave(w)
                        grid_w = torch.arange(w).repeat(h)
                        pos = torch.stack(
                            [torch.zeros(h * w, dtype=torch.long), grid_h, grid_w],
                            dim=-1,
                        )
                        sample_pos.append(pos)

                        token_groups.extend([group_counter] * (h * w))
                        token_modalities.extend(["visual"] * (h * w))
                        visual_label_tokens.extend(flat.tolist())
                        audio_label_tokens.extend(
                            [[-100] * self.config.audio.rvq_layers] * (h * w)
                        )
                        group_counter += 1

                elif item["type"] == "audio":
                    if self.audio_tokenizer is None:
                        raise ValueError("audio_tokenizer is required for audio inputs")
                    sample_ids.append(self.config.audio_token_id)
                    token_groups.append(-1)
                    token_modalities.append("text")
                    visual_label_tokens.append(-100)
                    audio_label_tokens.append([-100] * self.config.audio.rvq_layers)
                    sample_pos.append(
                        torch.tensor([[t_cursor, 0, 0]], dtype=torch.long)
                    )
                    t_cursor += 1

                    codes = self.audio_tokenizer.encode(item["value"].unsqueeze(0))
                    # codes: [B, layers, T]
                    codes = codes[0]  # [layers, T]
                    # Use first layer as tokens for unified stream, offset
                    audio_tokens = codes[0] + self.config.audio_vocab_offset
                    sample_ids.extend(audio_tokens.tolist())
                    seq_len = audio_tokens.numel()
                    pos = torch.zeros((seq_len, 3), dtype=torch.long)
                    pos[:, 0] = torch.arange(t_cursor, t_cursor + seq_len)
                    sample_pos.append(pos)
                    token_groups.extend([-1] * seq_len)
                    token_modalities.extend(["audio"] * seq_len)
                    visual_label_tokens.extend([-100] * seq_len)
                    # store full RVQ codes per token
                    audio_label_tokens.extend(
                        codes[:, :seq_len].transpose(0, 1).tolist()
                    )
                    t_cursor += seq_len

            # Pad or truncate
            if len(sample_ids) > self.config.max_position_embeddings:
                sample_ids = sample_ids[: self.config.max_position_embeddings]
                token_groups = token_groups[: self.config.max_position_embeddings]
                token_modalities = token_modalities[
                    : self.config.max_position_embeddings
                ]
                visual_label_tokens = visual_label_tokens[
                    : self.config.max_position_embeddings
                ]
                audio_label_tokens = audio_label_tokens[
                    : self.config.max_position_embeddings
                ]

            input_ids_list.append(torch.tensor(sample_ids, dtype=torch.long))

            visual_labels_list.append(
                torch.tensor(visual_label_tokens, dtype=torch.long)
            )
            audio_labels_list.append(torch.tensor(audio_label_tokens, dtype=torch.long))

            # position ids
            if len(sample_pos) > 0:
                pos_all = torch.cat(sample_pos, dim=0)
            else:
                pos_all = torch.zeros((len(sample_ids), 3), dtype=torch.long)
            position_ids_list.append(pos_all)

            # attention mask (FlashMask)
            seq_len = len(sample_ids)
            attn_mask = torch.ones((seq_len, seq_len), dtype=torch.bool)
            for i in range(seq_len):
                for j in range(seq_len):
                    if token_modalities[i] == "text" or token_modalities[i] == "audio":
                        if j <= i:
                            attn_mask[i, j] = False
                    else:
                        gi = token_groups[i]
                        gj = token_groups[j]
                        if gj == gi or (gj != -1 and gj < gi) or (gj == -1 and j <= i):
                            attn_mask[i, j] = False
            attention_mask_list.append(attn_mask)

            # modality masks
            text_mask_list.append(
                torch.tensor([m == "text" for m in token_modalities], dtype=torch.bool)
            )
            visual_mask_list.append(
                torch.tensor(
                    [m == "visual" for m in token_modalities], dtype=torch.bool
                )
            )
            audio_mask_list.append(
                torch.tensor([m == "audio" for m in token_modalities], dtype=torch.bool)
            )

        # Padding
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.pad_id
        )

        max_len = input_ids.size(1)
        position_ids = torch.zeros((len(batch), max_len, 3), dtype=torch.long)
        attention_mask = torch.ones((len(batch), max_len, max_len), dtype=torch.bool)

        for i in range(len(batch)):
            seq_len = position_ids_list[i].size(0)
            position_ids[i, :seq_len, :] = position_ids_list[i]
            attention_mask[i, :seq_len, :seq_len] = attention_mask_list[i]

        text_mask = torch.nn.utils.rnn.pad_sequence(
            text_mask_list, batch_first=True, padding_value=False
        )
        visual_mask = torch.nn.utils.rnn.pad_sequence(
            visual_mask_list, batch_first=True, padding_value=False
        )
        audio_mask = torch.nn.utils.rnn.pad_sequence(
            audio_mask_list, batch_first=True, padding_value=False
        )

        # Labels usually same as Input for AR (shifted inside model).
        # 对于 padding 位置必须使用 -100 以避免被 CE loss 计入。
        labels = input_ids.clone()
        labels[labels == self.pad_id] = -100
        visual_labels = torch.nn.utils.rnn.pad_sequence(
            visual_labels_list, batch_first=True, padding_value=-100
        )
        audio_labels = torch.nn.utils.rnn.pad_sequence(
            audio_labels_list, batch_first=True, padding_value=-100
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "text_mask": text_mask,
            "visual_mask": visual_mask,
            "audio_mask": audio_mask,
            "visual_labels": visual_labels,
            "audio_labels": audio_labels,
        }
