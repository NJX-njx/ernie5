from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ernie5.configs.model_config import AudioConfig
from ernie5.tokenizers.visual_tokenizer import VectorQuantizer


class VectorQuantizer1d(nn.Module):
    """
    1D向量量化器，用于音频特征 [B, C, T]
    """

    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta
        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1.0 / self.K, 1.0 / self.K)

    def forward(self, latents):
        # latents: [B, C, T] -> [B, T, C]
        latents = latents.transpose(1, 2).contiguous()
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)

        dist = (
            torch.sum(flat_latents**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_latents, self.embedding.weight.t())
        )

        encoding_indices = torch.argmin(dist, dim=1).unsqueeze(1)
        quantized = self.embedding(encoding_indices).view(latents_shape)

        e_latent_loss = F.mse_loss(quantized.detach(), latents)
        q_latent_loss = F.mse_loss(quantized, latents.detach())
        loss = q_latent_loss + self.beta * e_latent_loss

        quantized = latents + (quantized - latents).detach()
        quantized = quantized.transpose(1, 2).contiguous()

        return (
            quantized,
            loss,
            encoding_indices.view(latents_shape[0], latents_shape[1]),
        )


class ResidualVectorQuantizer(nn.Module):
    """
    残差向量量化器 (RVQ)
    """

    def __init__(self, num_quantizers, num_embeddings, embedding_dim, dropout=0.0):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.dropout = dropout

        self.layers = nn.ModuleList(
            [
                VectorQuantizer1d(num_embeddings, embedding_dim)
                for _ in range(num_quantizers)
            ]
        )

    def forward(self, x, n_layers: int = None):
        # x: [B, C, T]
        if n_layers is None:
            n_layers = self.num_quantizers

        quantized_out = 0.0
        residual = x

        all_losses = []
        all_indices = []

        for i, layer in enumerate(self.layers[:n_layers]):
            # layer returns: quantized, loss, indices
            quantized, loss, indices = layer(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized

            all_losses.append(loss)
            all_indices.append(indices)

        return (
            quantized_out,
            torch.stack(all_losses).mean(),
            torch.stack(all_indices, dim=1),
        )


class AudioTokenizer(nn.Module):
    """
    音频Tokenizer (Encoder-Decoder + RVQ)
    简化的EnCodec风格架构
    """

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config

        # Encoder (Conv1d stacks)
        # 输入: [B, 1, T_sample] -> 输出: [B, D, T_token]
        # 简化版：几层Strided Conv
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3),
            nn.ELU(),
            nn.Conv1d(32, 64, kernel_size=8, stride=4, padding=2),  # /4
            nn.ELU(),
            nn.Conv1d(64, 128, kernel_size=8, stride=4, padding=2),  # /16
            nn.ELU(),
            nn.Conv1d(128, 256, kernel_size=8, stride=4, padding=2),  # /64
            nn.ELU(),
            nn.Conv1d(256, 512, kernel_size=8, stride=5, padding=2),  # /320
            nn.ELU(),
            nn.Conv1d(512, 512, kernel_size=8, stride=4, padding=2),  # /1280
        )
        # Total stride = 4*4*4*5 = 320.
        # For 16khz, token rate = 16000/320 = 50Hz (Standard is often higher, config says 12.5Hz)
        # 调整stride以匹配 12.5Hz (16000 / 1280 = 12.5) => Need factor 1280
        # 4*4*4*5*4 = 1280

        self.final_enc_norm = nn.Conv1d(512, 512, kernel_size=1)

        # RVQ
        self.quantizer = ResidualVectorQuantizer(
            num_quantizers=config.rvq_layers,
            num_embeddings=config.codebook_size,
            embedding_dim=512,
        )

        # Decoder (ConvTranspose1d stacks)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(512, 512, kernel_size=8, stride=4, padding=2),
            nn.ELU(),
            nn.ConvTranspose1d(512, 256, kernel_size=8, stride=5, padding=2),
            nn.ELU(),
            nn.ConvTranspose1d(256, 128, kernel_size=8, stride=4, padding=2),
            nn.ELU(),
            nn.ConvTranspose1d(128, 64, kernel_size=8, stride=4, padding=2),
            nn.ELU(),
            nn.ConvTranspose1d(64, 32, kernel_size=8, stride=4, padding=2),
            nn.ELU(),
            nn.Conv1d(32, 1, kernel_size=7, stride=1, padding=3),
            nn.Tanh(),  # Audio usually normalized -1 to 1
        )

    def forward(self, x, teacher_model: Optional[nn.Module] = None):
        # x: [B, 1, T]
        z = self.encoder(x)
        z = self.final_enc_norm(z)

        z_q, loss, indices = self.quantizer(z)

        x_recon = self.decoder(z_q)
        loss_dict = {"vq_loss": loss}

        if self.config.use_whisper_distill and teacher_model is not None:
            with torch.no_grad():
                teacher_feat = teacher_model(x)
            # Use level-0 quantized features for distillation alignment
            student_feat = z_q.mean(dim=-1)
            distill_loss = F.mse_loss(student_feat, teacher_feat)
            loss_dict["distill_loss"] = distill_loss

        return x_recon, loss_dict, indices

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode audio to codes
        Returns: [B, n_codebooks, T_token]
        """
        z = self.encoder(x)
        z = self.final_enc_norm(z)
        _, _, indices = self.quantizer(z)
        return indices

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode codes to audio
        indices: [B, n_codebooks, T_token]
        """
        # Look up embeddings from RVQ layers
        z_q = 0.0
        for i in range(indices.size(1)):  # Iterate over layers
            # layer i
            layer_indices = indices[:, i, :]  # [B, T]
            layer_emb = self.quantizer.layers[i].embedding(layer_indices)  # [B, T, D]
            z_q = z_q + layer_emb.transpose(1, 2)  # [B, D, T]

        x_recon = self.decoder(z_q)
        return x_recon
