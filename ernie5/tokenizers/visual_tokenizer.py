import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
from ernie5.configs.model_config import VisualConfig

class CausalConv2d(nn.Module):
    """
    因果2D卷积：只看左上方的上下文
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        
        # Padding calculation for causality
        # 对于kernel_size=3, padding需要是2 (on top/left) 来保持对齐或者根据具体需求
        # 这里简化处理：Pad (k-1) * d on top and left
        self.pad_size = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=0, # Manual padding
            dilation=dilation
        )
        
    def forward(self, x):
        # x: [B, C, H, W]
        # Pad: (Left, Right, Top, Bottom)
        # 只在Left和Top填充
        if self.pad_size > 0:
            x = F.pad(x, (self.pad_size, 0, self.pad_size, 0))
        return self.conv(x)


class CausalConv3d(nn.Module):
    """
    因果3D卷积：只看过去帧与左上方空间上下文
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        self.pad_size = (kernel_size - 1) * dilation

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation
        )

    def forward(self, x):
        # x: [B, C, T, H, W]
        if self.pad_size > 0:
            # Pad: (W_left, W_right, H_top, H_bottom, T_front, T_back)
            x = F.pad(x, (self.pad_size, 0, self.pad_size, 0, self.pad_size, 0))
        return self.conv(x)


class VectorQuantizer(nn.Module):
    """
    简化版向量量化器
    """
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta
        
        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1.0 / self.K, 1.0 / self.K)

    def forward(self, latents):
        # latents: [B, C, H, W] -> [B, H, W, C]
        latents = latents.permute(0, 2, 3, 1).contiguous()
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)
        
        # 计算距离
        # dist = (x - e)^2 = x^2 + e^2 - 2xe
        dist = (
            torch.sum(flat_latents ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_latents, self.embedding.weight.t())
        )
        
        # 获取最近的indices
        encoding_indices = torch.argmin(dist, dim=1).unsqueeze(1)
        
        # 量化
        quantized = self.embedding(encoding_indices).view(latents_shape)
        
        # 损失
        e_latent_loss = F.mse_loss(quantized.detach(), latents)
        q_latent_loss = F.mse_loss(quantized, latents.detach())
        loss = q_latent_loss + self.beta * e_latent_loss
        
        # 直通估计 (Straight Through Estimator)
        quantized = latents + (quantized - latents).detach()
        
        # permute back: [B, C, H, W]
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        return quantized, loss, encoding_indices.view(latents_shape[0], latents_shape[1], latents_shape[2])


class BitwiseQuantizer(nn.Module):
    """
    Bit-wise quantization wrapper for different bit widths.
    """
    def __init__(self, bits: int, embedding_dim: int, beta: float = 0.25):
        super().__init__()
        self.bits = bits
        self.codebook_size = 2 ** bits
        self.vq = VectorQuantizer(self.codebook_size, embedding_dim, beta=beta)

    def forward(self, latents):
        return self.vq(latents)


class VisualTokenizer(nn.Module):
    """
    视觉Tokenizer (因果CNN + VQ-VAE)
    
    能够分别处理2D图像和3D视频(简化为逐帧处理或统一3D处理)
    这里实现一个简化的2D版本，视频处理可以通过Time-Distributed方式
    """
    
    def __init__(self, config: VisualConfig):
        super().__init__()
        self.config = config
        
        # Encoder
        # 简化结构：Downsample x2 -> Downsample x2 ...
        self.encoder = nn.Sequential(
            CausalConv2d(3, 64, kernel_size=4, stride=2), # /2
            nn.ReLU(),
            CausalConv2d(64, 128, kernel_size=4, stride=2), # /4
            nn.ReLU(),
            CausalConv2d(128, 256, kernel_size=4, stride=2), # /8
            nn.ReLU(),
            CausalConv2d(256, 256, kernel_size=3, stride=1),
        )
        
        # Quantizer - Bit-wise quantization with progressive schedule
        if config.use_progressive_tokenizer:
            self.quantizers = nn.ModuleList([
                BitwiseQuantizer(bits=b, embedding_dim=256)
                for b in config.tokenizer_bit_schedule
            ])
        else:
            self.quantizers = nn.ModuleList([BitwiseQuantizer(bits=config.tokenizer_bits, embedding_dim=256)])
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # x2
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # x2
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1), # x2
            nn.Tanh() 
        )
        
    def forward(self, x, discriminator: Optional[nn.Module] = None, semantic_model: Optional[nn.Module] = None):
        # x: [B, 3, H, W]
        z = self.encoder(x)
        z_q, loss, indices = self.quantizers[-1](z)
        x_recon = self.decoder(z_q)
        loss_dict = {"vq_loss": loss}

        if discriminator is not None:
            # Simple GAN loss (hinge-style)
            real_logits = discriminator(x)
            fake_logits = discriminator(x_recon.detach())
            gan_loss = F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()
            loss_dict["gan_loss"] = gan_loss

        if semantic_model is not None:
            with torch.no_grad():
                target_feat = semantic_model(x)
            pred_feat = semantic_model(x_recon)
            semantic_loss = F.mse_loss(pred_feat, target_feat)
            loss_dict["semantic_loss"] = semantic_loss

        return x_recon, loss_dict, indices
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image/video to tokens
        Returns: [B, T_visual] indices
        """
        z = self.encoder(x)
        _, _, indices = self.quantizers[-1](z)
        # Flatten indices: [B, H*W]
        return indices.view(indices.size(0), -1)

    def encode_multiscale(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Multi-scale encoding with progressive bit schedule.
        Returns a list of token grids per scale.
        """
        scales = self.config.max_scales
        tokens_per_scale = []
        current = x
        for scale_idx in range(scales):
            z = self.encoder(current)
            quantizer = self.quantizers[min(scale_idx, len(self.quantizers) - 1)]
            _, _, indices = quantizer(z)
            tokens_per_scale.append(indices)
            # Downscale for next scale
            if scale_idx < scales - 1:
                current = F.interpolate(current, scale_factor=0.5, mode="bilinear", align_corners=False)
        return tokens_per_scale
    
    def decode(self, indices: torch.Tensor, shape: Tuple[int, int] = None) -> torch.Tensor:
        """
        Decode tokens to image
        indices: [B, Seq_Len]
        """
        # 需要知道原始grid shape
        if shape is None:
            # 假设正方形
            size = int(indices.size(1) ** 0.5)
            shape = (size, size)
            
        h, w = shape
        indices = indices.view(-1, h, w)
        z_q = self.quantizers[-1].vq.embedding(indices).permute(0, 3, 1, 2) # [B, C, H, W]
        x_recon = self.decoder(z_q)
        return x_recon
