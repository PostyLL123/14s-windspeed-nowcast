import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEmbedding(nn.Module):
    """
    用于噪音水平 `sigma` 的位置嵌入 (sin/cos 编码)。
    与 SongUNet 中的 `PositionalEmbedding` 类似。
    """
    def __init__(self, num_channels: int, endpoint: bool = False, amp_mode: bool = False):
        super().__init__()
        self.num_channels = num_channels
        self.endpoint = endpoint
        self.amp_mode = amp_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = torch.float32 # 始终使用 float32 计算嵌入
        if self.endpoint:
            freqs = torch.arange(self.num_channels // 2, device=x.device, dtype=dtype)
            freqs = freqs / (self.num_channels // 2 - 1)
        else:
            freqs = torch.arange(self.num_channels // 2, device=x.device, dtype=dtype)
            freqs = freqs / (self.num_channels // 2)
        
        freqs = (10000. ** -freqs).to(dtype=dtype)
        args = x.to(dtype=dtype).ger(freqs) # [B] @ [C/2] -> [B, C/2]
        
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
        return emb.to(dtype=x.dtype) # 转换回原始数据类型

class ResidualBlock1D(nn.Module):
    """
    1D 卷积残差块 (TCN / WaveNet 风格)。
    它接收 (y_noisy) 和 (emb) 作为输入。
    """
    def __init__(self, in_channels: int, out_channels: int, emb_channels: int, dilation: int):
        super().__init__()
        
        # 用于 `emb` 向量的仿射变换 (Affine transform)
        self.emb_proj = nn.Linear(emb_channels, out_channels * 2) # [B, E] -> [B, 2*C_out]
        
        # 1D 卷积
        self.conv1 = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            padding=dilation, 
            dilation=dilation
        )
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1) # Dilation 1
        
        # 残差连接
        self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # x 形状: [B, C, L]
        # emb 形状: [B, Emb]
        
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        
        # 注入条件 (emb)
        # emb_proj 输出 [B, 2*C_out], 拆分为 scale 和 shift
        emb_out = self.emb_proj(emb) # [B, 2*C_out]
        scale, shift = torch.chunk(emb_out, 2, dim=1) # 2 x [B, C_out]
        
        # FiLM 层 (Feature-wise Linear Modulation)
        # [B, C_out, L] = [B, C_out, 1] * [B, C_out, L] + [B, C_out, 1]
        h = h * scale.unsqueeze(-1) + shift.unsqueeze(-1)
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        
        # 残差连接
        return h + self.residual_conv(x)


class Model(nn.Module):
    """
    1D 条件扩散模型 (`net`)。
    它接收 `y_noisy`, `sigma`, 和 `x_past`。
    """
    def __init__(
        self,
        cfg
    ):
        c_out = cfg.c_out
        l_out = cfg.l_out
        c_in = cfg.c_in
        l_in = cfg.l_in
        model_channels = cfg.model_channels
        emb_channels = cfg.emb_channels

        

        super().__init__()
        
        # --- 1. 条件编码器 (Condition Encoder for x_past) ---
        # 目标：将 x_past [B, C_in, L_in] 压缩为上下文向量 c [B, Emb]
        self.condition_encoder = nn.Sequential(
            nn.Conv1d(c_in, model_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
            nn.Conv1d(model_channels, model_channels, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(1), # 全局平均池化, [B, C_model, L_in] -> [B, C_model, 1]
            nn.Flatten(), # [B, C_model]
            nn.Linear(model_channels, emb_channels), # [B, Emb]
            nn.SiLU()
        )

        # --- 2. 噪音/时间步嵌入 (Noise/Time Embedding) ---
        self.map_noise = PositionalEmbedding(num_channels=model_channels)
        self.time_embedding = nn.Sequential(
            nn.Linear(model_channels, emb_channels),
            nn.SiLU(),
            nn.Linear(emb_channels, emb_channels),
            nn.SiLU()
        )
        
        # --- 3. 去噪主干网络 (Denoising Backbone for y_noisy) ---
        # 这是一个 1D TCN (WaveNet 风格)，处理 y_noisy [B, C_out, L_out]
        self.in_conv = nn.Conv1d(c_out, model_channels, kernel_size=3, padding=1)
        
        # 堆叠残差块
        self.blocks = nn.ModuleList([
            ResidualBlock1D(model_channels, model_channels, emb_channels, dilation=1),
            ResidualBlock1D(model_channels, model_channels, emb_channels, dilation=2),
            ResidualBlock1D(model_channels, model_channels, emb_channels, dilation=4),
        ])
        
        self.out_conv = nn.Conv1d(model_channels, c_out, kernel_size=3, padding=1)
        

    def forward(self, 
                y_noisy: torch.Tensor, 
                sigma: torch.Tensor, 
                x_past: torch.Tensor
               ) -> torch.Tensor:
        # y_noisy 形状: [B, C_out, L_out]
        # sigma 形状: [B]
        # x_past 形状: [B, C_in, L_in]

        # 1. 计算条件嵌入 `c` (来自 x_past)
        c = self.condition_encoder(x_past) # [B, Emb]

        # 2. 计算噪音嵌入 `emb` (来自 sigma)
        emb = self.map_noise(sigma) # [B, C_model]
        emb = self.time_embedding(emb) # [B, Emb]
        
        # 3. 合并嵌入
        total_emb = c + emb # [B, Emb]

        # 4. 运行去噪主干网络
        h = self.in_conv(y_noisy) # [B, C_model, L_out]
        
        for block in self.blocks:
            h = block(h, total_emb) # [B, C_model, L_out]
            
        y_pred_residual = self.out_conv(h) # [B, C_out, L_out]
        
        return y_pred_residual
