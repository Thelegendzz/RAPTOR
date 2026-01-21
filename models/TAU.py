#编解码参考simvp重新设计
import time
import logging
import math  # For basic math operations
import torch  # For tensor operations
import numpy as np  # For array operations
import torch.nn as nn  # For neural network layers
import torch.nn.functional as F  # For functional operations
import matplotlib.pyplot as plt
# import utils
from torch.nn import BatchNorm1d
import os
from piqa import SSIM
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
# utils.assert_gpu_runtime()
import os
import sys


def convert_seconds_to_dhms(seconds):
    # days, remainder = divmod(seconds, 86400)  # 一天有86400秒
    hours, remainder = divmod(seconds, 3600)  # 一小时有3600秒
    minutes, seconds = divmod(remainder, 60)  # 一分钟有60秒
    return hours, minutes, int(seconds)

def positional_encoding(sequence_length, dimensions):
    # Half the dimensions since sine and cosine will duplicate it
    half_dimensions = dimensions // 2

    # Generate positions and depths
    positions = torch.arange(sequence_length).unsqueeze(1)  # Shape: (sequence_length, 1)
    depths = torch.arange(half_dimensions).unsqueeze(0)  # Shape: (1, half_dimensions)
    angle_rates = 1 / torch.pow(10000, (depths / float(half_dimensions)))  # Shape: (1, half_dimensions)
    angle_radians = positions * angle_rates  # Shape: (sequence_length, half_dimensions)

    # Apply sine and cosine, and concatenate results
    encoded_positions = torch.cat(
        [torch.sin(angle_radians), torch.cos(angle_radians)], dim=-1)  # Shape: (sequence_length, dimensions)

    return encoded_positions


def load_images_from_folder(folder_path):
    images = []
    filenames = sorted(os.listdir(folder_path))
    for filename in filenames:
        if filename.endswith('.jpg'):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path)
            images.append(img)
    return images


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)

class Encoder(nn.Module):
    """保持原始SimVP的编码器实现"""
    def __init__(self, C_in, C_hid, N_S, spatio_kernel):
        super().__init__()
        samplings = sampling_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0]),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s) 
              for s in samplings[1:]]
        )

    def forward(self, x):
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1

class Decoder(nn.Module):
    """保持原始SimVP的解码器实现"""
    def __init__(self, C_hid, C_out, N_S, spatio_kernel):
        super().__init__()
        samplings = sampling_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s)
              for s in samplings[:-1]],
            ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1])
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](hid + enc1)
        Y = self.readout(Y)
        
        # 确保输出是3通道，并且值域合适
        assert Y.shape[1] == 3, f"Expected 3 channels, got {Y.shape[1]}"
        # Y = torch.tanh(Y)  # 将输出限制在[-1,1]范围
        return Y

class ConvSC(nn.Module):
    """保持原始SimVP的基础卷积块实现"""
    def __init__(self, C_in, C_out, kernel_size=3, downsampling=False, upsampling=False):
        super().__init__()
        padding = (kernel_size - 1) // 2
        stride = 2 if downsampling else 1
        
        self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(C_out)
        self.act = nn.GELU()
        self.upsampling = upsampling

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        if self.upsampling:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x

def sampling_generator(N, reverse=False):
    """生成采样模式"""
    samplings = [False, True] * (N // 2)
    if reverse:
        return list(reversed(samplings[:N]))
    else:
        return samplings[:N]
    

class TAU(nn.Module):
    """Temporal Attention Unit (TAU) - 时间注意力单元"""
    def __init__(self, channel_in, channel_hid, N_T, config_dict=None):
        super().__init__()
        self.channel_in = channel_in
        self.channel_hid = channel_hid
        self.N_T = N_T
        self.config_dict = config_dict
        
        # 计算实际的输入维度：C*H*W
        # 根据编码器的下采样过程计算最终的空间维度
        if config_dict is not None:
            H, W = config_dict['desired_shape']
            N_S = config_dict['N_S']
            
            # 计算下采样后的空间维度
            # sampling_generator(N_S) 产生 [False, True, False, True, False, True] 对于 N_S=6
            # 每个True表示下采样2倍
            downsampling_count = sum(1 for i in range(N_S) if i % 2 == 1)  # 对于N_S=6，有3次下采样
            
            final_H = H // (2 ** downsampling_count)
            final_W = W // (2 ** downsampling_count)
            
            # 实际输入维度是 C * final_H * final_W
            actual_input_dim = channel_in * final_H * final_W
            
            print(f"TAU模块初始化: 输入维度 = {channel_in} * {final_H} * {final_W} = {actual_input_dim}")
        else:
            # 如果没有配置，使用默认值（这种情况不应该发生）
            actual_input_dim = channel_in
            print(f"警告: 没有配置信息，使用默认输入维度 {actual_input_dim}")
        
        # 使用实际的输入维度初始化投影层
        self.input_proj = nn.Linear(actual_input_dim, channel_hid)
        self.output_proj = nn.Linear(channel_hid, actual_input_dim)
        
        # 时间注意力层
        self.temporal_blocks = nn.ModuleList([
            TemporalAttentionBlock(channel_hid, config_dict) for _ in range(N_T)
        ])
        
        # 层归一化
        self.norm = nn.LayerNorm(channel_hid)
        
    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) - 输入特征
        Returns:
            (B, T, C, H, W) - 输出特征
        """
        B, T, C, H, W = x.shape
        
        # 重塑为 (B, T, C*H*W) 进行时间建模
        x = x.view(B, T, -1)
        
        # 输入投影
        x = self.input_proj(x)  # (B, T, channel_hid)
        
        # 通过时间注意力块
        for block in self.temporal_blocks:
            x = block(x) + x  # 残差连接
            x = self.norm(x)
        
        # 输出投影
        x = self.output_proj(x)  # (B, T, C*H*W)
        
        # 重塑回原始形状
        x = x.view(B, T, C, H, W)
        
        return x

class TemporalAttentionBlock(nn.Module):
    """时间注意力块"""
    def __init__(self, dim, config_dict=None):
        super().__init__()
        self.dim = dim
        
        # 从配置文件获取参数，如果没有则使用默认值
        if config_dict is not None:
            default_num_heads = config_dict.get('tau_num_heads', 8)
            dropout = config_dict.get('tau_dropout', 0.1)
        else:
            default_num_heads = 8
            dropout = 0.1

        # 确保 num_heads 能整除 dim
        self.num_heads = min(default_num_heads, dim)
        # 找到能整除 dim 的最大 num_heads
        while dim % self.num_heads != 0 and self.num_heads > 1:
            self.num_heads -= 1
            
        self.head_dim = dim // self.num_heads
        
        print(f"TemporalAttentionBlock: dim={dim}, adjusted num_heads={self.num_heads}, head_dim={self.head_dim}")
        
        # 多头注意力
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (B, T, dim)
        Returns:
            (B, T, dim)
        """
        # 多头自注意力
        residual = x
        x = self.norm1(x)
        
        B, T, dim = x.shape
        
        # 计算 Q, K, V
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        
        # 计算注意力权重
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 因果掩码（可选，用于时间序列预测）
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, v)  # (B, num_heads, T, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, dim)
        
        # 输出投影
        attn_output = self.out_proj(attn_output)
        x = residual + self.dropout(attn_output)
        
        # 前馈网络
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x

class TAUWithRegularization(nn.Module):
    """带有差分散度正则化的TAU模块"""
    def __init__(self, channel_in, channel_hid, N_T, config_dict=None):
        super().__init__()
        # 从配置文件获取tau参数
        if config_dict is not None:
            self.tau = config_dict.get('tau', 0.1)
        else:
            self.tau = 0.1
            
        self.tau_module = TAU(channel_in, channel_hid, N_T, config_dict)
        
    def forward(self, x):
        return self.tau_module(x)
    
    def diff_div_reg(self, pred_y, batch_y, eps=1e-12):
        """
        计算差分散度正则化损失
        Args:
            pred_y: 预测结果 (B, T, C, H, W)
            batch_y: 真实标签 (B, T, C, H, W)
        Returns:
            正则化损失
        """
        B, T = pred_y.shape[:2]
        if T <= 2:
            return torch.tensor(0.0, device=pred_y.device)
        
        # 计算时间差分
        gap_pred_y = (pred_y[:, 1:] - pred_y[:, :-1]).reshape(B, T-1, -1)
        gap_batch_y = (batch_y[:, 1:] - batch_y[:, :-1]).reshape(B, T-1, -1)
        
        # 计算softmax
        softmax_gap_p = F.softmax(gap_pred_y / self.tau, -1)
        softmax_gap_b = F.softmax(gap_batch_y / self.tau, -1)
        
        # 计算KL散度
        loss_gap = softmax_gap_p * torch.log(softmax_gap_p / (softmax_gap_b + eps) + eps)
        
        return loss_gap.mean()

# 更新PredictionRWKV类以使用新的TAU模块
class PredictionRWKV(nn.Module):
    """修改后的SimVP模型，使用TAU模块"""
    def __init__(self, config_dict):
        super().__init__()
        T, C = config_dict['n_prediction'], config_dict['n_channels']
        H, W = config_dict['desired_shape']
        
        # 编码器
        self.enc = Encoder(
            C_in=C,
            C_hid=config_dict['hid_S'],
            N_S=config_dict['N_S'],
            spatio_kernel=config_dict['spatio_kernel_enc']
        )
        
        # 解码器
        self.dec = Decoder(
            C_hid=config_dict['hid_S'],
            C_out=C,
            N_S=config_dict['N_S'],
            spatio_kernel=config_dict['spatio_kernel_dec']
        )
        
        # TAU模块替代原来的MetaFormer
        self.hid = TAUWithRegularization(
            channel_in=config_dict['hid_S'],
            channel_hid=config_dict['hid_S'],
            N_T=config_dict['N_T'],
            config_dict=config_dict
        )
        self.config_dict = config_dict
    
    def forward(self, img, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        # 编码
        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape
        
        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)
        hid = hid.reshape(B*T, C_, H_, W_)
        
        # 解码
        x = self.dec(hid, skip)
        x = x.reshape(B, T, C, H, W)
        return x