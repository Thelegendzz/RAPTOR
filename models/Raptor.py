#编解码参考simvp重新设计
import time
import logging
import math  # For basic math operations
import torch  # For tensor operations
import numpy as np  # For array operations
import torch.nn as nn  # For neural network layers
import torch.nn.functional as F  # For functional operations
import matplotlib.pyplot as plt
from torch.nn import BatchNorm1d
import os
from piqa import SSIM
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
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
    
class TimeMix(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.batch_norm = nn.LayerNorm(config['t_embd'])

        hidden_sz = 3 * config['t_ffn'] // 2  # can use largeer hidden_sz because of receptance gating
        self.key = nn.Linear(config['t_embd'], hidden_sz)
        self.value = nn.Linear(config['t_embd'], hidden_sz)
        self.weight = nn.Linear(hidden_sz, config['t_embd'])
        self.receptance = nn.Linear(config['t_embd'], config['t_embd'])
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

    def forward(self, x):
        B, T, C = x.size()
        x = torch.cat([self.time_shift(x[:, :, :C // 2]), x[:, :, C // 2:]], dim=-1)

        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)

        wkv = self.weight(F.mish(k) * v)  # i find mish is a bit better than gelu

        rwkv = torch.sigmoid(r) * wkv

        return rwkv
    
class SpaceMix(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.batch_norm = nn.LayerNorm(config['n_embd'])

        hidden_sz = 3 * config['n_ffn'] // 2  # can use largeer hidden_sz because of receptance gating
        self.key = nn.Linear(config['n_embd'], hidden_sz)
        self.value = nn.Linear(config['n_embd'], hidden_sz)
        self.weight = nn.Linear(hidden_sz, config['n_embd'])
        self.receptance = nn.Linear(config['n_embd'], config['n_embd'])

        self.receptance.scale_init = 0
        self.weight.scale_init = 0

    def forward(self, x):
        B, T, C = x.size()
        x = self.batch_norm(x) 
 
        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)

        wkv = self.weight(F.mish(k) * v)  # i find mish is a bit better than gelu

        rwkv = torch.sigmoid(r) * wkv

        return rwkv

class MetaBlock(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.time_mix = TimeMix(config, layer_id)
        self.channel_mix = SpaceMix(config, layer_id)
        self.norm_t = nn.LayerNorm(config['t_embd'])
        self.norm_n = nn.LayerNorm(config['n_embd'])
        
    def forward(self, x):
        # 输入x的形状为 [B, T, C, H, W]
        B, t_embd, n_embd = x.shape
        
        # 重塑为序列形式供注意力层使用
        x = x.permute(0, 2, 1)
        # 应用TimeMix
        residual = x
        x = self.norm_t(x)
        x = self.time_mix(x)
        x = x + residual
        x = x.permute(0, 2, 1)
        
        # 应用ChannelMix
        residual = x
        x = self.norm_n(x)
        x = self.channel_mix(x)
        x = x + residual
        
        return x

class MidMetaNet(nn.Module):
    def __init__(self, N2, config, input_resolution=None):
        super().__init__()
        self.N2 = N2
        
        # 创建N2个RWKVMetaBlock
        self.blocks = nn.ModuleList([
            MetaBlock(config, layer_id=i) 
            for i in range(N2)
        ])
        
    def forward(self, x):       
        # 依次通过每个RWKVMetaBlock
        for block in self.blocks:
            x = block(x)
        
        return x

class Raptor(nn.Module):
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
        
        # 中间的MetaFormer，使用RWKV注意力机制
        self.hid = MidMetaNet(
            N2=config_dict['N_T'],
            config=config_dict,
            input_resolution=(H//2**(config_dict['N_S']//2), W//2**(config_dict['N_S']//2))
        )
        self.in_proj_s = nn.Linear(config_dict['hid_S'] * H//2**(config_dict['N_S']//2) * W//2**(config_dict['N_S']//2), config_dict['n_embd'])
        self.out_proj_s = nn.Linear(config_dict['n_embd'], config_dict['hid_S'] * H//2**(config_dict['N_S']//2) * W//2**(config_dict['N_S']//2))
        self.in_proj_t = nn.Linear(config_dict['n_condition'], config_dict['t_embd'])
        self.out_proj_t = nn.Linear(config_dict['t_embd'], config_dict['n_prediction'])
        self.config_dict = config_dict
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, img, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        # 编码
        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape
        
        # 特征变换（使用RWKV注意力机制）
        x = embed.view(B, T, C_ * H_ * W_)
        x = self.in_proj_s(x)
        x = x.permute(0, 2, 1)
        x = self.in_proj_t(x)
        x = x.permute(0, 2, 1)
        x = self.hid(x)
        x = x.permute(0, 2, 1)
        x = self.out_proj_t(x)
        x = x.permute(0, 2, 1)
        x = self.out_proj_s(x)
        x = x.reshape(B*T, C_, H_, W_)
        
        # 解码
        x = self.dec(x, skip)
        x = x.reshape(B, T, C, H, W)
        x = self.sigmoid(x)
        return x

