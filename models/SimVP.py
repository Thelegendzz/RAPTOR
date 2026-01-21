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
    
class GroupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, act_norm=False):
        super(GroupConv2d, self).__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,groups=groups)
        self.norm = nn.GroupNorm(groups,out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y
    
class Inception(nn.Module):
    def __init__(self, C_in, C_hid, C_out, incep_ker=[3,5,7,11], groups=8):        
        super(Inception, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)
        layers = []
        for ker in incep_ker:
            layers.append(GroupConv2d(C_hid, C_out, kernel_size=ker, stride=1, padding=ker//2, groups=groups, act_norm=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y
    
    
class Mid_Xnet(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T, incep_ker = [3,5,7,11], groups=1):
        super(Mid_Xnet, self).__init__()

        self.N_T = N_T
        self.channel_in = channel_in
        # 第一个Inception层应该接收T*channel_in的输入通道数
        enc_layers = [Inception(channel_in, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))

        dec_layers = [Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_in, incep_ker= incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        # encoder
        skips = []
        z = x
        for i in range(self.N_T):
            z = self.enc[i](z)
            if i < self.N_T - 1:
                skips.append(z)

        # decoder
        z = self.dec[0](z)
        for i in range(1, self.N_T):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        y = z.reshape(B, T, C, H, W)
        return y


class PredictionRWKV(nn.Module):
    """修改后的SimVP模型，移除损失和度量计算"""
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
        self.hid = Mid_Xnet(
            channel_in=config_dict['hid_S'] * config_dict['n_prediction'],  # T*C
            channel_hid=config_dict['hid_S'],
            N_T=config_dict['N_T']
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

