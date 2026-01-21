import torch
import torch.nn.functional as F  # For functional operations
from piqa import SSIM
from torchmetrics import StructuralSimilarityIndexMeasure
import torch.nn as nn
from torchvision import models
import lpips

def calculate_abs_rel(pred, true):
    # Avoid division by zero
    eps = 1e-6
    abs_rel = torch.abs(true - pred) / (true + eps)
    return abs_rel.mean()

def calculate_psnr(tensor1, tensor2, max_pixel=1.0):
    # tensor1, tensor2 的形状为 [batch, channel, pixel]
    max_val = max(tensor1.max().item(), tensor2.max().item())
    mse = F.mse_loss(tensor1, tensor2, reduction='mean')  # 计算 MSE
    psnr = 10 * torch.log10(max_val ** 2 / mse)
    return psnr

def calculate_ssim(tensor1, tensor2, device='cpu'):
    # tensor1, tensor2 的形状为 [batch, channel, height, width]
    # 确保将函数 SSIM 导入并配置正确
    min_val = min(tensor1.min().item(), tensor2.min().item())
    max_val = max(tensor1.max().item(), tensor2.max().item())
    # 计算 data_range
    data_range = max_val - min_val
    tensor1 -= min_val
    tensor2 -= min_val
    tensor1 /= data_range
    tensor2 /= data_range
    ssim_fn = SSIM(n_channels=3).to(device)  # 这里假设通道数为 3
    ssim_value = ssim_fn(tensor1, tensor2)
    return ssim_value

def L1_loss(pred, target):
    """计算 L1 损失"""

    return torch.mean(torch.abs(pred - target))

def SSIM_loss(pred, target):
    """计算SSIM，处理5D张量"""
    device = pred.device
    
    # 如果是5D张量 [B, T, C, H, W]，需要重新排列
    if pred.dim() == 5:
        B, T, C, H, W = pred.shape
        # 重塑为 [B*T, C, H, W]
        pred = pred.view(B*T, C, H, W)
        target = target.view(B*T, C, H, W)
    
    # 创建SSIM度量
    ssim_metric = StructuralSimilarityIndexMeasure(
        data_range=1.0,
        kernel_size=11,  # 使用较小的核大小
        sigma=1.5,  # 调整高斯核的标准差
        reduction='elementwise_mean'
    ).to(device)
    
    try:
        ssim_val = ssim_metric(pred, target)
    except RuntimeError as e:
        print(f"SSIM计算错误: {str(e)}")
        print(f"输入形状: pred={pred.shape}, target={target.shape}")
        # 如果计算失败，返回一个替代值
        ssim_val = torch.tensor(0.5, device=device)
    
    return ssim_val

class vgg_perceptual_loss(nn.Module):
    """VGG感知损失模块"""
    def __init__(self):
        super().__init__()
        try:
            # 新版本PyTorch写法
            self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16]
        except:
            # 兼容旧版本PyTorch
            self.vgg = models.vgg16(pretrained=True).features[:16]
            
        self.vgg = self.vgg.eval()
        # for param in self.vgg.parameters():
        #     param.requires_grad = False
            
    def forward(self, pred, target):
        """计算感知损失"""
        self.vgg.to(pred.device)
        # 转换为float32
        
        with torch.no_grad():
            pred_features = self.vgg(pred)
            target_features = self.vgg(target)
        
        return F.mse_loss(pred_features, target_features)

# def vgg_perceptual_loss(pred, target):
#     """VGG感知损失"""
#     if pred.dim() == 5:
#         B, T, C, H, W = pred.shape
#         # 重塑为 [B*T, C, H, W]
#         pred = pred.view(B*T, C, H, W)
#         target = target.view(B*T, C, H, W)
#     perceptual_loss = VGGPerceptualLoss(pred.device)
#     return perceptual_loss(pred, target)

# def temporal_consistency_loss(pred,config):
#     """时序一致性损失"""
    # if pred.dim() == 5:  # [B, T, C, H, W]
    # temp_diff = pred[:, 1:] - pred[:, :-1]
    # else:  # [B*T, C, H, W]
    #     pred = pred.view(config['batch_size'],config['n_prediction'], config['n_channels'], config['height'], config['width'])
    #     temp_diff = pred[:, 1:] - pred[:, :-1]
    # return torch.mean(torch.abs(temp_diff))

def temporal_consistency_loss(pred):
    if pred.dim() == 5:  # [B, T, C, H, W]
        temp_diff = pred[:, 1:] - pred[:, :-1]
    else:  # [B*T, C, H, W]
        temp_diff = pred[1:] - pred[:-1]
    return torch.mean(torch.abs(temp_diff))

def gradient_difference_loss(pred, target):
    """梯度差分损失"""
    
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    
    # 垂直方向梯度
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    
    # 使用L1损失计算梯度差异
    dx_loss = torch.mean(torch.abs(pred_dx - target_dx))
    dy_loss = torch.mean(torch.abs(pred_dy - target_dy))
    
    return dx_loss + dy_loss

def Contrastive_loss(pred, target, temperature=0.07):
    # target=target.to(pred.device)
    pred_features = F.normalize(pred.view(pred.size(0), -1), dim=1)
    target_features = F.normalize(target.view(target.size(0), -1), dim=1)
    # print(pred_features.device,target_features.device)
    logits = torch.matmul(pred_features, target_features.T) / temperature
    labels = torch.arange(pred.size(0)).to(logits.device)
    # print(logits.device,labels.device)
    return F.cross_entropy(logits, labels)

class LPIPS_Loss(nn.Module):
    """LPIPS感知损失模块"""
    def __init__(self, model='vgg'):
        super().__init__()
        self.lpips = lpips.LPIPS(net=model)
        self.lpips.eval()
        for param in self.lpips.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        """计算LPIPS损失"""
        # 处理5D输入
        if pred.dim() == 5:
            B, T, C, H, W = pred.shape
            # 重塑为 [B*T, C, H, W]
            pred = pred.view(B*T, C, H, W)
            target = target.view(B*T, C, H, W)
            
        # 确保输入在[−1,1]范围内
        pred = (pred - 0.5) * 2
        target = (target - 0.5) * 2
        
        return self.lpips(pred, target).mean()