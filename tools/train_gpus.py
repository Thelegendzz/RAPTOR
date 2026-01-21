import os
import time
import logging
from datetime import datetime
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import torch.multiprocessing as mp
from torchmetrics import StructuralSimilarityIndexMeasure
import torch.nn.functional as F
from torchvision import models
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.utils.checkpoint import checkpoint
import math
from models.SIMVP import PredictionRWKV as EWKV, ResBlock
from configs.configurations import config_512
from datasets.CustomDatasets import (MMNISTDatasetCustominout, KTH_Dataset, KITTI_Dataset,AirMot_Dataset_v0, 
                          UAVID_Images_Dataset_v0, UAVID_Images_Dataset_v1, Jinan_Dataset)
from utils.utils import convert_seconds_to_dhms, get_prog_bars
from utils.metrics import calculate_psnr, calculate_abs_rel, L1_loss, SSIM_loss, vgg_perceptual_loss, gradient_difference_loss, temporal_consistency_loss, Contrastive_loss, LPIPS_Loss#, VAE_Loss  # 添加 LPIPS_Loss

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
vgg_perceptual_loss = vgg_perceptual_loss()
def setup(rank, world_size):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12349'
    # os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """清理分布式训练环境"""
    dist.destroy_process_group()

def get_dataloader(dataset, config, rank, world_size, is_train=True):
    """创建分布式数据加载器"""
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=is_train
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=False,  # 由Sampler管理shuffle
        num_workers=config['num_workers'],
        pin_memory=True,
        sampler=sampler,
        prefetch_factor=config['prefetch_factor'],
        persistent_workers=True
    )
    
    return dataloader, sampler

def save_checkpoint(model, optimizer, epoch, loss, path, is_best=False):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    if is_best:
        torch.save(checkpoint, os.path.join(path, 'best_model.pth'))
    else:
        torch.save(checkpoint, os.path.join(path, f'checkpoint_epoch_{epoch}.pth'))


def validate(model, val_loader, criterion, ssim_metric, lpips_metric, config, device):
    """验证模型"""
    model.eval()
    torch.cuda.empty_cache()  # 验证前清理显存
    total_loss = 0
    loss_stats = {
        'l1': 0,
        'mse': 0,
        'ssim': 0,
        'gdl': 0,
        'temporal': 0,
        'perceptual': 0,
        'contrastive': 0,
        'lpips': 0  # 添加LPIPS
    }
    metric_stats = {
        'ssim': [],
        'psnr': [],
        'abs_rel': [],
        'lpips': []  # 添加LPIPS
    }
    batch_count = 0

    with torch.no_grad():
        for image, flows, label, _ in val_loader:
            image = image.to(device)
            flows = flows.to(device)
            label = label.to(device)
            
            output = model(image, flows)
            
            if output is None or torch.isnan(output).any():
                continue
            
            x_flat = output.reshape(
                image.size(0) * config['n_prediction'],
                config['n_channels'],
                config['desired_shape'][0],
                config['desired_shape'][1]
            ).float()
            labels_flat = label.reshape(
                label.size(0) * config['n_prediction'],
                config['n_channels'],
                config['desired_shape'][0],
                config['desired_shape'][1]
            ).float()
            ssim_value = ssim_metric(x_flat, labels_flat)
            psnr_value = calculate_psnr(output, label)
            rel_value = calculate_abs_rel(output, label)
            # 计算所有损失
            l1_loss = L1_loss(x_flat, labels_flat)
            mse_loss = criterion(x_flat, labels_flat)
            ssim_loss = 1 - ssim_value
            gdl_loss = gradient_difference_loss(x_flat, labels_flat)
            temp_loss = temporal_consistency_loss(output) if config.get('weight_temporal', 0) > 0 else 0
            perceptual_loss = vgg_perceptual_loss(x_flat, labels_flat) if config.get('weight_perceptual', 0) > 0 else 0
            contrastive_loss = Contrastive_loss(x_flat, labels_flat) if config.get('weight_contrastive', 0) > 0 else 0
            lpips_loss = lpips_metric(x_flat, labels_flat) if config.get('weight_lpips', 0) > 0 else 0  # 计算LPIPS损失
            
           
            # 计算评估指标
            
            
            # 累计损失和指标
            total_loss += (
                config.get('weight_l1', 1.0) * l1_loss.item()
                + config.get('weight_mse', 1.0) * mse_loss.item()
                + config.get('weight_ssim', 0.01) * ssim_loss.item()
                + config.get('weight_gdl', 0.1) * gdl_loss.item()
                + config.get('weight_temporal', 0.3) * (temp_loss.item() if temp_loss != 0 else 0)
                + config.get('weight_perceptual', 0.001) * (perceptual_loss.item() if perceptual_loss != 0 else 0)
                + config.get('weight_contrastive', 0.1) * (contrastive_loss.item() if contrastive_loss != 0 else 0)
                + config.get('weight_lpips', 0.1) * (lpips_loss.item() if lpips_loss != 0 else 0)  # 添加LPIPS损失
            )
            loss_stats['l1'] += l1_loss.item()
            loss_stats['mse'] += mse_loss.item()
            # loss_stats['ssim'] += ssim_loss.item()
            loss_stats['gdl'] += gdl_loss.item()
            if temp_loss != 0:
                loss_stats['temporal'] += temp_loss.item()
            if perceptual_loss != 0:
                loss_stats['perceptual'] += perceptual_loss.item()
            if contrastive_loss != 0:
                loss_stats['contrastive'] += contrastive_loss.item()
            if lpips_loss != 0:
                loss_stats['lpips'] += lpips_loss.item()  # 累计LPIPS损失
                
            metric_stats['ssim'].append(ssim_value.item())
            metric_stats['psnr'].append(psnr_value.item())
            metric_stats['abs_rel'].append(rel_value.item())
            
            
            batch_count += 1
    
    # 在计算平均值之前添加检查
    if batch_count == 0:
        logging.warning("No valid batches found during validation!")
        return {
            'loss': float('inf'),
            'mse': 0,
            'ssim': 0,
            'gdl': 0,
            'temporal': 0,
            'perceptual': 0,
            'contrastive': 0,
            'lpips': 0,  # 添加LPIPS
            'psnr': 0,
            'abs_rel': 0
        }
    
    # 计算平均
    metrics = {
        'loss': total_loss / batch_count,
        **{k: v/batch_count for k, v in loss_stats.items()},
        **{k: np.mean(v) if v else 0 for k, v in metric_stats.items()}
    }
    return metrics

# 添加梯度检查点装饰器
def checkpoint_wrapper(module):
    def forward_wrapper(*args, **kwargs):
        return checkpoint(module, *args, **kwargs)
    return forward_wrapper

# 添加优化器创建函数
def create_optimizer(model, config):
    # 参数分组 - 区分权重衰减
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'bias' in name or 'norm' in name or 'ln' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    
    optimizer_grouped_params = [
        {'params': decay_params, 'weight_decay': config['weight_decay']},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    optimizer = torch.optim.AdamW(
        optimizer_grouped_params,
        lr=config['lr_init'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    return optimizer

# 添加学习率调度器创建函数
def create_scheduler(optimizer, config, num_training_steps):
    warmup_steps = int(config['warmup_epochs'] * num_training_steps / config['n_epochs'])
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
        return max(config['lr_min'] / config['lr_init'], 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler

def train(rank, world_size, config):
    """DDP训练主函数"""
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    
    # 初始化模型
    model = EWKV(config)
    # 应用梯度检查点
    if config.get('use_checkpoint', False):
        for module in model.modules():
            if isinstance(module, (RWKVMetaBlock, ResBlock)):
                module.forward = checkpoint_wrapper(module.forward)
    
    # 加载模型(保持原有的加载逻辑)
    if config['if_load_model']:
        try:
            model_dict = os.path.join(config['output_path'], 'best_model.pth ')
            checkpoint = torch.load(model_dict, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print(f"成功加载模型权重从: {model_dict}")
        except Exception as e:
            print(f"警告: 加载模型时发生错误: {e}")
            print("将使用随机初始化的模型继续训练")
    
    model.to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # 创建优化器和调度器
    optimizer = create_optimizer(model, config)
    
    # 设置数据集和数据加载器
    train_dataset = UAVID_Images_Dataset_v1(
        config['train_path'],
        device,
        config['desired_shape'],
        config['n_condition'],
        config['n_prediction'],
        if_flow=config['if_flow'],
        samples_per_dir=config['samples_per_dir']
    )
    val_dataset = UAVID_Images_Dataset_v1(
        config['val_path'],
        device,
        config['desired_shape'],
        config['n_condition'],
        config['n_prediction'],
        if_flow=config['if_flow'],
        samples_per_dir=config['samples_per_dir']
    )
    train_loader, train_sampler = get_dataloader(train_dataset, config, rank, world_size, True)
    val_loader, val_sampler = get_dataloader(val_dataset, config, rank, world_size, False)

    # for imgs, flows, labels, label_filenames in train_loader:
    #     print(f"Batch imgs shape: {imgs.shape}")          # 期望 [4, 3, 128, 128]
    #     print(f"Batch flows shape: {flows.shape}")        # 期望 [4, 8, 3, 128, 128]
    #     break
    
    # 设置日志
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            filename=os.path.join(config['output_path'], 'training.log'),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    best_loss = float('inf')
    start_time = time.time()
    
    # 添加学习率调度器
    scheduler = get_lr_scheduler(optimizer, config, train_loader)
    
    # 添加EMA
    # ema = EMA(model, config['ema_decay']) #ema需要较多额外显存
    # ema.register()  # 注册初始参数
    
    # num_training_steps = len(train_loader) * config['n_epochs']
    # scheduler = create_scheduler(optimizer, config, num_training_steps)
    
    # 创建损失函数和其他指标
    criterion = torch.nn.MSELoss()
    ssim_metric = StructuralSimilarityIndexMeasure().to(device)
    scaler = GradScaler()
    
    # 初始化LPIPS损失
    # lpips_loss_fn = LPIPS_Loss(model=config.get('lpips_model', 'vgg')).to(device)
    lpips_loss_fn = None
    # 修改train_epoch函数
    def train_epoch(model, train_loader, optimizer, scaler, criterion, ssim_metric, lpips_loss_fn, config, device, epoch, sampler=None):
        model.train()
        if sampler:
            sampler.set_epoch(epoch)
        
        # 从config获取梯度累积步数
        accumulation_steps = config.get('accumulate_grad_batches', 1)  # 默认为1
        
        # 初始化损失统计字典
        losses = {
            'total_loss': 0,
            'l1': 0,
            'mse': 0,
            'ssim': 0,
            'gdl': 0,
            'temporal': 0,
            'perceptual': 0,
            'contrastive': 0,
            'lpips': 0  # 添加LPIPS损失
        }
        batch_count = 0
        
        for idx, (image, flows, label, _) in enumerate(train_loader):
            image = image.to(device)
            flows = flows.to(device)
            label = label.to(device)
            
            # 检查输入数据是否包含NaN
            if torch.isnan(image).any() or torch.isnan(flows).any() or torch.isnan(label).any():
                print(f"警告: 输入数据包含NaN，跳过批次 {idx}")
                continue
            
            with autocast():
                output = model(image, flows)
            
            # 检查模型输出是否包含NaN
            if output is None or torch.isnan(output).any():
                print(f"警告: 模型输出包含NaN，跳过批次 {idx}")
                continue
            
            x_flat = output.reshape(
                image.size(0) * config['n_prediction'],
                config['n_channels'],
                config['desired_shape'][0],
                config['desired_shape'][1]
            ).float()
            labels_flat = label.reshape(
                label.size(0) * config['n_prediction'],
                config['n_channels'],
                config['desired_shape'][0],
                config['desired_shape'][1]
            ).float()
            
            # 检查重塑后的数据
            if torch.isnan(x_flat).any() or torch.isnan(labels_flat).any():
                print(f"警告: 重塑后数据包含NaN，跳过批次 {idx}")
                continue
            
            # print(x_flat.shape,labels_flat.shape)
               # 计算各种损失
            l1_loss = L1_loss(x_flat, labels_flat)
            mse_loss = criterion(x_flat, labels_flat)
            ssim_loss = 1 - SSIM_loss(x_flat, labels_flat)
            gdl_loss = gradient_difference_loss(x_flat, labels_flat)
            temp_loss = temporal_consistency_loss(x_flat)
            perceptual_loss = vgg_perceptual_loss(x_flat, labels_flat)
            contrastive_loss = Contrastive_loss(x_flat, labels_flat)
            # lpips_loss = lpips_loss_fn(x_flat, labels_flat)  # 计算LPIPS损失
            
            # 检查各个损失值是否包含NaN
            loss_components = [l1_loss, mse_loss, ssim_loss, gdl_loss, temp_loss, perceptual_loss, contrastive_loss]
            loss_names = ['l1_loss', 'mse_loss', 'ssim_loss', 'gdl_loss', 'temp_loss', 'perceptual_loss', 'contrastive_loss']
            
            for loss_val, loss_name in zip(loss_components, loss_names):
                if torch.isnan(loss_val).any():
                    print(f"警告: {loss_name} 包含NaN，跳过批次 {idx}")
                    continue
            
            # 总损失
            loss = (
                config['weight_l1'] * l1_loss
                + config['weight_mse'] * mse_loss
                + config['weight_ssim'] * ssim_loss
                + config['weight_gdl'] * gdl_loss
                + config['weight_temporal'] * temp_loss
                + config['weight_perceptual'] * perceptual_loss
                + config['weight_contrastive'] * contrastive_loss
                # + config['weight_lpips'] * lpips_loss  # 添加LPIPS损失
            )
            
            # 检查总损失是否包含NaN
            if torch.isnan(loss).any():
                print(f"警告: 总损失包含NaN，跳过批次 {idx}")
                continue
                
            # 梯度累积
            loss = loss / accumulation_steps
            
            # 反向传播
            scaler.scale(loss).backward()
            
            # 梯度累积完成后更新参数
            if (idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                
                # 检查梯度是否包含NaN
                has_nan_grad = False
                for param in model.parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    print(f"警告: 梯度包含NaN，跳过参数更新")
                    optimizer.zero_grad()
                    continue
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('gradient_clip_val', 1.0))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            # 累计各项损失
            losses['total_loss'] += loss.item() * accumulation_steps  # 注意这里要乘回accumulation_steps
            losses['l1'] += l1_loss.item()
            losses['mse'] += mse_loss.item()
            losses['ssim'] += ssim_loss.item()
            losses['gdl'] += gdl_loss.item()
            losses['temporal'] += temp_loss.item() if isinstance(temp_loss, torch.Tensor) else temp_loss
            losses['perceptual'] += perceptual_loss.item() if isinstance(perceptual_loss, torch.Tensor) else perceptual_loss
            losses['contrastive'] += contrastive_loss.item()
            # losses['lpips'] += lpips_loss.item()  # 累计LPIPS损失
            
            batch_count += 1
        
        # 计算平均损失
        for key in losses.keys():
            losses[key] /= batch_count
        
        return losses
    
    # 训练循环
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(config['n_epochs']):
        if epoch < 30:
            # # 前期主要关注重建质量
            config['weight_temporal'] = 0.0
            config['weight_perceptual'] = 0.0
            config['weight_contrastive'] = 0.0
            config['weight_ssim'] = 0.0
            config['weight_gdl'] = 0.0
            config['weight_mse'] = 0.0
            config['weight_l1'] = 1.0
            config['weight_lpips'] = 0.0  # 设置LPIPS权重
        elif epoch < 60:
            # 逐步增加时序和感知损失
            config['weight_temporal'] = 0.3
            # config['weight_perceptual'] = 0.0
            # config['weight_contrastive'] = 0.0
            # # config['weight_ssim'] = 0.1
            config['weight_gdl'] = 0.3
            # config['weight_mse'] = 1.0
            config['weight_l1'] = 1.0
            # config['weight_lpips'] = 0.1  # 设置LPIPS权重
        else:
            # 完整权重配置
            config['weight_temporal'] = 0.3
            config['weight_perceptual'] = 0.0005
            config['weight_contrastive'] = 0.0005
            # # config['weight_ssim'] = 0.1
            config['weight_gdl'] = 0.3
            # config['weight_mse'] = 1.0
            config['weight_l1'] = 1.0
            # config['weight_lpips'] = 0.1  # 设置LPIPS权重
        
        train_losses = train_epoch(
            model, train_loader, optimizer, scaler,
            criterion, ssim_metric, lpips_loss_fn, config, device, epoch,
            train_sampler
        )
        # ema.update()  # 更新EMA参数
        
        # 验证和保存模型的逻辑保持不变
        if epoch % config['interval_val'] == 0:
            # ema.apply_shadow()  # 使用EMA参数进行验证
            metrics = validate(model, val_loader, criterion, ssim_metric, lpips_loss_fn, config, device)
            # ema.restore()  # 恢复原始参数
            
            if rank == 0:  # 只在主进程中保存模型和记录日志
                is_best = metrics['loss'] < best_loss
                if is_best:
                    best_loss = metrics['loss']
                    save_checkpoint(model, optimizer, epoch, metrics['loss'],
                                 config['output_path'], is_best=True)
                
                # 记录训练信息
                # current_lr = optimizer.param_groups[0]['lr']
                elapsed_time = time.time() - start_time
                remaining_time = elapsed_time / (epoch + 1) * (config['n_epochs'] - epoch - 1)
                log_message = (
                    f"\nEpoch: {epoch}, lr: {optimizer.param_groups[0]['lr']:.6f}\n"
                    f"Train Losses:\n"
                    f"  - Total: {train_losses['total_loss']:.5f}\n"
                    f"  - L1: {train_losses['l1']*config['weight_l1']:.5f}\n"
                    f"  - MSE: {train_losses['mse']*config['weight_mse']:.5f}\n"
                    f"  - SSIM: {train_losses['ssim']*config['weight_ssim']:.5f}\n"
                    f"  - GDL: {train_losses['gdl']*config['weight_gdl']:.5f}\n"
                    f"  - Temporal: {train_losses['temporal']*config['weight_temporal']:.5f}\n"
                    f"  - Perceptual: {train_losses['perceptual']*config['weight_perceptual']:.5f}\n"
                    f"  - Contrastive: {train_losses['contrastive']*config['weight_contrastive']:.5f}\n"
                    f"  - LPIPS: {train_losses['lpips']*config['weight_lpips']:.5f}\n"  # 记录LPIPS损失
                    f"Validation Metrics:\n"
                    f"  - Loss: {metrics['loss']:.5f}\n"
                    f"  - PSNR: {metrics['psnr']:.4f}\n"
                    f"  - SSIM: {metrics['ssim']:.4f}\n"
                    f"  - Rel: {metrics['abs_rel']:.4f}\n"
                    f"Time:\n"
                    f"  - Elapsed: {convert_seconds_to_dhms(elapsed_time)}hms\n"
                    f"  - Remaining: {convert_seconds_to_dhms(remaining_time)}hms"
                )
                print(log_message)
                logging.info(log_message)
                
                if config['if_save_interval_pkl'] and epoch % config['interval_save_pkl'] == 0:
                    save_checkpoint(model, optimizer, epoch, train_losses['total_loss'],
                                 config['output_path'], is_best=False)
    
    cleanup()


# 检查EMA实现
class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()
                
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
                
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# 检查学习率调度器
def get_lr_scheduler(optimizer, config, train_loader):
    if config.get('scheduler_type', 'cosine') == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config['n_epochs'] * len(train_loader),
            eta_min=config.get('lr_min', 1e-6)
        )
    else:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config['lr_init'],
            epochs=config['n_epochs'],
            steps_per_epoch=len(train_loader),
            pct_start=0.1
        )
    return scheduler

if __name__ == "__main__":
    # 更新配置
    config = config_512
    config.update({
        'n_epochs': 100,
        'train_path': None,
        'val_path': None,
        'output_path': None,
        'interval_val': 1,
        'if_save_interval_pkl': True,
        'interval_save_pkl': 10,
        'accumulate_grad_batches': 2,  # 梯度累积步数
        'use_checkpoint': False,        # 使用梯度检查点
        'gradient_clip_val': 0.5,      # 降低梯度裁剪阈值
        'ema_decay': 0.9999,          # EMA衰减率
        'warmup_epochs': 5,            # 预热轮数
        'lr_min': 1e-6,               # 最小学习率
        'lr_init': 1e-4,              # 降低初始学习率
    })
    
    # 创建输出目录
    os.makedirs(config['output_path'], exist_ok=True)
    
    # 获取可用的GPU数量
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs!")
    
    # 启动分布式训练
    mp.spawn(
        train,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )