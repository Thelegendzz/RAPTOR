import imageio
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
from datetime import datetime

import numpy as np  # For array operations
import torch
from networkx.algorithms import flow
from torch.utils.data import DataLoader, random_split
from utils import utils
import torch.nn as nn


from datasets.CustomDatasets import UAVID_Images_Dataset_v1 as CustomDataset # v0 static load, v1 dynamic load
from configs.configurations import config_64, config_128, config_256, config_512, config_1024
from utils.metrics import calculate_psnr, calculate_abs_rel, L1_loss, SSIM_loss, vgg_perceptual_loss, gradient_difference_loss, temporal_consistency_loss, Contrastive_loss, LPIPS_Loss

from torch.utils.data import DataLoader
# from prefetch_generator import BackgroundGenerator
# class DataLoaderX(DataLoader):
#     def __iter__(self):
#         return BackgroundGenerator(super().__iter__())

from utils.metrics import calculate_psnr, calculate_ssim, calculate_abs_rel
from utils.utils import convert_seconds_to_dhms
from models.SIMVP import PredictionRWKV as EWKV
from torchvision import models
from torchmetrics import StructuralSimilarityIndexMeasure
import torch.nn.functional as F
from torch.cuda.amp import autocast

interval_val = 1
if_save_interval_pkl = True
interval_save_pkl = 100

train_paths =[
                None
             ] #改成你的数据集位置

val_paths =[
                None
            ] #改成你的数据集位置

output_paths =[
                None
]
config_subpath = ['config_64', 'config_128', 'config_256', 'config_512', 'config_1024']
dataset_type=10
train_path=train_paths[dataset_type]
val_path=val_paths[dataset_type]
output_path=output_paths[dataset_type]
config_dict = config_512
output_path = os.path.join(output_path, 'SIMVP','config_512') #PredictionRWKV, EWKV_v0
os.makedirs(output_path, exist_ok=True)
best_model_name_list = ['best_model.pth','best_model.pth','best_model.pth','best_model.pth','best_model.pth','best_model.pth','best_model.pth','best_model.pth','best_model.pth','best_model.pth','best_model.pth']#'checkpoint_epoch_5.pth'
normal_model_name_list = ['_model.pth','_model.pth','_model.pth','_model.pth','_model.pth','_model.pth','_model.pth','_model.pth','_model.pth','_model.pth','_model.pth']
dataset_subpath_list = ['','','','','','','','','','','']
device = torch.device('cuda')
utils.set_seeds(123)

model_sub_name = ''

pram_from_dict = True
model_dict = os.path.join(output_path, best_model_name_list[dataset_type])+model_sub_name  # 如果加载模型权重的路径名称
best_model_name = best_model_name_list[dataset_type]+model_sub_name
normal_model_name = normal_model_name_list[dataset_type]+model_sub_name
dataset_subpath = dataset_subpath_list[dataset_type]
tpath = os.path.join(train_path, dataset_subpath)
vpath = os.path.join(val_path, dataset_subpath)

config_dict["seq_len"] = config_dict["n_condition"] - config_dict["if_flow"]
config_dict["n_pixels"] = 3 * config_dict["desired_shape"][0] * config_dict["desired_shape"][1]
config_dict['batch_size'] = 2  # 显著减小批次大小
config_dict['num_workers'] = 2  # 减少工作线程数
print(config_dict)

def get_device_ids():
    return list(range(torch.cuda.device_count()))
# config_dict.update({
#     'weight_mse': 1.0,
#     'weight_temporal': 0.3, 
#     'weight_ssim': 0.05,
#     'weight_gdl': 0.05,
#     'weight_perceptual': 0.001,
# })
config_dict.update({
    'device_ids': get_device_ids(),
    'weight_temporal': 0.3,
    'weight_perceptual': 0.0005,
    'weight_contrastive': 0.0005,
    'weight_ssim': 0.1,
    'weight_gdl': 0.3,
    'weight_mse': 1.0,
    'weight_l1': 1.0,
    'weight_lpips': 0.1  # 设置LPIPS权重
})

def initialize_model(config):
    model = EWKV(config)
    if config['parallel_mode'] == 'ddl' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=config['device_ids'])
    elif config['parallel_mode'] == 'model_parallel':
        # 示例: 将模型的不同部分分配到不同的GPU
        device0 = torch.device(f'cuda:{config["device_ids"][0]}')
        device1 = torch.device(f'cuda:{config["device_ids"][1]}')
        model.part1.to(device0)
        model.part2.to(device1)
    return model

# def vgg_perceptual_loss(pred, target, vgg_model):
#     pred_features = vgg_model(pred)
#     target_features = vgg_model(target)
#     return F.mse_loss(pred_features, target_features)

def evaluate_batch(model, batch, config, device, vgg=None):
    image, flows, label = [x.to(device) for x in batch[:31]]
    with torch.no_grad():
        output = model(image, flows)
        
        x_flat = output.reshape(
            image.size(0) * config['n_prediction'],
            config['n_channels'],
            config['desired_shape'][0],
            config['desired_shape'][1]
        )
        labels_flat = label.reshape(
            label.size(0) * config['n_prediction'],
            config['n_channels'],
            config['desired_shape'][0],
            config['desired_shape'][1]
        )
        
        lpips_loss_fn = LPIPS_Loss(model=config.get('lpips_model', 'vgg')).to(device)

        # 计算各种损失
        l1_loss = L1_loss(x_flat, labels_flat)
        mse_loss = F.mse_loss(x_flat, labels_flat)
        ssim_value = SSIM_loss(x_flat, labels_flat)
        ssim_loss = 1 - ssim_value
        gdl_loss = gradient_difference_loss(x_flat, labels_flat)
        temp_loss = temporal_consistency_loss(output)
        contrastive_loss = Contrastive_loss(x_flat, labels_flat)
        lpips_loss = lpips_loss_fn(x_flat, labels_flat)  # 计算LPIPS损失
        
        perceptual_loss = 0
        if vgg is not None:
            perceptual_loss = vgg_perceptual_loss(x_flat, labels_flat)
            
        # 计算评估指标
        psnr = calculate_psnr(output, label)
        rel = calculate_abs_rel(output, label)
        
        metrics = {
            'l1': l1_loss.item(),
            'mse': mse_loss.item(),
            'ssim': ssim_loss.item(),
            'gdl': gdl_loss.item(),
            'temporal': temp_loss.item(),
            'contrastive': contrastive_loss.item(),
            'lpips': lpips_loss.item(),
            'perceptual': perceptual_loss if isinstance(perceptual_loss, float) else perceptual_loss.item(),
            'psnr': psnr.item(),
            'abs_rel': rel.item()
        }
        
        return output, metrics

def test_model(model, dataloader, config, device):
    model.eval()

        
    # 初始化VGG模型(如果需要)
    vgg = None
    if config.get('weight_perceptual', 0) > 0:
        vgg = models.vgg16(pretrained=True).features[:16].eval().to(device)
    
    all_metrics = []
    for batch in dataloader:
        output, metrics = evaluate_batch(model, batch, config, device, vgg)
        all_metrics.append(metrics)
        
        # 保存输出视频
        # save_video_outputs(output, batch[-1], config['output_path'])
    
    # 计算平均指标
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
    
    return avg_metrics


if __name__ == '__main__':
    # 设置内存分配器配置
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = (
        'max_split_size_mb:64,'
        'garbage_collection_threshold:0.8'
    )
    
    torch.cuda.empty_cache()  # 开始前清理显存

    # Load the full dataset
    # full_dataset = CustomDataset(tpath, device, config_dict['desired_shape'], config_dict['n_condition'], if_flow=config_dict['if_flow'])
    dataset_val = CustomDataset(
        tpath, 
        device,  
        config_dict['desired_shape'],
        config_dict['n_condition'],
        config_dict['n_prediction'],
        if_flow=config_dict['if_flow'],
        samples_per_dir=config_dict['samples_per_dir']
    )

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=config_dict['batch_size'],
        num_workers=config_dict['num_workers'],
        shuffle=False,  # 验证集通常不需要打乱
        pin_memory=True,  # 启用固定内存
        prefetch_factor=2,  # 根据需要调整
        persistent_workers=True  # 保持 workers 活动（如果 PyTorch 版本支持）
    )

    dataloaders = {}
    dataloaders['val'] = dataloader_val
    # print(dataloaders.keys())
    model = initialize_model(config_dict)

    # if config['if_load_model'] == True:
    # 模型加载部分的错误处理
    try:
        checkpoint = torch.load(model_dict, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    except FileNotFoundError:
        print(f"错误: 找不到模型文件 '{model_dict}'")
        print(f"请检查模型文件路径是否正确")
        exit(1)
    except KeyError as e:
        print(f"错误: 模型文件格式不正确,缺少关键数据 '{e}'")
        print("请确保模型文件包含 'model_state_dict' 字段")
        exit(1)
    except RuntimeError as e:
        print(f"错误: 加载模型权重失败")
        print(f"详细信息: {e}")
        print("请检查:")
        print("1. 模型结构是否与保存时一致")
        print("2. 是否使用了正确的模型权重文件")
        exit(1)
    except Exception as e:
        print(f"加载模型时发生未知错误: {e}")
        exit(1)

    print(f"成功加载模型权重从: {model_dict}")

    print(torch.cuda.is_available())
    model = model.to(device)

    # initialize lists to keep track of the losses
    losses, train_losses, eval_losses = [], [], []

    # move the model to the GPU where the computations will be performed.
    # model = model.to(device)
    # 计算模型的参数数量
    total_params = sum(p.numel() for p in model.parameters())
    total_params_million = total_params / 1_000_000  # 将参数数量转换为以百万为单位
    print(f"Model Parameters: {total_params_million:.2f}M")

    # create progress bars for visualization during training
    prog_bar, loss_bar = utils.get_prog_bars(config_dict['n_epochs'])

    # the main training loop. we run for a certain number of epochs.
    start_time = time.time()
    tmp_val_loss = 5

    model.eval()
    
    # 初始化VGG模型(如果需要)
    # vgg = None
    if config_dict.get('weight_perceptual', 0) > 0:
        vgg = models.vgg16(pretrained=True).features[:16].eval().to(device)
    
    # 获取当前时间
    now = datetime.now()
    current_time = now.strftime('%H:%M:%S')
    start_time = time.time()
    print("当前时间:", current_time)
    
    all_metrics = []
    width, height = config_dict['desired_shape']
    
    for image, flows, label, filenames in dataloaders['val']:
        with torch.no_grad():
            image, flows, label = image.to(device), flows.to(device), label.to(device)
            reconstructed = model(image, flows)
            
            if reconstructed is not None:
                # 处理光流(如果需要)
                if config_dict['if_flow']:
                    print('flow true')
                    img_expanded = image[:, None, :, :, :]
                    reconstructed = reconstructed + img_expanded
                
                # 保存视频输出
                for i in range(len(reconstructed)):
                    # 保存预测结果
                    frames = [reconstructed[i][j].permute(1, 2, 0).cpu().numpy() for j in range(reconstructed.shape[1])]
                    frames = [np.clip(np.round(frame * 255), 0, 255).astype(np.uint8) for frame in frames]
                    
                    filename_without_extension = os.path.basename(filenames[i].replace('.jpg', ''))
                    print(filename_without_extension)
                    mp4_filename = os.path.join(output_path, 'output/' + filename_without_extension + ".mp4")
                    mp4_dir = os.path.dirname(mp4_filename)
                    os.makedirs(mp4_dir, exist_ok=True)
                    writer = imageio.get_writer(mp4_filename, fps=5, format='FFMPEG', mode='I')
                    for frame in frames:
                        writer.append_data(frame)
                    writer.close()

                # 保存光流
                for i in range(len(flows)):
                    frames = [flows[i][j].permute(1, 2, 0).cpu().numpy() for j in range(flows.shape[1])]
                    frames = [np.clip(np.round(frame * 255), 0, 255).astype(np.uint8) for frame in frames]
                    
                    filename_without_extension = filenames[i].replace('.jpg', '')
                    mp4_filename = os.path.join(output_path, 'input/' + filename_without_extension + ".mp4")
                    mp4_dir = os.path.dirname(mp4_filename)
                    os.makedirs(mp4_dir, exist_ok=True)
                    writer = imageio.get_writer(mp4_filename, fps=5, format='FFMPEG', mode='I')
                    for frame in frames:
                        writer.append_data(frame)
                    writer.close()

                # 保存标签
                for i in range(len(label)):
                    frames = [label[i][j].permute(1, 2, 0).cpu().numpy() for j in range(label.shape[1])]
                    frames = [np.clip(np.round(frame * 255), 0, 255).astype(np.uint8) for frame in frames]
                    
                    filename_without_extension = filenames[i].replace('.jpg', '')
                    mp4_filename = os.path.join(output_path, 'labels/' + filename_without_extension + ".mp4")
                    mp4_dir = os.path.dirname(mp4_filename)
                    os.makedirs(mp4_dir, exist_ok=True)
                    writer = imageio.get_writer(mp4_filename, fps=5, format='FFMPEG', mode='I')
                    for frame in frames:
                        writer.append_data(frame)
                    writer.close()

                # 计算评估指标
                x_flat = reconstructed.reshape(
                    image.size(0) * config_dict['n_prediction'],
                    config_dict['n_channels'],
                    config_dict['desired_shape'][0],
                    config_dict['desired_shape'][1]
                )
                labels_flat = label.reshape(
                    label.size(0) * config_dict['n_prediction'],
                    config_dict['n_channels'],
                    config_dict['desired_shape'][0],
                    config_dict['desired_shape'][1]
                )
                
                # 计算各种损失和指标
                l1_loss = L1_loss(x_flat, labels_flat)
                mse_loss = F.mse_loss(x_flat, labels_flat)
                ssim_value = SSIM_loss(x_flat, labels_flat)
                ssim_loss = 1 - ssim_value

                gdl_loss = gradient_difference_loss(x_flat, labels_flat)

                temp_loss = temporal_consistency_loss(x_flat)
                
                # 感知损失
                perceptual_loss_fn = vgg_perceptual_loss()
                perceptual_loss = perceptual_loss_fn(x_flat, labels_flat) 

                contrastive_loss = Contrastive_loss(x_flat, labels_flat)

                lpips_loss_fn = LPIPS_Loss(model=config_dict.get('lpips_model', 'vgg')).to(device)
                lpips_loss = lpips_loss_fn(x_flat, labels_flat)  # 计算LPIPS损失
                # 计算PSNR和相对误差
                psnr_value = calculate_psnr(reconstructed, label)
                rel_value = calculate_abs_rel(reconstructed, label)
                
                metrics = {
                    'l1': l1_loss.item(),
                    'mse': mse_loss.item(),
                    'ssim': ssim_value.item(),
                    'gdl': gdl_loss.item(),
                    'temporal': temp_loss.item(),
                    'perceptual': perceptual_loss if isinstance(perceptual_loss, float) else perceptual_loss.item(),
                    'contrastive': contrastive_loss.item(),
                    'lpips': lpips_loss.item(),
                    'psnr': psnr_value.item(),
                    'abs_rel': rel_value.item()
                }
                all_metrics.append(metrics)
                
                # 处理光流(如果需要)
                if config_dict['if_flow']:
                    img_expanded = image[:, None, :, :, :]
                    reconstructed = reconstructed + img_expanded
                
                # [保存视频的代码保持不变...]
                
                print('rate：', len(all_metrics)/len(dataloaders['val']))
    
    # 计算平均指标
    if all_metrics:
        avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
    else:
        avg_metrics = {
            'l1': 0, 'mse': 0, 'ssim': 0, 'gdl': 0, 'temporal': 0,
            'perceptual': 0, 'contrastive': 0, 'lpips': 0, 'psnr': 0, 'abs_rel': 0
        }
    
    elapsed_time = time.time() - start_time
    log_message = (
        f"\nTest Metrics:\n"
        f"  - L1 Loss: {avg_metrics['l1']:.5f}\n"
        f"  - MSE Loss: {avg_metrics['mse']:.5f}\n"
        f"  - SSIM: {avg_metrics['ssim']:.4f}\n"
        f"  - GDL Loss: {avg_metrics['gdl']:.5f}\n"
        f"  - Temporal Loss: {avg_metrics['temporal']:.5f}\n"
        f"  - Perceptual Loss: {avg_metrics['perceptual']:.5f}\n"
        f"  - Contrastive Loss: {avg_metrics['contrastive']:.5f}\n"
        f"  - LPIPS Loss: {avg_metrics['lpips']:.5f}\n"
        f"  - PSNR: {avg_metrics['psnr']:.4f}\n"
        f"  - Abs Rel: {avg_metrics['abs_rel']:.4f}\n"
        f"Time: {convert_seconds_to_dhms(elapsed_time)}hms"
    )
    print(log_message)

