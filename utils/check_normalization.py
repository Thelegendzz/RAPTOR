import sys
import os

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取父目录（即包含 CustomDatasets.py 的目录）
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

# 将父目录添加到 sys.path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
from torch.utils.data import DataLoader
from CustomDatasets import (MMNISTDatasetCustominout,UAVID_Images_Dataset_v1)
from configurations_v0 import config_64
from train_gpus_v3 import get_dataloader

os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'

def check_normalization(dataloader, num_batches=5):
    """
    检查 DataLoader 中图片是否经过归一化。

    参数:
    - dataloader (DataLoader): PyTorch DataLoader 对象。
    - num_batches (int): 检查的批次数量（默认检查前5个批次）。
    """
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        # 检查批次中包含的元素数量
        num_elements = len(batch)
        print(f"批次 {batch_idx + 1} 包含 {num_elements} 个元素")

        for i, element in enumerate(batch):
            if isinstance(element, torch.Tensor):
                print(f"  元素 {i + 1} 的维度: {element.shape}")
            else:
                print(f"  元素 {i + 1} 的类型: {type(element)}")

        images = batch[batch_idx]
        # 移动到 CPU 并转换为 NumPy 数组
        images_np = images.cpu().numpy()

        # 计算每个批次的最小值、最大值、均值和标准差
        min_val = images_np.min()
        max_val = images_np.max()
        mean_val = images_np.mean()
        std_val = images_np.std()

        print(f"批次 {batch_idx + 1}:")
        print(f"  最小像素值: {min_val:.4f}")
        print(f"  最大像素值: {max_val:.4f}")
        print(f"  均值: {mean_val:.4f}")
        print(f"  标准差: {std_val:.4f}\n")

def main():
    world_size = torch.cuda.device_count()
    rank = 0
    device = torch.device(f'cuda:{rank}')
    print(f"Using device: {device}")
    config = config_64
    # config['train_path'] = '/datasets_active/MMNIST/train'
    config['train_path'] = '/datasets_active/UAVID-images/uavid_train_sample'

    dataset = UAVID_Images_Dataset_v1(
        config['train_path'],
        device,
        config['desired_shape'],
        config['n_condition'],
        config['n_prediction'],
        if_flow=config['if_flow'],
        samples_per_dir=config['samples_per_dir']
    )

    train_loader, train_sampler = get_dataloader(dataset, config, rank, world_size, True)
    check_normalization(train_loader, num_batches=5)

if __name__ == "__main__":
        main()