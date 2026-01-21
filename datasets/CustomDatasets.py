import cv2
import os
import random
import re
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache, partial

class MMNISTDatasetCustominout(Dataset):
    def __init__(self, folder_path, device, desired_shape, n_condition, n_prediction, if_flow=False, samples_per_dir=10):
        super().__init__()
        self.device = device
        self.desired_shape = desired_shape
        self.transform = transforms.Compose([
            transforms.Resize(desired_shape),
            transforms.ToTensor(),
        ])
        self.samples_per_dir = samples_per_dir
        self.images, self.flows, self.labels, self.filenames = self.read_and_sort_images(folder_path, n_condition, n_prediction, if_flow=if_flow)

    def read_and_sort_images(self, folder_path, n_condition, n_prediction, if_flow=False):
        images = []
        labels = []
        flows = []
        filenames = []
        for dir in os.listdir(folder_path):
            image_files = [f for f in os.listdir(os.path.join(folder_path, dir)) if f.endswith('.jpg')]
            image_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
            # print(image_files)

            # Random sampling within the available range
            max_index = len(image_files) - n_condition - n_prediction + 1
            if max_index < 1:
                continue  # Skip directories with not enough images
            sampled_indices = random.sample(range(max_index), min(self.samples_per_dir, max_index))

            for i in sampled_indices:
                img_filename = image_files[i + n_condition - 1]
                img_path = os.path.join(os.path.join(folder_path, dir), img_filename)
                img = Image.open(img_path).convert('RGB')
                
                img = self.transform(img)
                images.append(img)

                if if_flow:
                    flows_tmp = torch.zeros((n_condition - 1,) + img.shape)
                    labels_tmp = torch.zeros((n_prediction,) + img.shape)
                    for j in range(n_condition - 1):
                        img_path_1 = os.path.join(os.path.join(folder_path, dir), image_files[i + j])
                        img_path_2 = os.path.join(os.path.join(folder_path, dir), image_files[i + j + 1])

                        img_1 = Image.open(img_path_1).convert('RGB')
                        img_2 = Image.open(img_path_2).convert('RGB')
                        img_1 = self.transform(img_1)
                        img_2 = self.transform(img_2)
                        flows_tmp[j] = img_2 - img_1

                    for k in range(n_prediction):
                        label_filename = image_files[i + n_condition + k]
                        label_path = os.path.join(os.path.join(folder_path, dir), label_filename)
                        label = Image.open(label_path).convert('RGB')
                        label = self.transform(label)
                        labels_tmp[k] = label
                else:
                    flows_tmp = torch.zeros((n_condition,) + img.shape)
                    labels_tmp = torch.zeros((n_prediction,) + img.shape)
                    for j in range(n_condition):
                        img_path = os.path.join(os.path.join(folder_path, dir), image_files[i + j])
                        img = Image.open(img_path).convert('RGB')
                        img = self.transform(img)
                        flows_tmp[j] = img
                    for k in range(n_prediction):
                        label_filename = image_files[i + n_condition + k]
                        label_path = os.path.join(os.path.join(folder_path, dir), label_filename)
                        label = Image.open(label_path).convert('RGB')
                        label = self.transform(label)
                        labels_tmp[k] = label

                flows.append(flows_tmp)
                labels.append(labels_tmp)
                label_filename = os.path.join(dir, f"{dir}_{i}-{n_condition}_{n_prediction}")
                filenames.append(label_filename)

        return images, flows, labels, filenames

    def __len__(self):
        return len(self.flows)

    def __getitem__(self, i):
        img = self.images[i]
        flow = self.flows[i]
        label = self.labels[i]
        filename = self.filenames[i]
        return img, flow, label, filename
    
class KTH_Dataset(Dataset):  # dynamic load
    def __init__(self, folder_path, device, desired_shape, n_condition, n_prediction, if_flow=False, samples_per_dir=1):
        super().__init__()
        self.folder_path = folder_path
        self.device = device
        self.desired_shape = desired_shape
        self.samples_per_dir = samples_per_dir
        self.n_condition = n_condition
        self.n_prediction = n_prediction
        self.if_flow = if_flow
        self.transform = transforms.Compose([
            transforms.Resize(desired_shape),
            transforms.ToTensor(),
        ])
        # Store metadata instead of actual images
        self.data_info, self.action_to_label = self._prepare_data()

    def _prepare_data(self):
        """
        准备数据索引，收集所有样本的信息，并创建动作类别到标签的映射。

        返回:
            tuple: (data_info, action_to_label) 其中
                - data_info (list): 包含每个样本信息的字典列表。
                - action_to_label (dict): 动作类别到标签的映射字典。
        """
        data_info = []
        actions = sorted([d for d in os.listdir(self.folder_path) if os.path.isdir(os.path.join(self.folder_path, d))])
        action_to_label = {action: idx for idx, action in enumerate(actions)}

        for action in actions:
            action_dir = os.path.join(self.folder_path, action)
            if not os.path.isdir(action_dir):
                print(f"警告: 动作类别目录不存在: {action_dir}")
                continue

            video_folders = [d for d in os.listdir(action_dir) if os.path.isdir(os.path.join(action_dir, d))]
            for video_folder in video_folders:
                video_path = os.path.join(action_dir, video_folder)
                image_files = sorted([
                    f for f in os.listdir(video_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ], key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0)

                max_index = len(image_files) - self.n_condition - self.n_prediction + 1
                if max_index < 1:
                    print(f"警告: 视频 '{video_folder}' 帧数不足，跳过。")
                    continue

                sampled_indices = random.sample(range(max_index), min(self.samples_per_dir, max_index))  # 每个视频采样指定数量的样本
                for idx in sampled_indices:
                    data_entry = {
                        'dir': video_path,
                        'image_files': image_files,
                        'start_idx': idx,
                        'label': action,  # 使用动作类别作为标签
                        'label_filename': f"{video_folder}_{idx}"  # 添加 label_filename
                    }
                    data_info.append(data_entry)

        print(f"数据集划分中共加载了 {len(data_info)} 个样本。")
        return data_info, action_to_label


    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        data_entry = self.data_info[idx]
        dir_path = data_entry['dir']
        image_files = data_entry['image_files']
        start_idx = data_entry['start_idx']
        label_filename = data_entry['label_filename']

        # Load images dynamically
        if self.if_flow:
            flows = []
            for j in range(self.n_condition - 1):
                img_path_1 = os.path.join(dir_path, image_files[start_idx + j])
                img_path_2 = os.path.join(dir_path, image_files[start_idx + j + 1])

                img_1 = Image.open(img_path_1).convert('RGB')
                img_2 = Image.open(img_path_2).convert('RGB')
                img_1 = self.transform(img_1)
                img_2 = self.transform(img_2)
                flow = img_2 - img_1  # Simplified optical flow
                flows.append(flow)

            flows = torch.stack(flows)

            # Load labels (n_prediction frames)
            labels = []
            for k in range(self.n_prediction):
                label_path = os.path.join(dir_path, image_files[start_idx + self.n_condition + k])
                label = Image.open(label_path).convert('RGB')
                label = self.transform(label)
                labels.append(label)
            labels = torch.stack(labels)

            # Use the last image as the img
            img_path = os.path.join(dir_path, image_files[start_idx + self.n_condition - 1])
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)

            # return img.to(self.device), flows.to(self.device), labels.to(self.device), label_filename
            return img, flows, labels, label_filename

        else:
            # For non-flow, load n_condition frames
            flows = []
            for j in range(self.n_condition):
                img_path = os.path.join(dir_path, image_files[start_idx + j])
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
                flows.append(img)
            flows = torch.stack(flows)

            # Load labels (n_prediction frames)
            labels = []
            for k in range(self.n_prediction):
                label_path = os.path.join(dir_path, image_files[start_idx + self.n_condition + k])
                label = Image.open(label_path).convert('RGB')
                label = self.transform(label)
                labels.append(label)
            labels = torch.stack(labels)

            # Use the last image of flows as the img
            img = flows[-1]

            # return img.to(self.device), flows.to(self.device), labels.to(self.device), label_filename
            return img, flows, labels, label_filename
        
class KITTI_Dataset(Dataset):  # dynamic load
    """
    KITTI数据集加载器，单目模式（同时使用image_02和image_03作为独立样本）
    
    数据组织方式（类似KTH数据集）：
    - flows维度为[n_condition, C, H, W]（非光流模式）或[n_condition-1, C, H, W]（光流模式）
    - labels维度为[n_prediction, C, H, W]
    - 从image_02和image_03目录分别加载数据，每个相机的数据都作为独立的样本
    
    示例：如果n_condition=4，n_prediction=2，每个样本的数据结构为：
    - flows[0:4] = 某个相机的条件帧 (t0, t1, t2, t3)
    - labels[0:2] = 同一相机的预测帧 (t4, t5)
    
    数据集会包含两倍的样本数量（每个场景的每个相机都产生独立的样本）
    """
    def __init__(self, folder_path, device, desired_shape, n_condition, n_prediction, if_flow=False, samples_per_dir=1):
        super().__init__()
        self.folder_path = folder_path
        self.device = device
        self.desired_shape = desired_shape
        self.samples_per_dir = samples_per_dir
        self.n_condition = n_condition
        self.n_prediction = n_prediction
        self.if_flow = if_flow
        self.transform = transforms.Compose([
            transforms.Resize(desired_shape),
            transforms.ToTensor(),
        ])
        # Store metadata instead of actual images
        self.data_info = self._prepare_data()

    def _prepare_data(self):
        """
        准备数据索引，收集所有样本的信息（使用image_02和image_03）
        """
        data_info = []
        
        # 获取所有场景文件夹
        scenes = sorted([d for d in os.listdir(self.folder_path) 
                        if os.path.isdir(os.path.join(self.folder_path, d))])
        
        for scene in scenes:
            scene_path = os.path.join(self.folder_path, scene)
            print(f"处理场景: {scene}")
            
            # 处理image_02和image_03
            for camera in ['image_02', 'image_03']:
                camera_path = os.path.join(scene_path, camera)
                
                if not os.path.exists(camera_path):
                    print(f"  警告: 场景 {scene} 缺少{camera}目录，跳过")
                    continue
                
                image_files = sorted([
                    f for f in os.listdir(camera_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ], key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0)
                
                if len(image_files) == 0:
                    print(f"  警告: 场景 {scene} {camera} 没有图像文件")
                    continue
                
                # 检查是否有足够的帧
                max_index = len(image_files) - self.n_condition - self.n_prediction + 1
                if max_index < 1:
                    print(f"  警告: 场景 {scene} {camera} 帧数不足，跳过")
                    continue
                
                # 随机采样
                sampled_indices = random.sample(range(max_index), 
                                              min(self.samples_per_dir, max_index))
                
                for idx in sampled_indices:
                    data_entry = {
                        'scene': scene,
                        'camera': camera,
                        'dir': camera_path,
                        'image_files': image_files,
                        'start_idx': idx,
                        'label_filename': f"{scene}_{camera}_{idx}"
                    }
                    data_info.append(data_entry)
        
        print(f"KITTI数据集共加载了 {len(data_info)} 个样本")
        return data_info


    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        data_entry = self.data_info[idx]
        start_idx = data_entry['start_idx']
        label_filename = data_entry['label_filename']

        # 单目模式（只使用image_02）
        dir_path = data_entry['dir']
        image_files = data_entry['image_files']
        return self._load_single_camera_data(dir_path, image_files, start_idx, label_filename)


    def _load_single_camera_data(self, image_dir, image_files, start_idx, label_filename):
        """加载单相机数据（兼容原有逻辑）"""
        # 这里保持原有的单相机逻辑
        if self.if_flow:
            flows = []
            for j in range(self.n_condition - 1):
                img_path_1 = os.path.join(image_dir, image_files[start_idx + j])
                img_path_2 = os.path.join(image_dir, image_files[start_idx + j + 1])
                
                img_1 = Image.open(img_path_1).convert('RGB')
                img_2 = Image.open(img_path_2).convert('RGB')
                img_1 = self.transform(img_1)
                img_2 = self.transform(img_2)
                flow = img_2 - img_1
                flows.append(flow)
            
            flows = torch.stack(flows)
            
            # 加载预测标签
            labels = []
            for k in range(self.n_prediction):
                label_path = os.path.join(image_dir, image_files[start_idx + self.n_condition + k])
                label = Image.open(label_path).convert('RGB')
                label = self.transform(label)
                labels.append(label)
            labels = torch.stack(labels)
            
            # 当前图像
            img_path = os.path.join(image_dir, image_files[start_idx + self.n_condition - 1])
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            
            return img, flows, labels, label_filename
        else:
            # 非光流模式的单相机处理
            flows = []
            for j in range(self.n_condition):
                img_path = os.path.join(image_dir, image_files[start_idx + j])
                # print(img_path)
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
                flows.append(img)
            flows = torch.stack(flows)
            
            labels = []
            for k in range(self.n_prediction):
                label_path = os.path.join(image_dir, image_files[start_idx + self.n_condition + k])
                label = Image.open(label_path).convert('RGB')
                label = self.transform(label)
                labels.append(label)
            labels = torch.stack(labels)
            
            img = flows[-1]
            
            return img, flows, labels, label_filename



    def get_scene_info(self):
        """
        获取数据集中的场景信息
        
        Returns:
            dict: 场景统计信息
        """
        scenes = {}
        for data_entry in self.data_info:
            scene = data_entry['scene']
            if scene not in scenes:
                scenes[scene] = {'samples': 0}
            scenes[scene]['samples'] += 1
        
        return scenes

class AirMot_Dataset_v0(Dataset):  # static load
    def __init__(self, folder_path, device, desired_shape, n_condition, n_prediction, if_flow=False, samples_per_dir=1, cache_size=150):
        super().__init__()
        self.device = device
        self.desired_shape = desired_shape
        self.transform = transforms.Compose([
            transforms.Resize(desired_shape),
            transforms.ToTensor(),
        ])
        # self.cache_size = cache_size
        # self.cache = {}
        self.samples_per_dir = samples_per_dir
        self.images, self.flows, self.labels, self.filenames = self.read_and_sort_images(folder_path, n_condition, n_prediction, if_flow=if_flow)

    def read_and_sort_images(self, folder_path, n_condition, n_prediction, if_flow=False):
        images = []
        labels = []
        flows = []
        filenames = []
        for dir in os.listdir(folder_path):
            dir_path = os.path.join(folder_path, dir, 'img')
            if not os.path.isdir(dir_path):
                continue  # 跳过非目录文件
            print(f"Processing directory: {dir}")
            image_files = [f for f in os.listdir(dir_path) if f.endswith('.jpg')]
            image_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

            # Random sampling within the available range
            max_index = len(image_files) - n_condition - n_prediction + 1
            if max_index < 1:
                continue  # Skip directories with not enough images
            sampled_indices = random.sample(range(max_index), min(self.samples_per_dir, max_index))

            for i in sampled_indices:
                img_filename = image_files[i + n_condition - 1]
                img_path = os.path.join(dir_path, img_filename)
                img = Image.open(img_path).convert('RGB')  # 确保图像为RGB
                img = self.transform(img)
                images.append(img)

                if if_flow:
                    flows_tmp = torch.zeros((n_condition - 1,) + img.shape)
                    labels_tmp = torch.zeros((n_prediction,) + img.shape)
                    for j in range(n_condition - 1):
                        img_path_1 = os.path.join(dir_path, image_files[i + j])
                        img_path_2 = os.path.join(dir_path, image_files[i + j + 1])

                        img_1 = Image.open(img_path_1).convert('RGB')
                        img_2 = Image.open(img_path_2).convert('RGB')
                        img_1 = self.transform(img_1)
                        img_2 = self.transform(img_2)
                        flows_tmp[j] = img_2 - img_1

                    for k in range(n_prediction):
                        label_filename = image_files[i + n_condition + k]
                        label_path = os.path.join(dir_path, label_filename)
                        label = Image.open(label_path).convert('RGB')
                        label = self.transform(label)
                        labels_tmp[k] = label
                else:
                    flows_tmp = torch.zeros((n_condition,) + img.shape)
                    labels_tmp = torch.zeros((n_prediction,) + img.shape)
                    for j in range(n_condition):
                        img_path = os.path.join(dir_path, image_files[i + j])
                        img = Image.open(img_path).convert('RGB')
                        img = self.transform(img)
                        flows_tmp[j] = img
                    for k in range(n_prediction):
                        label_filename = image_files[i + n_condition + k]
                        label_path = os.path.join(dir_path, label_filename)
                        label = Image.open(label_path).convert('RGB')
                        label = self.transform(label)
                        labels_tmp[k] = label

                flows.append(flows_tmp)
                labels.append(labels_tmp)
                # 修改filename，包含开始帧序号i
                label_filename = os.path.join(dir, f"{dir}_{i}-{n_condition}_{n_prediction}")
                filenames.append(label_filename)

        return images, flows, labels, filenames

    def __len__(self):
        return len(self.flows)

    def __getitem__(self, i):
        img = self.images[i]
        flow = self.flows[i]
        label = self.labels[i]
        filename = self.filenames[i]
        # return img.to(self.device), flow.to(self.device), label.to(self.device), filename
        return img, flow, label, filename
    
class AirMot_Dataset_v1(Dataset):  # dynamic load
    def __init__(self, folder_path, device, desired_shape, n_condition, n_prediction, if_flow=False, samples_per_dir=1):
        super().__init__()
        self.device = device
        self.desired_shape = desired_shape
        self.samples_per_dir = samples_per_dir
        self.n_condition = n_condition
        self.n_prediction = n_prediction
        self.if_flow = if_flow
        self.transform = transforms.Compose([
            transforms.Resize(desired_shape),
            transforms.ToTensor(),
        ])
        # Store metadata instead of actual images
        self.data_info = self.read_and_sort_images(folder_path)

    def read_and_sort_images(self, folder_path):
        data_info = []
        for dir_name in os.listdir(folder_path):
            dir_path = os.path.join(folder_path, dir_name, 'img')
            if not os.path.isdir(dir_path):
                continue  # Skip if not a directory
            print(f"Processing directory: {dir_name}")
            image_files = [f for f in os.listdir(dir_path) if f.endswith('.jpg')]
            image_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

            # Random sampling within the available range
            max_index = len(image_files) - self.n_condition - self.n_prediction + 1
            if max_index < 1:
                continue  # Skip directories with not enough images
            sampled_indices = random.sample(range(max_index), min(self.samples_per_dir, max_index))

            for i in sampled_indices:
                data_entry = {
                    'dir': dir_path,
                    'image_files': image_files,
                    'start_idx': i,
                    'label_filename': f"{dir_name}_{i}-{self.n_condition}_{self.n_prediction}"
                }
                data_info.append(data_entry)
        return data_info

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        data_entry = self.data_info[idx]
        dir_path = data_entry['dir']
        image_files = data_entry['image_files']
        start_idx = data_entry['start_idx']
        label_filename = data_entry['label_filename']

        # Load images dynamically
        if self.if_flow:
            flows = []
            for j in range(self.n_condition - 1):
                img_path_1 = os.path.join(dir_path, image_files[start_idx + j])
                img_path_2 = os.path.join(dir_path, image_files[start_idx + j + 1])

                img_1 = Image.open(img_path_1).convert('RGB')
                img_2 = Image.open(img_path_2).convert('RGB')
                img_1 = self.transform(img_1)
                img_2 = self.transform(img_2)
                flow = img_2 - img_1  # Simplified optical flow
                flows.append(flow)

            flows = torch.stack(flows)

            # Load labels (n_prediction frames)
            labels = []
            for k in range(self.n_prediction):
                label_path = os.path.join(dir_path, image_files[start_idx + self.n_condition + k])
                label = Image.open(label_path).convert('RGB')
                label = self.transform(label)
                labels.append(label)
            labels = torch.stack(labels)

            # Use the last image as the img
            img_path = os.path.join(dir_path, image_files[start_idx + self.n_condition - 1])
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)

            # return img.to(self.device), flows.to(self.device), labels.to(self.device), label_filename
            return img, flows, labels, label_filename

        else:
            # For non-flow, load n_condition frames
            flows = []
            for j in range(self.n_condition):
                img_path = os.path.join(dir_path, image_files[start_idx + j])
                # print(img_path)
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
                flows.append(img)
            flows = torch.stack(flows)

            # Load labels (n_prediction frames)
            labels = []
            for k in range(self.n_prediction):
                label_path = os.path.join(dir_path, image_files[start_idx + self.n_condition + k])
                label = Image.open(label_path).convert('RGB')
                label = self.transform(label)
                labels.append(label)
            labels = torch.stack(labels)

            # Use the last image of flows as the img
            img = flows[-1]

            # return img.to(self.device), flows.to(self.device), labels.to(self.device), label_filename
            return img, flows, labels, label_filename

class AirMOTDatasetRGB(Dataset):
    def __init__(self, folder_path, device, desired_shape, n_condition):
        super().__init__()
        self.device = device
        self.desired_shape = desired_shape
        self.transform = transforms.Compose([
            transforms.Resize(desired_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化每个通道
        ])
        self.images, self.flows, self.labels = self.read_and_sort_images(folder_path, n_condition)
        # self.image_mean, self.image_std = self.calculate_global_mean_std(self.images)
        # self.label_mean, self.label_std = self.calculate_global_mean_std(self.labels)

        # 归一化的transform
        # self.normalize_transform = transforms.Normalize(mean=self.image_mean, std=self.image_std)
        # 计算全局最小值和最大值
        # self.global_min, self.global_max = self.calculate_global_min_max()
        # print('min max', self.global_min, self.global_max)

    def read_and_sort_images(self, folder_path, n_condition):
        images = []
        labels = []
        flows = []
        # print(folder_path)
        for dir in os.listdir(folder_path):
            # print('percentage:', f'{int(dir) / len(os.listdir(folder_path)):.2f}')
            image_files = [f for f in os.listdir(os.path.join(folder_path, dir, 'img')) if f.endswith('.jpg')]
            image_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

            for i in range(len(image_files) - n_condition):  # -1 because the last image won't have a label

                img_path = os.path.join(os.path.join(folder_path, dir), image_files[i + n_condition - 1])
                img = Image.open(img_path).convert('RGB')
                # print(img.size)
                img = self.transform(img)
                # channels, desired_width, desired_height = img.shape
                # img = img.view(-1, channels * desired_width * desired_height)
                images.append(img)
                # print(img.shape)

                label_path = os.path.join(os.path.join(folder_path, dir), image_files[i + n_condition])
                label = Image.open(label_path).convert('RGB')
                label = self.transform(label)
                # label = label.view(-1, channels * desired_width * desired_height)
                labels.append(label)
                # print(label.shape)
                # zero = torch.zeros_like(label)
                flows_tmp = torch.zeros((n_condition - 1,) + label.shape)

                for j in range(n_condition-1):
                    img_path_1 = os.path.join(os.path.join(folder_path, dir), image_files[i + j])
                    img_path_2 = os.path.join(os.path.join(folder_path, dir), image_files[i + j + 1])

                    img_1 = Image.open(img_path_1).convert('RGB')
                    img_2 = Image.open(img_path_2).convert('RGB')
                    img_1 = self.transform(img_1)
                    img_2 = self.transform(img_2)
                    flow = img_2 - img_1
                    # channels, desired_width, desired_height = flow.shape
                    # print("flow:",flow.shape)
                    # flow = flow.view(-1) # 压缩成1维
                    # print(flow.shape)
                    flows_tmp[j] = flow #
                flows.append(flows_tmp)
        return images, flows, labels

    def __len__(self):
        return len(self.flows)

    def __getitem__(self, i):
        # 获取原始图像、光流和标签
        img = self.images[i]
        flow = self.flows[i]
        label = self.labels[i]

        return img.to(self.device), flow.to(self.device), label.to(self.device)

class UAVID_Images_Dataset_v0(Dataset):  # static load
    def __init__(self, folder_path, device, desired_shape, n_condition, n_prediction, if_flow=False, samples_per_dir=1, cache_size=150):
        super().__init__()
        self.device = device
        self.desired_shape = desired_shape
        self.transform = transforms.Compose([
            transforms.Resize(desired_shape),
            transforms.ToTensor(),
        ])
        # self.cache_size = cache_size
        # self.cache = {}
        self.samples_per_dir = samples_per_dir
        self.images, self.flows, self.labels, self.filenames = self.read_and_sort_images(folder_path, n_condition, n_prediction, if_flow=if_flow)

    def read_and_sort_images(self, folder_path, n_condition, n_prediction, if_flow=False):
        images = []
        labels = []
        flows = []
        filenames = []
        for dir in os.listdir(folder_path):
            dir_path = os.path.join(folder_path, dir)
            # dir_path = folder_path
            if not os.path.isdir(dir_path):
                continue  # 跳过非目录文件
            print(f"Processing directory: {dir}")
            image_files = [f for f in os.listdir(dir_path) if f.endswith('.jpg')]
            image_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

            # Random sampling within the available range
            max_index = len(image_files) - n_condition - n_prediction + 1
            if max_index < 1:
                continue  # Skip directories with not enough images
            sampled_indices = random.sample(range(max_index), min(self.samples_per_dir, max_index))

            for i in sampled_indices:
                img_filename = image_files[i + n_condition - 1]
                img_path = os.path.join(dir_path, img_filename)
                img = Image.open(img_path).convert('RGB')  # 确保图像为RGB
                img = self.transform(img)
                images.append(img)

                if if_flow:
                    flows_tmp = torch.zeros((n_condition - 1,) + img.shape)
                    labels_tmp = torch.zeros((n_prediction,) + img.shape)
                    for j in range(n_condition - 1):
                        img_path_1 = os.path.join(dir_path, image_files[i + j])
                        img_path_2 = os.path.join(dir_path, image_files[i + j + 1])

                        img_1 = Image.open(img_path_1).convert('RGB')
                        img_2 = Image.open(img_path_2).convert('RGB')
                        img_1 = self.transform(img_1)
                        img_2 = self.transform(img_2)
                        flows_tmp[j] = img_2 - img_1

                    for k in range(n_prediction):
                        label_filename = image_files[i + n_condition + k]
                        label_path = os.path.join(dir_path, label_filename)
                        label = Image.open(label_path).convert('RGB')
                        label = self.transform(label)
                        labels_tmp[k] = label
                else:
                    flows_tmp = torch.zeros((n_condition,) + img.shape)
                    labels_tmp = torch.zeros((n_prediction,) + img.shape)
                    for j in range(n_condition):
                        img_path = os.path.join(dir_path, image_files[i + j])
                        img = Image.open(img_path).convert('RGB')
                        img = self.transform(img)
                        flows_tmp[j] = img
                    for k in range(n_prediction):
                        label_filename = image_files[i + n_condition + k]
                        label_path = os.path.join(dir_path, label_filename)
                        label = Image.open(label_path).convert('RGB')
                        label = self.transform(label)
                        labels_tmp[k] = label

                flows.append(flows_tmp)
                labels.append(labels_tmp)
                # 修改filename，包含开始帧序号i
                label_filename = os.path.join(dir, f"{dir}_{i}-{n_condition}_{n_prediction}")
                filenames.append(label_filename)

        return images, flows, labels, filenames

    def __len__(self):
        return len(self.flows)

    def __getitem__(self, i):
        img = self.images[i]
        flow = self.flows[i]
        label = self.labels[i]
        filename = self.filenames[i]
        # return img.to(self.device), flow.to(self.device), label.to(self.device), filename
        return img, flow, label, filename

class UAVID_Images_Dataset_v1(Dataset):  # dynamic load
    def __init__(self, folder_path, device, desired_shape, n_condition, n_prediction, if_flow=False, samples_per_dir=1):
        super().__init__()
        self.device = device
        self.desired_shape = desired_shape
        self.samples_per_dir = samples_per_dir
        self.n_condition = n_condition
        self.n_prediction = n_prediction
        self.if_flow = if_flow
        self.transform = transforms.Compose([
            transforms.Resize(desired_shape),
            transforms.ToTensor(),
        ])
        # Store metadata instead of actual images
        self.data_info = self.read_and_sort_images(folder_path)

    def read_and_sort_images(self, folder_path):
        data_info = []
        for dir_name in os.listdir(folder_path):
            # dir_path = os.path.join(folder_path, dir_name)
            dir_path = folder_path
            if not os.path.isdir(dir_path):
                continue  # Skip if not a directory
            print(f"Processing directory: {dir_name}")
            image_files = [f for f in os.listdir(dir_path) if f.endswith('.jpg')]
            image_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

            # Random sampling within the available range
            max_index = len(image_files) - self.n_condition - self.n_prediction + 1
            if max_index < 1:
                continue  # Skip directories with not enough images
            sampled_indices = random.sample(range(max_index), min(self.samples_per_dir, max_index))

            for i in sampled_indices:
                data_entry = {
                    'dir': dir_path,
                    'image_files': image_files,
                    'start_idx': i,
                    'label_filename': f"{dir_name}_{i}-{self.n_condition}_{self.n_prediction}"
                }
                data_info.append(data_entry)
        return data_info

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        data_entry = self.data_info[idx]
        dir_path = data_entry['dir']
        image_files = data_entry['image_files']
        start_idx = data_entry['start_idx']
        label_filename = data_entry['label_filename']

        # Load images dynamically
        if self.if_flow:
            flows = []
            for j in range(self.n_condition - 1):
                img_path_1 = os.path.join(dir_path, image_files[start_idx + j])
                img_path_2 = os.path.join(dir_path, image_files[start_idx + j + 1])

                img_1 = Image.open(img_path_1).convert('RGB')
                img_2 = Image.open(img_path_2).convert('RGB')
                img_1 = self.transform(img_1)
                img_2 = self.transform(img_2)
                flow = img_2 - img_1  # Simplified optical flow
                flows.append(flow)

            flows = torch.stack(flows)

            # Load labels (n_prediction frames)
            labels = []
            for k in range(self.n_prediction):
                label_path = os.path.join(dir_path, image_files[start_idx + self.n_condition + k])
                label = Image.open(label_path).convert('RGB')
                label = self.transform(label)
                labels.append(label)
            labels = torch.stack(labels)

            # Use the last image as the img
            img_path = os.path.join(dir_path, image_files[start_idx + self.n_condition - 1])
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)

            # return img.to(self.device), flows.to(self.device), labels.to(self.device), label_filename
            return img, flows, labels, label_filename

        else:
            # For non-flow, load n_condition frames
            flows = []
            for j in range(self.n_condition):
                img_path = os.path.join(dir_path, image_files[start_idx + j])
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
                flows.append(img)
            flows = torch.stack(flows)

            # Load labels (n_prediction frames)
            labels = []
            for k in range(self.n_prediction):
                label_path = os.path.join(dir_path, image_files[start_idx + self.n_condition + k])
                label = Image.open(label_path).convert('RGB')
                label = self.transform(label)
                labels.append(label)
            labels = torch.stack(labels)

            # Use the last image of flows as the img
            img = flows[-1]

            # return img.to(self.device), flows.to(self.device), labels.to(self.device), label_filename
            return img, flows, labels, label_filename

class UAVID_Images_Dataset_v2(Dataset):  # static load
    def __init__(self, folder_path, device, desired_shape, n_condition, n_prediction, if_flow=False, samples_per_dir=10, max_cache_size=10000):
        super().__init__()
        self.device = device
        self.desired_shape = desired_shape
        self.transform = transforms.Compose([
            transforms.Resize(desired_shape),
            transforms.ToTensor(),
        ])
        self.samples_per_dir = samples_per_dir
        self.max_cache_size = max_cache_size
        self.folder_path = folder_path  # 保存folder_path以供后续使用

        self.images, self.flows, self.labels, self.filenames = self.read_and_sort_images(folder_path, n_condition, n_prediction, if_flow=if_flow)

    @lru_cache(maxsize=10000)
    def load_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
        return self.transform(img)

    def read_and_sort_images(self, folder_path, n_condition, n_prediction, if_flow=False):
        images = []
        labels = []
        flows = []
        filenames = []

        with ProcessPoolExecutor() as executor:
            # 使用 partial 绑定额外的参数
            process_func = partial(
                process_directory,
                folder_path=folder_path,
                n_condition=n_condition,
                n_prediction=n_prediction,
                if_flow=if_flow,
                transform=self.transform,
                samples_per_dir=self.samples_per_dir
            )
            results = executor.map(process_func, os.listdir(folder_path))
            for dir_images, dir_flows, dir_labels, dir_filenames in results:
                images.extend(dir_images)
                flows.extend(dir_flows)
                labels.extend(dir_labels)
                filenames.extend(dir_filenames)

        return images, flows, labels, filenames

    def __len__(self):
        return len(self.flows)

    def __getitem__(self, i):
        img = self.images[i]
        flow = self.flows[i]
        label = self.labels[i]
        filename = self.filenames[i]
        return img, flow, label, filename


class UAVID_Images_Dataset_v1(Dataset):  # dynamic load
    def __init__(self, folder_path, device, desired_shape, n_condition, n_prediction, if_flow=False, samples_per_dir=1):
        super().__init__()
        self.device = device
        self.desired_shape = desired_shape
        self.samples_per_dir = samples_per_dir
        self.n_condition = n_condition
        self.n_prediction = n_prediction
        self.if_flow = if_flow
        self.transform = transforms.Compose([
            transforms.Resize(desired_shape),
            transforms.ToTensor(),
        ])
        # Store metadata instead of actual images
        self.data_info = self.read_and_sort_images(folder_path)

    def read_and_sort_images(self, folder_path):
        data_info = []
        for dir_name in os.listdir(folder_path):
            dir_path = os.path.join(folder_path, dir_name)
            if not os.path.isdir(dir_path):
                continue  # Skip if not a directory
            print(f"Processing directory: {dir_name}")
            image_files = [f for f in os.listdir(dir_path) if f.endswith('.jpg')]
            image_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

            # Random sampling within the available range
            max_index = len(image_files) - self.n_condition - self.n_prediction + 1
            if max_index < 1:
                continue  # Skip directories with not enough images
            sampled_indices = random.sample(range(max_index), min(self.samples_per_dir, max_index))

            for i in sampled_indices:
                data_entry = {
                    'dir': dir_path,
                    'image_files': image_files,
                    'start_idx': i,
                    'label_filename': f"{dir_name}_{i}-{self.n_condition}_{self.n_prediction}"
                }
                data_info.append(data_entry)
        return data_info

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        data_entry = self.data_info[idx]
        dir_path = data_entry['dir']
        image_files = data_entry['image_files']
        start_idx = data_entry['start_idx']
        label_filename = data_entry['label_filename']

        # Load images dynamically
        if self.if_flow:
            flows = []
            for j in range(self.n_condition - 1):
                img_path_1 = os.path.join(dir_path, image_files[start_idx + j])
                img_path_2 = os.path.join(dir_path, image_files[start_idx + j + 1])

                img_1 = Image.open(img_path_1).convert('RGB')
                img_2 = Image.open(img_path_2).convert('RGB')
                img_1 = self.transform(img_1)
                img_2 = self.transform(img_2)
                flow = img_2 - img_1  # Simplified optical flow
                flows.append(flow)

            flows = torch.stack(flows)

            # Load labels (n_prediction frames)
            labels = []
            for k in range(self.n_prediction):
                label_path = os.path.join(dir_path, image_files[start_idx + self.n_condition + k])
                label = Image.open(label_path).convert('RGB')
                label = self.transform(label)
                labels.append(label)
            labels = torch.stack(labels)

            # Use the last image as the img
            img_path = os.path.join(dir_path, image_files[start_idx + self.n_condition - 1])
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)

            # return img.to(self.device), flows.to(self.device), labels.to(self.device), label_filename
            return img, flows, labels, label_filename

        else:
            # For non-flow, load n_condition frames
            flows = []
            for j in range(self.n_condition):
                img_path = os.path.join(dir_path, image_files[start_idx + j])
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
                flows.append(img)
            flows = torch.stack(flows)

            # Load labels (n_prediction frames)
            labels = []
            for k in range(self.n_prediction):
                label_path = os.path.join(dir_path, image_files[start_idx + self.n_condition + k])
                label = Image.open(label_path).convert('RGB')
                label = self.transform(label)
                labels.append(label)
            labels = torch.stack(labels)

            # Use the last image of flows as the img
            img = flows[-1]

            # return img.to(self.device), flows.to(self.device), labels.to(self.device), label_filename
            return img, flows, labels, label_filename

class UAVID_Images_Dataset_v2(Dataset):  # static load
    def __init__(self, folder_path, device, desired_shape, n_condition, n_prediction, if_flow=False, samples_per_dir=10, max_cache_size=10000):
        super().__init__()
        self.device = device
        self.desired_shape = desired_shape
        self.transform = transforms.Compose([
            transforms.Resize(desired_shape),
            transforms.ToTensor(),
        ])
        self.samples_per_dir = samples_per_dir
        self.max_cache_size = max_cache_size
        self.folder_path = folder_path  # 保存folder_path以供后续使用

        self.images, self.flows, self.labels, self.filenames = self.read_and_sort_images(folder_path, n_condition, n_prediction, if_flow=if_flow)

    @lru_cache(maxsize=10000)
    def load_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
        return self.transform(img)

    def read_and_sort_images(self, folder_path, n_condition, n_prediction, if_flow=False):
        images = []
        labels = []
        flows = []
        filenames = []

        with ProcessPoolExecutor() as executor:
            # 使用 partial 绑定额外的参数
            process_func = partial(
                process_directory,
                folder_path=folder_path,
                n_condition=n_condition,
                n_prediction=n_prediction,
                if_flow=if_flow,
                transform=self.transform,
                samples_per_dir=self.samples_per_dir
            )
            results = executor.map(process_func, os.listdir(folder_path))
            for dir_images, dir_flows, dir_labels, dir_filenames in results:
                images.extend(dir_images)
                flows.extend(dir_flows)
                labels.extend(dir_labels)
                filenames.extend(dir_filenames)

        return images, flows, labels, filenames

    def __len__(self):
        return len(self.flows)

    def __getitem__(self, i):
        img = self.images[i]
        flow = self.flows[i]
        label = self.labels[i]
        filename = self.filenames[i]
        return img, flow, label, filename
    

class UAVID_Images_Dataset_v3(Dataset):  # dynamic load+multiresolution
    def __init__(self, folder_path, device, desired_shape, n_condition, n_prediction, if_flow=False, samples_per_dir=1):
        super().__init__()
        self.device = device
        self.desired_shape = desired_shape
        self.samples_per_dir = samples_per_dir
        self.n_condition = n_condition
        self.n_prediction = n_prediction
        self.if_flow = if_flow
        self.transform = transforms.Compose([
            transforms.Resize(desired_shape),
            transforms.ToTensor(),
        ])
        # 存储原始分辨率信息
        self.original_resolutions = {}
        # Store metadata instead of actual images
        self.data_info = self.read_and_sort_images(folder_path)

    def read_and_sort_images(self, folder_path):
        data_info = []
        for dir_name in os.listdir(folder_path):
            dir_path = os.path.join(folder_path, dir_name)
            if not os.path.isdir(dir_path):
                continue  # Skip if not a directory
            print(f"Processing directory: {dir_name}")
            image_files = [f for f in os.listdir(dir_path) if f.endswith('.jpg')]
            image_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

            # Random sampling within the available range
            max_index = len(image_files) - self.n_condition - self.n_prediction + 1
            if max_index < 1:
                continue  # Skip directories with not enough images
            sampled_indices = random.sample(range(max_index), min(self.samples_per_dir, max_index))

            for i in sampled_indices:
                # 获取第一帧的原始分辨率作为参考
                first_img_path = os.path.join(dir_path, image_files[i])
                with Image.open(first_img_path) as img:
                    orig_size = img.size  # (width, height)

                data_entry = {
                    'dir': dir_path,
                    'image_files': image_files,
                    'start_idx': i,
                    'label_filename': f"{dir_name}_{i}-{self.n_condition}_{self.n_prediction}",
                    'original_resolution': orig_size  # 添加原始分辨率信息
                }
                data_info.append(data_entry)
        return data_info
    
    def load_and_preprocess_image(self, image_path):
        """加载并预处理图像，自动处理分辨率"""
        image = Image.open(image_path)
        orig_w, orig_h = image.size
        
        # 转换为张量
        image = transforms.ToTensor()(image)
        
        # 如果分辨率不匹配，进行调整
        if (orig_h, orig_w) != self.desired_shape:
            image = F.interpolate(
                image.unsqueeze(0),
                size=self.desired_shape,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
        return image

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        data_entry = self.data_info[idx]
        dir_path = data_entry['dir']
        image_files = data_entry['image_files']
        start_idx = data_entry['start_idx']
        label_filename = data_entry['label_filename']
        original_resolution = data_entry['original_resolution']  # 获取原始分辨率

        # Load images dynamically
        if self.if_flow:
            flows = []
            for j in range(self.n_condition - 1):
                img_path_1 = os.path.join(dir_path, image_files[start_idx + j])
                img_path_2 = os.path.join(dir_path, image_files[start_idx + j + 1])

                img_1 = Image.open(img_path_1).convert('RGB')
                img_2 = Image.open(img_path_2).convert('RGB')
                img_1 = self.transform(img_1)
                img_2 = self.transform(img_2)
                flow = img_2 - img_1  # Simplified optical flow
                flows.append(flow)

            flows = torch.stack(flows)

            # Load labels (n_prediction frames)
            labels = []
            for k in range(self.n_prediction):
                label_path = os.path.join(dir_path, image_files[start_idx + self.n_condition + k])
                label = Image.open(label_path).convert('RGB')
                label = self.transform(label)
                labels.append(label)
            labels = torch.stack(labels)

            # Use the last image as the img
            img_path = os.path.join(dir_path, image_files[start_idx + self.n_condition - 1])
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)

            # return img.to(self.device), flows.to(self.device), labels.to(self.device), label_filename
            return img, flows, labels, label_filename

        else:
            # For non-flow, load n_condition frames
            flows = []
            for j in range(self.n_condition):
                img_path = os.path.join(dir_path, image_files[start_idx + j])
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
                flows.append(img)
            flows = torch.stack(flows)

            # Load labels (n_prediction frames)
            labels = []
            for k in range(self.n_prediction):
                label_path = os.path.join(dir_path, image_files[start_idx + self.n_condition + k])
                label = Image.open(label_path).convert('RGB')
                label = self.transform(label)
                labels.append(label)
            labels = torch.stack(labels)

            # Use the last image of flows as the img
            img = flows[-1]

            # return img.to(self.device), flows.to(self.device), labels.to(self.device), label_filename
            return img, flows, labels, label_filename, original_resolution


# 定义顶级函数
def process_directory(dir, folder_path, n_condition, n_prediction, if_flow, transform, samples_per_dir):
    images = []
    flows = []
    labels = []
    filenames = []

    dir_path = os.path.join(folder_path, dir, 'img')
    if not os.path.isdir(dir_path):
        return images, flows, labels, filenames  # 跳过非目录文件

    print(f"Processing directory: {dir}")
    image_files = [f for f in os.listdir(dir_path) if f.endswith('.jpg')]
    image_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

    # Random sampling within the available range
    max_index = len(image_files) - n_condition - n_prediction + 1
    if max_index < 1:
        return images, flows, labels, filenames  # Skip directories with not enough images
    sampled_indices = random.sample(range(max_index), min(samples_per_dir, max_index))

    for i in sampled_indices:
        img_filename = image_files[i + n_condition - 1]
        img_path = os.path.join(dir_path, img_filename)
        img = Image.open(img_path).convert('RGB')  # 确保图像为RGB
        img = transform(img)
        images.append(img)

        if if_flow:
            flows_tmp = torch.zeros((n_condition - 1,) + img.shape)
            labels_tmp = torch.zeros((n_prediction,) + img.shape)
            for j in range(n_condition - 1):
                img_path_1 = os.path.join(dir_path, image_files[i + j])
                img_path_2 = os.path.join(dir_path, image_files[i + j + 1])

                img_1 = Image.open(img_path_1).convert('RGB')
                img_2 = Image.open(img_path_2).convert('RGB')
                img_1 = transform(img_1)
                img_2 = transform(img_2)
                flows_tmp[j] = img_2 - img_1

            for k in range(n_prediction):
                label_filename = image_files[i + n_condition + k]
                label_path = os.path.join(dir_path, label_filename)
                label = Image.open(label_path).convert('RGB')
                label = transform(label)
                labels_tmp[k] = label
        else:
            flows_tmp = torch.zeros((n_condition,) + img.shape)
            labels_tmp = torch.zeros((n_prediction,) + img.shape)
            for j in range(n_condition):
                img_path = os.path.join(dir_path, image_files[i + j])
                img = Image.open(img_path).convert('RGB')
                img = transform(img)
                flows_tmp[j] = img
            for k in range(n_prediction):
                label_filename = image_files[i + n_condition + k]
                label_path = os.path.join(dir_path, label_filename)
                label = Image.open(label_path).convert('RGB')
                label = transform(label)
                labels_tmp[k] = label

        flows.append(flows_tmp)
        labels.append(labels_tmp)
        label_filename = os.path.join(dir, f"{dir}_{i}-{n_condition}_{n_prediction}")
        filenames.append(label_filename)

    return images, flows, labels, filenames

class Caltech_Pedestrian_Dataset(Dataset):  # dynamic load
    def __init__(self, folder_path, device, desired_shape, n_condition, n_prediction, if_flow=False, samples_per_dir=1):
        super().__init__()
        self.device = device
        self.desired_shape = desired_shape
        self.samples_per_dir = samples_per_dir
        self.n_condition = n_condition
        self.n_prediction = n_prediction
        self.if_flow = if_flow
        self.transform = transforms.Compose([
            transforms.Resize(desired_shape),
            transforms.ToTensor(),
        ])
        # Store metadata instead of actual images
        self.data_info = self.read_and_sort_images(folder_path)

    def read_and_sort_images(self, folder_path):
        data_info = []
        for dir_name in os.listdir(folder_path):
            dir_path = os.path.join(folder_path, dir_name)
            if not os.path.isdir(dir_path):
                continue  # Skip if not a directory
            # print(f"Processing directory: {dir_name}")
            for dir_name in os.listdir(dir_path):
                dirpath = os.path.join(dir_path, dir_name)
                if not os.path.isdir(dirpath):
                    continue  # Skip if not a directory
                # print(f"Processing directory: {dir_name}")

                image_files = [f for f in os.listdir(dirpath) if f.endswith('.jpg')]
                image_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

                # Random sampling within the available range
                max_index = len(image_files) - self.n_condition - self.n_prediction + 1
                if max_index < 1:
                    continue  # Skip directories with not enough images
                sampled_indices = random.sample(range(max_index), min(self.samples_per_dir, max_index))

                for i in sampled_indices:
                    data_entry = {
                        'dir': dirpath,
                        'image_files': image_files,
                        'start_idx': i,
                        'label_filename': f"{dirpath}_{i}-{self.n_condition}_{self.n_prediction}"
                    }
                    data_info.append(data_entry)
        return data_info

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        data_entry = self.data_info[idx]
        dir_path = data_entry['dir']
        image_files = data_entry['image_files']
        start_idx = data_entry['start_idx']
        label_filename = data_entry['label_filename']

        # Load images dynamically
        if self.if_flow:
            flows = []
            for j in range(self.n_condition - 1):
                img_path_1 = os.path.join(dir_path, image_files[start_idx + j])
                img_path_2 = os.path.join(dir_path, image_files[start_idx + j + 1])

                img_1 = Image.open(img_path_1).convert('RGB')
                img_2 = Image.open(img_path_2).convert('RGB')
                img_1 = self.transform(img_1)
                img_2 = self.transform(img_2)
                flow = img_2 - img_1  # Simplified optical flow
                flows.append(flow)

            flows = torch.stack(flows)

            # Load labels (n_prediction frames)
            labels = []
            for k in range(self.n_prediction):
                label_path = os.path.join(dir_path, image_files[start_idx + self.n_condition + k])
                label = Image.open(label_path).convert('RGB')
                label = self.transform(label)
                labels.append(label)
            labels = torch.stack(labels)

            # Use the last image as the img
            img_path = os.path.join(dir_path, image_files[start_idx + self.n_condition - 1])
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)

            # return img.to(self.device), flows.to(self.device), labels.to(self.device), label_filename
            return img, flows, labels, label_filename

        else:
            # For non-flow, load n_condition frames
            flows = []
            for j in range(self.n_condition):
                img_path = os.path.join(dir_path, image_files[start_idx + j])
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
                flows.append(img)
            flows = torch.stack(flows)

            # Load labels (n_prediction frames)
            labels = []
            for k in range(self.n_prediction):
                label_path = os.path.join(dir_path, image_files[start_idx + self.n_condition + k])
                label = Image.open(label_path).convert('RGB')
                label = self.transform(label)
                labels.append(label)
            labels = torch.stack(labels)

            # Use the last image of flows as the img
            img = flows[-1]

            # return img.to(self.device), flows.to(self.device), labels.to(self.device), label_filename
            return img, flows, labels, label_filename

class Jinan_Dataset(Dataset):  # dynamic load for Jinan dataset
    """
    济南数据集加载器，支持嵌套目录结构
    数据组织方式：
    - 根目录/类别目录/视频目录/frame_xxx.jpg
    - 例如：/train/多云-动态目标-车/DJI_20250618153614_0011_V/frame_000000.jpg
    """
    def __init__(self, folder_path, device, desired_shape, n_condition, n_prediction, if_flow=False, samples_per_dir=1):
        super().__init__()
        self.device = device
        self.desired_shape = desired_shape
        self.samples_per_dir = samples_per_dir
        self.n_condition = n_condition
        self.n_prediction = n_prediction
        self.if_flow = if_flow
        self.transform = transforms.Compose([
            transforms.Resize(desired_shape),
            transforms.ToTensor(),
        ])
        # Store metadata instead of actual images
        self.data_info = self.read_and_sort_images(folder_path)
        print(f"济南数据集共加载了 {len(self.data_info)} 个样本")

    def read_and_sort_images(self, folder_path):
        data_info = []
        
        # 遍历类别目录
        for category_name in os.listdir(folder_path):
            category_path = os.path.join(folder_path, category_name)
            if not os.path.isdir(category_path):
                continue
            
            print(f"处理类别: {category_name}")
            
            # 遍历每个类别下的视频目录
            for video_name in os.listdir(category_path):
                video_path = os.path.join(category_path, video_name)
                if not os.path.isdir(video_path):
                    continue
                
                print(f"  处理视频: {video_name}")
                
                # 获取视频目录中的所有图像文件
                try:
                    image_files = [f for f in os.listdir(video_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    image_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0)
                    
                    # 检查是否有足够的帧
                    max_index = len(image_files) - self.n_condition - self.n_prediction + 1
                    if max_index < 1:
                        print(f"    警告: 视频 {video_name} 帧数不足 ({len(image_files)} 帧)，跳过")
                        continue
                    
                    # 随机采样
                    sampled_indices = random.sample(range(max_index), min(self.samples_per_dir, max_index))
                    
                    for i in sampled_indices:
                        data_entry = {
                            'category': category_name,
                            'video': video_name,
                            'dir': video_path,
                            'image_files': image_files,
                            'start_idx': i,
                            'label_filename': f"{category_name}_{video_name}_{i}-{self.n_condition}_{self.n_prediction}"
                        }
                        data_info.append(data_entry)
                        
                except PermissionError:
                    print(f"    权限错误: 无法访问 {video_path}")
                    continue
                except Exception as e:
                    print(f"    错误: 处理 {video_path} 时发生异常: {e}")
                    continue
        
        return data_info

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        data_entry = self.data_info[idx]
        dir_path = data_entry['dir']
        image_files = data_entry['image_files']
        start_idx = data_entry['start_idx']
        label_filename = data_entry['label_filename']

        try:
            # Load images dynamically
            if self.if_flow:
                flows = []
                for j in range(self.n_condition - 1):
                    img_path_1 = os.path.join(dir_path, image_files[start_idx + j])
                    img_path_2 = os.path.join(dir_path, image_files[start_idx + j + 1])

                    img_1 = Image.open(img_path_1).convert('RGB')
                    img_2 = Image.open(img_path_2).convert('RGB')
                    img_1 = self.transform(img_1)
                    img_2 = self.transform(img_2)
                    flow = img_2 - img_1  # Simplified optical flow
                    flows.append(flow)

                flows = torch.stack(flows)

                # Load labels (n_prediction frames)
                labels = []
                for k in range(self.n_prediction):
                    label_path = os.path.join(dir_path, image_files[start_idx + self.n_condition + k])
                    label = Image.open(label_path).convert('RGB')
                    label = self.transform(label)
                    labels.append(label)
                labels = torch.stack(labels)

                # Use the last image as the img
                img_path = os.path.join(dir_path, image_files[start_idx + self.n_condition - 1])
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)

                return img, flows, labels, label_filename

            else:
                # For non-flow, load n_condition frames
                flows = []
                for j in range(self.n_condition):
                    img_path = os.path.join(dir_path, image_files[start_idx + j])
                    img = Image.open(img_path).convert('RGB')
                    img = self.transform(img)
                    flows.append(img)
                flows = torch.stack(flows)

                # Load labels (n_prediction frames)
                labels = []
                for k in range(self.n_prediction):
                    label_path = os.path.join(dir_path, image_files[start_idx + self.n_condition + k])
                    label = Image.open(label_path).convert('RGB')
                    label = self.transform(label)
                    labels.append(label)
                labels = torch.stack(labels)

                # Use the last image of flows as the img
                img = flows[-1]

                return img, flows, labels, label_filename
                
        except Exception as e:
            print(f"加载样本时发生错误: {e}")
            # 返回一个默认的空样本，避免训练中断
            dummy_img = torch.zeros(3, *self.desired_shape)
            if self.if_flow:
                dummy_flows = torch.zeros(self.n_condition - 1, 3, *self.desired_shape)
            else:
                dummy_flows = torch.zeros(self.n_condition, 3, *self.desired_shape)
            dummy_labels = torch.zeros(self.n_prediction, 3, *self.desired_shape)
            return dummy_img, dummy_flows, dummy_labels, f"error_{idx}"

    def get_dataset_info(self):
        """获取数据集统计信息"""
        categories = {}
        for data_entry in self.data_info:
            category = data_entry['category']
            if category not in categories:
                categories[category] = {'videos': set(), 'samples': 0}
            categories[category]['videos'].add(data_entry['video'])
            categories[category]['samples'] += 1
        
        # 转换为可序列化的格式
        for category in categories:
            categories[category]['videos'] = len(categories[category]['videos'])
            
        return categories
