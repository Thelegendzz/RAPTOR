import os
import random
import shutil
import cv2
from pathlib import Path

def get_all_videos(source_dir):
    """
    获取所有视频文件的路径
    源文件夹结构: source_dir/[一级目录]/[二级目录]/视频文件
    动态搜索二级目录，不使用固定列表
    """
    video_extensions = ['.MP4', '.mp4']
    all_videos = []
    
    # 第一级目录名称
    first_level_dirs = ['多云-动态目标', '晴朗-动态目标', '晴朗-静态目标', '夜晚-动态目标', '夜晚-静态目标']
    
    for first_dir in first_level_dirs:
        first_dir_path = os.path.join(source_dir, first_dir)
        
        if not os.path.exists(first_dir_path):
            print(f"一级目录不存在: {first_dir_path}")
            continue
        
        print(f"检查一级目录: {first_dir}")
        
        # 动态搜索二级目录
        try:
            second_level_items = os.listdir(first_dir_path)
            second_level_dirs = [item for item in second_level_items 
                               if os.path.isdir(os.path.join(first_dir_path, item))]
            
            print(f"  发现 {len(second_level_dirs)} 个二级目录: {second_level_dirs}")
            
            for second_dir in second_level_dirs:
                # 构建完整路径
                full_path = os.path.join(first_dir_path, second_dir)
                
                print(f"  检查二级目录: {full_path}")
                
                # 遍历该目录下的所有文件
                try:
                    files = os.listdir(full_path)
                    video_count = 0
                    
                    for file in files:
                        file_path = os.path.join(full_path, file)
                        
                        # 检查是否为视频文件
                        if os.path.isfile(file_path):
                            _, ext = os.path.splitext(file)
                            if ext.lower() in video_extensions:
                                # 存储相对路径信息，便于后续处理
                                relative_path = os.path.join(first_dir, second_dir, file)
                                all_videos.append({
                                    'full_path': file_path,
                                    'relative_path': relative_path,
                                    'first_dir': first_dir,
                                    'second_dir': second_dir,
                                    'filename': file
                                })
                                video_count += 1
                    
                    print(f"    找到 {video_count} 个视频文件")
                    
                except Exception as e:
                    print(f"    无法访问目录 {full_path}: {e}")
                    
        except Exception as e:
            print(f"  无法访问一级目录 {first_dir_path}: {e}")
    
    return all_videos

def check_existing_folders(target_dir):
    """
    检查目标目录中是否已经存在所有预期的文件夹
    动态检查源目录中的二级目录组合
    返回True表示所有文件夹都已存在，可以跳过复制
    """
    if not os.path.exists(target_dir):
        return False
    
    # 需要从源目录动态获取预期的文件夹组合
    source_directory = "/mnt/sdc/datasets_active/jinan/DCIM"
    first_level_dirs = ['多云-动态目标', '晴朗-动态目标', '晴朗-静态目标', '夜晚-动态目标', '夜晚-静态目标']
    
    expected_folders = []
    
    # 动态获取所有可能的文件夹组合
    for first_dir in first_level_dirs:
        first_dir_path = os.path.join(source_directory, first_dir)
        
        if os.path.exists(first_dir_path):
            try:
                second_level_items = os.listdir(first_dir_path)
                second_level_dirs = [item for item in second_level_items 
                                   if os.path.isdir(os.path.join(first_dir_path, item))]
                
                for second_dir in second_level_dirs:
                    folder_name = f"{first_dir}-{second_dir}"
                    expected_folders.append(folder_name)
            except Exception as e:
                print(f"无法访问目录 {first_dir_path}: {e}")
    
    if not expected_folders:
        print("未找到任何预期的文件夹组合")
        return False
    
    # 检查每个预期文件夹是否存在且包含视频文件
    existing_folders = []
    for folder_name in expected_folders:
        folder_path = os.path.join(target_dir, folder_name)
        if os.path.exists(folder_path):
            # 检查文件夹中是否有视频文件
            try:
                video_files = [f for f in os.listdir(folder_path) 
                              if f.lower().endswith(('.mp4', '.MP4'))]
                if video_files:
                    existing_folders.append(folder_name)
            except Exception as e:
                print(f"无法访问文件夹 {folder_path}: {e}")
    
    print(f"预期文件夹总数: {len(expected_folders)}")
    print(f"发现已存在的文件夹: {len(existing_folders)}/{len(expected_folders)}")
    for folder in existing_folders:
        print(f"  - {folder}")
    
    # 如果所有文件夹都存在，返回True
    return len(existing_folders) == len(expected_folders)

def check_frame_directory_exists(frame_dir):
    """
    检查帧目录是否存在且包含完整的帧数据
    返回True表示帧目录已存在且包含数据，可以跳过视频复制和分帧操作
    """
    if not os.path.exists(frame_dir):
        return False
    
    print(f"检查帧目录: {frame_dir}")
    
    # 检查是否有子文件夹
    try:
        subdirs = [item for item in os.listdir(frame_dir) 
                  if os.path.isdir(os.path.join(frame_dir, item))]
        
        if not subdirs:
            print("帧目录为空")
            return False
        
        print(f"发现 {len(subdirs)} 个子文件夹")
        
        # 检查每个子文件夹是否包含视频文件夹（已分帧的视频）
        total_video_folders = 0
        for subdir in subdirs:
            subdir_path = os.path.join(frame_dir, subdir)
            try:
                video_folders = [item for item in os.listdir(subdir_path) 
                               if os.path.isdir(os.path.join(subdir_path, item))]
                
                # 检查每个视频文件夹是否包含帧图像
                valid_video_folders = 0
                for video_folder in video_folders:
                    video_folder_path = os.path.join(subdir_path, video_folder)
                    try:
                        frame_files = [f for f in os.listdir(video_folder_path) 
                                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        if frame_files:
                            valid_video_folders += 1
                    except Exception as e:
                        print(f"无法访问视频文件夹 {video_folder_path}: {e}")
                
                print(f"  子文件夹 {subdir}: {valid_video_folders} 个有效视频文件夹")
                total_video_folders += valid_video_folders
                
            except Exception as e:
                print(f"无法访问子文件夹 {subdir_path}: {e}")
        
        print(f"总共发现 {total_video_folders} 个包含帧数据的视频文件夹")
        
        # 如果有视频文件夹包含帧数据，认为帧目录已存在
        return total_video_folders > 0
        
    except Exception as e:
        print(f"无法访问帧目录 {frame_dir}: {e}")
        return False

def sample_video_frames(video_path, output_dir, target_fps=10):
    """
    对视频进行帧采样，按照指定fps输出帧图像
    target_fps=10表示每秒保留10帧（3帧抽1帧，原视频通常是30fps）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return False
    
    # 获取视频信息
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  视频信息: {original_fps:.2f}fps, 总帧数: {total_frames}")
    
    # 计算采样间隔（3帧抽1帧）
    frame_interval = int(original_fps / target_fps)
    if frame_interval < 1:
        frame_interval = 1
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 按间隔保存帧
        if frame_count % frame_interval == 0:
            frame_filename = f"frame_{saved_count:06d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"  采样完成: 从 {total_frames} 帧中保存了 {saved_count} 帧")
    return True

def process_video_sampling(target_dir):
    """
    遍历视频目录中的所有视频文件夹，对每个视频进行帧采样
    从 jinan_sample 目录复制视频到 jinan_sample_frame 目录并进行采样
    """
    # 视频源目录
    video_source_dir = "/mnt/sdc/datasets_active/jinan_sample"
    
    if not os.path.exists(video_source_dir):
        print(f"视频源目录不存在: {video_source_dir}")
        return
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        print(f"创建帧目标目录: {target_dir}")
    
    print(f"\n开始处理视频采样...")
    print(f"视频源目录: {video_source_dir}")
    print(f"帧目标目录: {target_dir}")
    
    # 遍历视频源目录中的所有子文件夹
    for folder_name in os.listdir(video_source_dir):
        video_folder_path = os.path.join(video_source_dir, folder_name)
        
        if not os.path.isdir(video_folder_path):
            continue
        
        print(f"\n处理文件夹: {folder_name}")
        
        # 在帧目标目录中创建对应的文件夹
        frame_folder_path = os.path.join(target_dir, folder_name)
        os.makedirs(frame_folder_path, exist_ok=True)
        
        # 获取视频文件夹中的所有视频文件
        try:
            video_files = [f for f in os.listdir(video_folder_path) 
                          if f.lower().endswith(('.mp4', '.MP4'))]
        except Exception as e:
            print(f"  无法访问视频文件夹 {video_folder_path}: {e}")
            continue
        
        if not video_files:
            print(f"  文件夹中没有视频文件")
            continue
        
        print(f"  找到 {len(video_files)} 个视频文件")
        
        # 对每个视频进行采样
        for i, video_file in enumerate(video_files, 1):
            video_path = os.path.join(video_folder_path, video_file)
            
            # 创建与视频文件同名的输出文件夹
            video_name = os.path.splitext(video_file)[0]
            output_dir = os.path.join(frame_folder_path, video_name)
            
            print(f"  [{i}/{len(video_files)}] 处理视频: {video_file}")
            
            # 如果输出文件夹已存在且包含帧图像，跳过
            if os.path.exists(output_dir):
                try:
                    existing_frames = [f for f in os.listdir(output_dir) 
                                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if existing_frames:
                        print(f"    跳过（已存在 {len(existing_frames)} 帧）")
                        continue
                except Exception as e:
                    print(f"    检查现有帧时出错: {e}")
            
            # 进行帧采样
            success = sample_video_frames(video_path, output_dir, target_fps=10)
            if not success:
                print(f"    采样失败")
    
    print(f"\n视频采样处理完成！")

def split_dataset(frame_dir, train_ratio=0.75, val_ratio=0.167, test_ratio=0.083):
    """
    按视频为单位，以9:2:1的比例将数据集划分到train, val, test文件夹中
    frame_dir: 包含帧数据的根目录
    train_ratio: 训练集比例 (0.75 = 9/12)
    val_ratio: 验证集比例 (0.167 = 2/12)
    test_ratio: 测试集比例 (0.083 = 1/12)
    """
    if not os.path.exists(frame_dir):
        print(f"帧目录不存在: {frame_dir}")
        return
    
    print(f"\n开始数据集划分...")
    print(f"帧目录: {frame_dir}")
    print(f"划分比例 - 训练集: {train_ratio:.3f}, 验证集: {val_ratio:.3f}, 测试集: {test_ratio:.3f}")
    
    # 创建输出目录
    output_base_dir = os.path.join(os.path.dirname(frame_dir), "jinan_dataset")
    train_dir = os.path.join(output_base_dir, "train")
    val_dir = os.path.join(output_base_dir, "val")
    test_dir = os.path.join(output_base_dir, "test")
    
    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"输出目录: {output_base_dir}")
    
    # 获取所有类别文件夹
    try:
        category_folders = [item for item in os.listdir(frame_dir) 
                           if os.path.isdir(os.path.join(frame_dir, item))]
    except Exception as e:
        print(f"无法访问帧目录: {e}")
        return
    
    if not category_folders:
        print("未找到任何类别文件夹")
        return
    
    print(f"发现 {len(category_folders)} 个类别文件夹: {category_folders}")
    
    total_videos = 0
    total_train = 0
    total_val = 0
    total_test = 0
    
    # 处理每个类别文件夹
    for category in category_folders:
        category_path = os.path.join(frame_dir, category)
        
        print(f"\n处理类别: {category}")
        
        # 获取该类别下的所有视频文件夹
        try:
            video_folders = [item for item in os.listdir(category_path) 
                           if os.path.isdir(os.path.join(category_path, item))]
        except Exception as e:
            print(f"  无法访问类别文件夹 {category_path}: {e}")
            continue
        
        if not video_folders:
            print(f"  类别 {category} 中没有视频文件夹")
            continue
        
        print(f"  找到 {len(video_folders)} 个视频文件夹")
        
        # 随机打乱视频文件夹列表
        random.shuffle(video_folders)
        
        # 计算划分点
        total_videos_in_category = len(video_folders)
        train_count = int(total_videos_in_category * train_ratio)
        val_count = int(total_videos_in_category * val_ratio)
        test_count = total_videos_in_category - train_count - val_count
        
        print(f"  划分方案: 训练集 {train_count}, 验证集 {val_count}, 测试集 {test_count}")
        
        # 创建对应的类别文件夹
        for split_dir in [train_dir, val_dir, test_dir]:
            category_split_dir = os.path.join(split_dir, category)
            os.makedirs(category_split_dir, exist_ok=True)
        
        # 分配视频到不同的集合
        splits = [
            (video_folders[:train_count], train_dir, "训练集"),
            (video_folders[train_count:train_count+val_count], val_dir, "验证集"),
            (video_folders[train_count+val_count:], test_dir, "测试集")
        ]
        
        for video_list, target_base_dir, split_name in splits:
            target_category_dir = os.path.join(target_base_dir, category)
            
            for video_folder in video_list:
                source_video_path = os.path.join(category_path, video_folder)
                target_video_path = os.path.join(target_category_dir, video_folder)
                
                try:
                    # 复制整个视频文件夹
                    if os.path.exists(target_video_path):
                        shutil.rmtree(target_video_path)
                    shutil.copytree(source_video_path, target_video_path)
                    
                    # 统计帧数
                    frame_count = len([f for f in os.listdir(target_video_path) 
                                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    
                    print(f"    {split_name}: {video_folder} ({frame_count} 帧)")
                    
                except Exception as e:
                    print(f"    复制失败 {video_folder}: {e}")
        
        # 更新总计数
        total_videos += total_videos_in_category
        total_train += train_count
        total_val += val_count
        total_test += test_count
    
    print(f"\n数据集划分完成！")
    print(f"总计:")
    print(f"  - 总视频数: {total_videos}")
    print(f"  - 训练集: {total_train} 个视频")
    print(f"  - 验证集: {total_val} 个视频")
    print(f"  - 测试集: {total_test} 个视频")
    print(f"  - 输出目录: {output_base_dir}")

def copy_random_videos(source_dir, target_dir, num_videos_per_folder=10):
    """
    为每个一级和二级目录组合创建文件夹，并在每个文件夹中随机选择指定数量的视频文件
    """
    # 获取所有视频文件
    all_videos = get_all_videos(source_dir)
    
    if not all_videos:
        print("未找到任何视频文件！")
        return
    
    print(f"总共找到 {len(all_videos)} 个视频文件")
    
    # 按照一级和二级目录分组
    video_groups = {}
    for video_info in all_videos:
        key = (video_info['first_dir'], video_info['second_dir'])
        if key not in video_groups:
            video_groups[key] = []
        video_groups[key].append(video_info)
    
    print(f"找到 {len(video_groups)} 个目录组合")
    
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    
    total_copied = 0
    total_folders_created = 0
    
    # 为每个组合创建文件夹并复制视频
    for (first_dir, second_dir), videos in video_groups.items():
        target_folder_name = f"{first_dir}-{second_dir}"
        target_subdir = os.path.join(target_dir, target_folder_name)
        
        print(f"\n处理组合: {first_dir} - {second_dir}")
        print(f"该组合下有 {len(videos)} 个视频文件")
        
        # 创建目标文件夹（如果已存在则清空）
        if os.path.exists(target_subdir):
            print(f"目标文件夹已存在，清空: {target_folder_name}")
            shutil.rmtree(target_subdir)
        
        os.makedirs(target_subdir, exist_ok=True)
        total_folders_created += 1
        print(f"创建目标文件夹: {target_folder_name}")
        
        # 随机选择视频（不超过可用视频数量）
        num_to_select = min(num_videos_per_folder, len(videos))
        selected_videos = random.sample(videos, num_to_select)
        
        print(f"从 {len(videos)} 个视频中选择 {num_to_select} 个进行复制")
        
        # 复制选中的视频
        copied_in_folder = 0
        for i, video_info in enumerate(selected_videos, 1):
            source_path = video_info['full_path']
            target_path = os.path.join(target_subdir, video_info['filename'])
            
            try:
                shutil.copy2(source_path, target_path)
                print(f"  [{i}/{len(selected_videos)}] 复制成功: {video_info['filename']}")
                copied_in_folder += 1
                total_copied += 1
            except Exception as e:
                print(f"  [{i}/{len(selected_videos)}] 复制失败: {video_info['filename']}, 错误: {e}")
        
        print(f"文件夹 {target_folder_name} 完成，复制了 {copied_in_folder} 个视频")
    
    print(f"\n复制统计:")
    print(f"  - 创建文件夹数: {total_folders_created} 个")
    print(f"  - 总计复制视频: {total_copied} 个")
    print(f"  - 每个文件夹目标视频数: {num_videos_per_folder} 个")
    
    print("复制完成！")

def check_and_fix_existing_dataset(dataset_dir):
    """
    检查并修复现有的数据集，删除空的验证集类别
    dataset_dir: 数据集根目录，包含train, val, test子目录
    """
    if not os.path.exists(dataset_dir):
        print(f"数据集目录不存在: {dataset_dir}")
        return
    
    print(f"\n检查现有数据集: {dataset_dir}")
    
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")
    test_dir = os.path.join(dataset_dir, "test")
    
    # 检查各个子目录是否存在
    for split_name, split_dir in [("训练集", train_dir), ("验证集", val_dir), ("测试集", test_dir)]:
        if not os.path.exists(split_dir):
            print(f"  {split_name}目录不存在: {split_dir}")
            return
    
    # 获取所有类别
    try:
        train_categories = [item for item in os.listdir(train_dir) 
                           if os.path.isdir(os.path.join(train_dir, item))]
        val_categories = [item for item in os.listdir(val_dir) 
                         if os.path.isdir(os.path.join(val_dir, item))]
        test_categories = [item for item in os.listdir(test_dir) 
                          if os.path.isdir(os.path.join(test_dir, item))]
    except Exception as e:
        print(f"无法读取数据集目录: {e}")
        return
    
    print(f"  训练集类别: {len(train_categories)}")
    print(f"  验证集类别: {len(val_categories)}")
    print(f"  测试集类别: {len(test_categories)}")
    
    # 检查验证集中的空类别
    empty_val_categories = []
    for category in val_categories:
        val_category_path = os.path.join(val_dir, category)
        try:
            video_folders = [item for item in os.listdir(val_category_path) 
                           if os.path.isdir(os.path.join(val_category_path, item))]
            
            if len(video_folders) == 0:
                empty_val_categories.append(category)
                print(f"  发现空验证集类别: {category}")
            else:
                print(f"  验证集类别 {category}: {len(video_folders)} 个视频")
        except Exception as e:
            print(f"  检查验证集类别 {category} 时出错: {e}")
            empty_val_categories.append(category)
    
    # 删除空的验证集类别
    if empty_val_categories:
        print(f"\n删除 {len(empty_val_categories)} 个空的验证集类别...")
        for category in empty_val_categories:
            # 删除验证集中的空类别
            val_category_path = os.path.join(val_dir, category)
            if os.path.exists(val_category_path):
                try:
                    shutil.rmtree(val_category_path)
                    print(f"  删除验证集空类别: {category}")
                except Exception as e:
                    print(f"  删除验证集类别 {category} 失败: {e}")
            
            # 同时删除测试集中的对应类别（保持一致性）
            test_category_path = os.path.join(test_dir, category)
            if os.path.exists(test_category_path):
                try:
                    shutil.rmtree(test_category_path)
                    print(f"  删除测试集对应类别: {category}")
                except Exception as e:
                    print(f"  删除测试集类别 {category} 失败: {e}")
    else:
        print("  未发现空的验证集类别")
    
    # 重新统计
    try:
        final_val_categories = [item for item in os.listdir(val_dir) 
                               if os.path.isdir(os.path.join(val_dir, item))]
        
        def count_videos_in_split(split_dir):
            total = 0
            for category in os.listdir(split_dir):
                category_path = os.path.join(split_dir, category)
                if os.path.isdir(category_path):
                    video_count = len([item for item in os.listdir(category_path) 
                                     if os.path.isdir(os.path.join(category_path, item))])
                    total += video_count
            return total
        
        final_val_videos = count_videos_in_split(val_dir)
        
        print(f"\n修复后统计:")
        print(f"  验证集类别数: {len(final_val_categories)}")
        print(f"  验证集视频数: {final_val_videos}")
        
        if final_val_videos == 0:
            print("\n⚠️  警告: 验证集仍然为空！")
            print("建议重新运行数据集划分脚本。")
        else:
            print("✓ 数据集修复完成！")
            
    except Exception as e:
        print(f"统计修复结果时出错: {e}")

def main():
    # 配置源目录和目标目录
    source_directory = "/mnt/sdc/datasets_active/jinan/DCIM"  # 请修改为你的源目录路径
    target_directory = "/mnt/sdc/datasets_active/jinan_sample"  # 请修改为你的目标目录路径
    target_directory_2 = "/mnt/sdc/datasets_active/jinan_sample_frame"  # 请修改为你的目标目录路径
    final_dataset_dir = "/mnt/sdc/datasets_active/jinan_dataset"  # 最终数据集目录

    sample_video_frames(video_path='/hy-tmp/DJI_20250617212726_0007_V.MP4', output_dir='/hy-tmp/DJI_20250617212726_0007_V_frame')
    
    # 每个文件夹中要复制的视频数量
    # videos_per_folder = 12
    
    # print(f"源目录: {source_directory}")
    # print(f"目标目录: {target_directory}")
    # print(f"帧目录: {target_directory_2}")
    # print(f"最终数据集目录: {final_dataset_dir}")
    # print(f"每个文件夹要复制的视频数量: {videos_per_folder}")
    
    # # 检查源目录是否存在
    # if not os.path.exists(source_directory):
    #     print(f"错误: 源目录不存在 - {source_directory}")
    #     return
    
    # # 如果最终数据集已存在，先检查并修复
    # if os.path.exists(final_dataset_dir):
    #     print(f"\n发现现有数据集，进行检查和修复...")
    #     check_and_fix_existing_dataset(final_dataset_dir)
    #     return
    
    # # 首先检查帧目录是否已存在
    # if check_frame_directory_exists(target_directory_2):
    #     print("帧目录已存在且包含完整数据，跳过视频复制和分帧操作。")
    #     print("直接进行数据集划分...")
    #     split_dataset(target_directory_2)
    #     return
    
    # # 检查目标目录是否已存在所有预期文件夹
    # if check_existing_folders(target_directory):
    #     print("目标目录已包含所有预期文件夹，跳过复制。")
    #     # 如果目标目录已包含所有预期文件夹，则直接进行视频采样
    #     process_video_sampling(target_directory_2)
    # else:
    #     # 执行复制
    #     copy_random_videos(source_directory, target_directory, videos_per_folder)
    #     # 复制完成后进行视频采样
    #     process_video_sampling(target_directory_2)
    
    # # 最后进行数据集划分
    # split_dataset(target_directory_2)

if __name__ == "__main__":
    main()
