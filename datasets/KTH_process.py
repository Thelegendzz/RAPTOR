import os
import cv2
from tqdm import tqdm
import cv2
import numpy as np

def ensure_dir(path):
    """
    确保目录存在，不存在则创建。
    
    参数:
    - path (str): 目录路径。
    """
    if not os.path.exists(path):
        os.makedirs(path)

def save_frame(frame, save_path):
    """
    保存单个帧为图片文件。
    
    参数:
    - frame (numpy.ndarray): 帧数据。
    - save_path (str): 图片保存路径。
    """
    cv2.imwrite(save_path, frame)

def split_video_into_frames(video_path, output_dir, frame_rate=1):
    """
    将单个视频分割为帧并保存为图片。
    
    参数:
    - video_path (str): 视频文件路径。
    - output_dir (str): 保存帧的目标目录。
    - frame_rate (int): 每秒保存的帧数（默认每秒1帧）。
    """
    ensure_dir(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps > 0:
        interval = int(round(fps / frame_rate))
    else:
        interval = 1  # 默认每帧保存
    
    frame_count = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
            save_frame(frame, frame_filename)
            saved_count += 1
        frame_count += 1
    
    cap.release()

def split_videos_to_frames(dataset_dir, output_root, frame_rate=1):
    """
    遍历数据集中的所有视频，将每个视频分割为帧并保存为图片。
    
    参数:
    - dataset_dir (str): KTH 数据集的根目录，包含动作类别的子文件夹（如 'walking' 等）。
    - output_root (str): 分割后图片的根目录。
    - frame_rate (int): 每秒保存的帧数（默认为1帧每秒）。
    """
    ensure_dir(output_root)
    
    # 获取所有动作类别文件夹
    actions = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    num_actions = len(actions)
    print(f"共找到 {num_actions} 个动作类别文件夹。\n")
    
    for action in tqdm(actions, desc="处理动作类别"):
        action_dir = os.path.join(dataset_dir, action)
        # 获取该动作类别下的所有视频文件
        video_files = [f for f in os.listdir(action_dir) if f.lower().endswith(('.avi', '.mp4', '.mov', '.mkv'))]
        num_videos = len(video_files)
        print(f"动作类别 '{action}' 下共有 {num_videos} 个视频文件。")
        
        for video_file in tqdm(video_files, desc=f"处理 '{action}' 类别的视频", leave=False):
            video_path = os.path.join(action_dir, video_file)
            video_id = os.path.splitext(video_file)[0]
            
            # 定义保存路径：output_root/动作类别/视频编号/
            video_output_dir = os.path.join(output_root, action, video_id)
            ensure_dir(video_output_dir)
            
            # 分割视频为帧并保存
            split_video_into_frames(video_path, video_output_dir, frame_rate)
    
    print("\n所有视频帧已成功分割并保存。")

def main():
    """
    主函数，配置加载参数并执行视频分割流程。
    """
    # 配置参数
    dataset_dir = "/datasets_active/KTH"                         # 替换为您的 KTH 数据集根目录路径
    frames_output_path = "/datasets_active/KTH_frames" # 替换为您希望保存帧的路径
    frame_rate = 25                                # 每秒保存的帧数，可以根据需要调整
    
    # 确保输出目录存在
    ensure_dir(frames_output_path)
    
    # 执行视频分割并保存帧
    split_videos_to_frames(dataset_dir, frames_output_path, frame_rate)

if __name__ == "__main__":
    main()