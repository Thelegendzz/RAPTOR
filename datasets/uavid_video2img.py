import os

import cv2


def video_to_frames(video_path, output_dir):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    count = 0

    # 读取帧
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 如果没有帧了，就结束循环

        # 构建输出文件名（六位数字，填充零）
        frame_filename = os.path.join(output_dir, f"{count:06d}.jpg")
        cv2.imwrite(frame_filename, frame)  # 写入帧到文件
        count += 1

    cap.release()

# 使用示例
# video_path = '/datasets_active/UAVID/uavid_train/seq35/images.mp4'
# output_dir = '/datasets_active/UAVID-images/uavid_train/seq35'
path="/datasets_active/UAVID/uavid_test/"
output_path="/datasets_active/UAVID-images/uavid_test/"
for i in os.listdir(path):
    print(i)
    video_path = os.path.join(path, i,'images.mp4')
    output_dir = os.path.join(output_path, i)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_to_frames(video_path, output_dir)
    print('finish')