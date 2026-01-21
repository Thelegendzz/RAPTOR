import os
from PIL import Image
import torch
from torchvision import transforms
import numpy as np

def get_grayscale_pixel_values(image_path):
    """
    加载图片，转换为灰度图像，并提取每个像素的数值。

    参数:
    - image_path: str, 图片的文件路径

    返回:
    - pixel_values: numpy.ndarray, 灰度图像的像素值数组，形状为 [H, W]
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片路径不存在: {image_path}")

    # 打开图片
    gray_img = Image.open(image_path).convert('L')  # 确保是RGB格式

    # 转换为 NumPy 数组并去除单通道维度
    pixel_values = np.array(gray_img)  # 形状: [H, W]

    max_value = np.max(pixel_values)
    min_value = np.min(pixel_values)
    return pixel_values, max_value, min_value

def main():
    # 指定特定图片的路径列表
    specific_image_paths = [
        '/datasets_active/MMNIST/val/16/000001.jpg',
        '/datasets_active/MMNIST/val/16/000002.jpg',
        '/datasets_active/MMNIST/val/16/000003.jpg',
        # 添加更多图片路径
    ]

    # 遍历每个图片路径并提取像素值
    for image_path in specific_image_paths:
        try:
            pixel_values, max_value, min_value = get_grayscale_pixel_values(image_path)
            print(f"图片: {image_path}")
            print(f"灰度像素值 (形状: {pixel_values.shape}):\n{max_value}\n{min_value}\n{pixel_values}\n")
            
            # 如果需要将像素值保存为文件（例如CSV），可以取消注释以下代码
            # save_path = image_path.replace('.png', '_grayscale_pixels.csv')
            # np.savetxt(save_path, pixel_values, delimiter=",")
            # print(f"像素值已保存到: {save_path}\n")
            
            # 如果需要可视化灰度图像的像素分布，可以使用以下代码
            # import matplotlib.pyplot as plt
            # plt.imshow(pixel_values, cmap='gray')
            # plt.title(f"Grayscale Image: {os.path.basename(image_path)}")
            # plt.show()
            
        except Exception as e:
            print(f"处理图片 {image_path} 时出错: {e}")

    print("特定图片的灰度像素值提取完成。")

if __name__ == "__main__":
    main()