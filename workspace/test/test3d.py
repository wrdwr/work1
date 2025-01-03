import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.animation import FuncAnimation
from scipy.ndimage import label

def keep_largest_region(matrix):
    labeled_matrix, num_features = label(matrix)

    # 如果没有连通区域，直接返回全零矩阵
    if num_features == 0:
        return np.zeros_like(matrix, dtype=int)

    # 找到最大区域（直接计算每个区域的大小）
    region_sizes = np.bincount(labeled_matrix.ravel())
    region_sizes[0] = 0  # 忽略背景区域（标签为0）
    max_region_label = np.argmax(region_sizes)

    # 直接构造仅包含最大区域的矩阵
    return (labeled_matrix == max_region_label).astype(int)

def interpolate_3d_torch(matrix, new_shape):
    # 将 numpy 转为 torch 张量，并添加批量和通道维度（维度顺序：batch_size, channels, depth, height, width）
    matrix_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # 计算每个维度的 scale_factor
    old_shape = matrix.shape
    scale_factors = [n / o for n, o in zip(new_shape, old_shape)]

    # 使用 PyTorch 的三线性插值
    new_tensor = F.interpolate(matrix_tensor, scale_factor=scale_factors, mode='trilinear', align_corners=True)

    # 移除批量和通道维度，返回 numpy 数组
    return new_tensor.squeeze().numpy()

def matrix_look(data):
    fig, ax = plt.subplots()

    # 设置imshow
    im = ax.imshow(data[0], cmap='gray')
    plt.colorbar(im)

    # 更新函数
    def update(frame):
        im.set_array(data[frame])
        ax.set_title(f'Slice {frame}')
        return [im]

    # 创建动画
    ani = FuncAnimation(fig, update, frames=data.shape[0], interval=200, blit=True)

    # 显示动画
    plt.show()

    return 0

def matrix_look_one(data,slice_index):  # 选择切片索引
    plt.imshow(data[slice_index], cmap='gray')  # 使用'gray' colormap，或根据需要选择其他
    plt.colorbar()
    plt.title(f'Slice {slice_index}')
    plt.show()

    return 0

def matrix_look_one_2d(data):  # 选择切片索引
    plt.imshow(data, cmap='gray')  # 使用'gray' colormap，或根据需要选择其他
    plt.colorbar()
    plt.show()

    return 0
