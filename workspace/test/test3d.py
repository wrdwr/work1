import numpy as np
import matplotlib.pyplot as plt
from setuptools.sandbox import save_path


def shape_3d(new_matrix):
    x, y, z = np.indices(new_matrix.shape)

    # 展平矩阵和坐标
    values = new_matrix.flatten()
    z_coords = x.flatten()
    y_coords = y.flatten()
    x_coords = z.flatten()

    # 创建三维图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    zpos = z_coords  # 条形图的底部位置

    # 设置条形图的大小
    dx = dy = 0.5 * np.ones_like(zpos)  # 每个条的宽度
    dz = (values/np.max(new_matrix)*0.7)# 条的高度，即对应的矩阵值

    # 生成 3D 直方图
    ax.bar3d(x_coords, y_coords, zpos, dx, dy, dz)

    # 添加标签
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # 移除坐标轴的刻度和标签
    ax.set_xticks([])  # 不显示 X 轴的刻度
    ax.set_yticks([])  # 不显示 Y 轴的刻度
    ax.set_zticks([])  # 不显示 Z 轴的刻度


    # 设置坐标轴的范围（可选）
    ax.set_xlim([0, new_matrix.shape[0]])
    ax.set_ylim([0, new_matrix.shape[1]])
    ax.set_zlim([0, new_matrix.shape[2]])

    # 设置坐标轴比例
    ax.set_box_aspect([1, 1, 1.8])

    # 显示图形
    plt.show()

