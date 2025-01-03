import numpy as np
import matplotlib.pyplot as plt

def matrix_histogram(matrix, bins=10):
    """
    生成输入3维矩阵的统计直方图，并输出每个范围的内容。

    参数：
        matrix (np.ndarray): 输入的3维矩阵。
        bins (int): 直方图的区间数量。

    返回：
        dict: 每个区间对应的数值统计结果。
    """
    # 将矩阵展开为一维数组
    data = matrix.flatten()

    # 计算直方图
    hist, bin_edges = np.histogram(data, bins=bins)

    # 输出每个范围的内容
    range_values = {}
    for i in range(len(bin_edges) - 1):
        range_key = f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})"
        range_values[range_key] = data[(data >= bin_edges[i]) & (data < bin_edges[i+1])]

    # 绘制直方图
    plt.hist(data, bins=bins, edgecolor='black', alpha=0.7)
    plt.title("3D Matrix Value Histogram")
    plt.xlabel("Value Range")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    return range_values

# 示例3维矩阵
matrix = np.array([
    [
        [1, 50, 1],
        [1, 1, 1],
        [1, 1, 1]
    ],
    [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ],
    [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
])
matrix_histogram(matrix, bins=5)