import numpy as np

bin_size = 0.1  # 间隔
bins = np.arange(-1, 1 + bin_size, bin_size)  # 创建直方图的区间
counter = np.zeros(len(bins) - 1)  # 计数器的初始值为0

