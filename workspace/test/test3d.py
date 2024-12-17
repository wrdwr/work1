import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 假设 `new_matrix` 是插值后的 3x3x3 矩阵
new_matrix = np.random.random((3, 3, 3))  # 用随机数据替代

# 获取矩阵的 x, y, z 坐标
x, y, z = np.indices(new_matrix.shape)

# 展平矩阵和坐标
values = new_matrix.flatten()
x_coords = x.flatten()
y_coords = y.flatten()
z_coords = z.flatten()

# 创建三维图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制每个 (x, y, z) 对应的点的直方图
hist, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=new_matrix.shape[0])

# 使用 bar3d 创建三维直方图
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# 设置直方图的大小
dx = dy = 0.5 * np.ones_like(zpos)
dz = hist.ravel()

# 生成 3D 直方图
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

# 添加标签
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Value')

plt.show()
