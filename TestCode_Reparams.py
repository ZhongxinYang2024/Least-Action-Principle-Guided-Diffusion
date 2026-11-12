import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# 创建3D图形
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 生成曲线数据 (螺旋线示例)
t = np.linspace(0, 20, 1000)  # 参数t
x = np.cos(t)                  # x坐标
y = np.sin(t)                  # y坐标
z = t/2                        # z坐标（第三维度）

# 创建颜色映射 (从浅蓝到深蓝)
cmap = ListedColormap(plt.cm.Blues(np.linspace(0.3, 0.9, 256)))

# 将点分段用于颜色映射
points = np.array([x, y, z]).T.reshape(-1, 1, 3)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# 创建Line3DCollection对象 - 关键修正
lc = Line3DCollection(segments, cmap=cmap, linewidth=2)
lc.set_array(z)  # 根据z值设置颜色

# 将线段添加到3D坐标轴
ax.add_collection(lc)

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Curve with Color Gradient')

# 设置坐标轴范围以确保所有点都可见
ax.set_xlim([np.min(x), np.max(x)])
ax.set_ylim([np.min(y), np.max(y)])
ax.set_zlim([np.min(z), np.max(z)])

# 设置视角
ax.view_init(elev=30, azim=-60)

# 添加颜色条
cbar = fig.colorbar(lc, ax=ax, pad=0.1)
cbar.set_label('Z-value (color depth)')

plt.tight_layout()
plt.show()