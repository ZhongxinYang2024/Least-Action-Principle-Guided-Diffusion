import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_curves(curve_list, interval=100, title="Curve Animation"):
    """
    动态放映曲线列表中的曲线
    
    参数:
    curve_list -- 曲线列表，每个元素是(x, y)元组或数组
    interval -- 每条曲线显示时间(毫秒)，默认1000ms
    title -- 图表标题，默认"曲线族动态放映"
    """
    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    
    # 自动确定合适的坐标轴范围
    all_x = np.concatenate([curve[0] for curve in curve_list])
    all_y = np.concatenate([curve[1] for curve in curve_list])
    
    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)
    
    # 扩展5%的边界
    x_padding = (x_max - x_min) * 0.05
    y_padding = (y_max - y_min) * 0.05
    
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    # 初始化空曲线
    line, = ax.plot([], [], lw=2, color='blue')
    
    # 添加文本标注显示当前曲线序号
    curve_index_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                              fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    # 动画更新函数
    def update(frame):
        """根据帧号更新曲线"""
        curve_index = frame % len(curve_list)
        x, y = curve_list[curve_index]
        line.set_data(x, y)
        curve_index_text.set_text(f'Curve {curve_index + 1}/{len(curve_list)}')
        return line, curve_index_text
    
    # 创建动画
    animation = FuncAnimation(
        fig, 
        update, 
        frames=len(curve_list),  # 帧数等于曲线数量
        interval=interval,        # 每帧间隔时间
        blit=True,
        repeat=True               # 循环播放
    )
    
    plt.tight_layout()
    plt.show()
    return animation

# 示例用法
if __name__ == "__main__":
    # 创建示例曲线列表
    x = np.linspace(-5, 5, 200)
    
    curves = [
        (x, np.sin(x)),          # 正弦波
        (x, np.cos(x)),           # 余弦波
        (x, np.sinc(x)),          # sinc函数
        (x, np.exp(-x**2)),       # 高斯函数
        (x, np.tanh(x)),          # 双曲正切
        (x, np.sin(x) * np.cos(2*x))  # 组合波
    ]
    
    # 放映曲线，每条显示1.5秒
    animate_curves(curves, interval=1000, title="Demostration")