import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
def DrawNonuniformGrid(X, Y):
    """
    绘制非均匀结构化网格的边缘
    
    参数:
    X: 二维数组，网格点的x坐标矩阵
    Y: 二维数组，网格点的y坐标矩阵
    """
    # 获取网格尺寸
    rows, cols = X.shape
    
    # 创建图形
    plt.figure(figsize=(10, 8))
    
    # 绘制水平线（每行）
    for i in range(rows):
        plt.plot(X[i, :], Y[i, :], 'b-', linewidth=0.8)
    
    # 绘制垂直线（每列）
    for j in range(cols):
        plt.plot(X[:, j], Y[:, j], 'b-', linewidth=0.8)
    
    #plt.scatter(X.flatten(), Y.flatten(), color='red', s=10, alpha=0.6)
    
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(False)
    plt.axis('equal')  # 保持纵横比相等
    
    # 显示图形
    plt.show()
    


class StructuredGrid:
    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.blocks = []
    
    def add_block(self, block_no: int, nx: int, ny: int, x: np.ndarray, y: np.ndarray):
        """add a grid block"""
        block = {
            'block_no': block_no,
            'nx': nx,
            'ny': ny,
            'x': x,  # shape: [nx, ny]
            'y': y   # shape: [nx, ny]
        }
        self.blocks.append(block)
    
    def write_grid_to_Fortranfile(self, filename: str):
        """write the grid to a Fortran-readable file"""
        with open(filename, 'w') as f:
            # line 1：numbers of blocks
            f.write(f"{self.num_blocks}\n")
            
            # line 2：NI, NJ for each block
            ni_nj_line = ""
            for block in self.blocks:
                ni_nj_line += f"  {block['nx']}  {block['ny']}"
            f.write(f"{ni_nj_line}\n")
            
            # line 3 and onwards: x and y coordinates for each block
            for block in self.blocks:
                nx, ny = block['nx'], block['ny']
                x_data = block['x']
                y_data = block['y']
                
                counts = 0
                # write x coord
                for j in range(ny):
                    for i in range(nx):
                        f.write(f"{x_data[i, j]:.16f} ");counts += 1
                        if counts %3 == 0:
                            counts = 0
                            f.write('\n')
                        
                
                for j in range(ny):
                    for i in range(nx):
                        f.write(f"{y_data[i, j]:.16f} ");counts += 1
                        if counts %3 == 0:
                            counts = 0
                            f.write('\n')
                    
    
    def print_grid_summary(self):
        """print a summary of the grid"""
        print(f"number of block: {self.num_blocks}")
        for i, block in enumerate(self.blocks):
            print(f"block {i+1}: NI={block['nx']}, NJ={block['ny']}, "
                  f"X range: [{block['x'].min():.3f}, {block['x'].max():.3f}], "
                  f"Y range: [{block['y'].min():.3f}, {block['y'].max():.3f}]")





# 使用示例
if __name__ == "__main__":
    def create_example_grid() -> StructuredGrid:
        """创建一个示例网格（用于测试）"""
        grid = StructuredGrid(num_blocks=2)
        
        # 第一个网格块：10x5 的矩形网格
        nx1, ny1 = 10, 5
        x1 = np.zeros((nx1, ny1))
        y1 = np.zeros((nx1, ny1))
        
        for i in range(nx1):
            for j in range(ny1):
                x1[i, j] = i * 0.1
                y1[i, j] = j * 0.2
        
        grid.add_block(block_no=1, nx=nx1, ny=ny1, x=x1, y=y1)
        
        # 第二个网格块：6x8 的弧形网格
        nx2, ny2 = 6, 8
        x2 = np.zeros((nx2, ny2))
        y2 = np.zeros((nx2, ny2))
        
        for i in range(nx2):
            for j in range(ny2):
                angle = i * np.pi / (nx2 - 1)
                radius = 1.0 + j * 0.1
                x2[i, j] = radius * np.cos(angle)
                y2[i, j] = radius * np.sin(angle)
        
        grid.add_block(block_no=2, nx=nx2, ny=ny2, x=x2, y=y2)
        
        return grid
        

    

