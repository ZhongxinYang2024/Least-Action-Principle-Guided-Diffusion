#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于PyTorch的二维波动方程PINN求解器
来源：https://blog.csdn.net/qq_58509609/article/details/138239469
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免GUI问题
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn as nn
from pyDOE import lhs
import os

# 自动更换CPU,GPU，并设置浮点型位数
default_device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32
torch.set_default_dtype(dtype)
print(f"使用设备: {default_device}")

# 设置随机数种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(888888)

# 转换张量函数，存储损失的时候用到
def Tran_fun(tensor):
    a = []
    for i in tensor:
        i = i.detach().cpu().numpy()
        a.append(i)
    return np.array(a)

# 设置一种采样策略，这里用到lhs函数
def LHS_Sampling(N, be, en):
    return be + (en - be) * lhs(3, N)

# 求导函数
def gradients(u, x, order=1):
    u.requires_grad_(True)
    x.requires_grad_(True)
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), 
                                 create_graph=True, only_inputs=True)[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)

# PINN网络结构
class PINN(nn.Module):
    def __init__(self, in_features, out_features, num_neurons, activation=torch.sin):
        super(PINN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_neurons = num_neurons
        self.act_func = activation
        
        self.block1_layer1 = nn.Linear(self.in_features, self.num_neurons)
        self.block1_layer2 = nn.Linear(self.num_neurons, self.num_neurons)
        self.block1 = [self.block1_layer1, self.block1_layer2]
        
        self.block2_layer1 = nn.Linear(self.in_features + self.num_neurons, self.num_neurons)
        self.block2_layer2 = nn.Linear(self.num_neurons, self.num_neurons)
        self.block2 = [self.block2_layer1, self.block2_layer2]
        
        self.layer_after_block = nn.Linear(self.num_neurons + self.in_features, self.num_neurons)
        self.layer_output = nn.Linear(self.num_neurons, self.out_features)

    def forward(self, x):
        x_temp = x
        for dense in self.block1:
            x_temp = self.act_func(dense(x_temp))
        x_temp = torch.cat([x_temp, x], dim=-1)
        
        for dense in self.block2:
            x_temp = self.act_func(dense(x_temp))
        x_temp = torch.cat([x_temp, x], dim=-1)
        
        x_temp = self.act_func(self.layer_after_block(x_temp))
        x_temp = self.layer_output(x_temp)
        return x_temp

# 全局参数
c = 2
len_x = 4 * torch.pi
len_y = 4 * torch.pi
len_t = torch.pi / (1/8)**(1/2)

# 创建输出目录
os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/wave_fields", exist_ok=True)
os.makedirs("outputs/loss_plots", exist_ok=True)

# 损失函数，网络，优化器对象化
loss = torch.nn.MSELoss()
loss = loss.to(device=default_device)
pinn = PINN(3, 1, 20)  # 增加神经元数量以提高精度
pinn = pinn.to(device=default_device)
opt = torch.optim.Adam(pinn.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5000, gamma=0.9)

# PDE损失、边界条件损失、初始条件损失的设置
def interior_loss(Mat):
    # 内点
    Mat = torch.tensor(Mat, dtype=dtype, device=default_device)
    x = Mat[:, 0:1].float()
    y = Mat[:, 1:2].float()
    t = Mat[:, 2:3].float()
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)
    
    u = pinn(torch.cat([t, x, y], dim=1))
    cond1 = gradients(u, t, order=2)
    cond2 = c * (gradients(u, x, order=2) + gradients(u, y, order=2))
    return loss(cond1, cond2)

def left_loss(Mat):
    # 左边界点
    Mat = torch.tensor(Mat, dtype=dtype, device=default_device)
    Mat[:, 0:1] = torch.zeros_like(Mat[:, 0:1])
    x = Mat[:, 0:1].float()
    y = Mat[:, 1:2].float()
    t = Mat[:, 2:3].float()
    zero = torch.zeros_like(t).float()
    
    u = pinn(torch.cat([t, x, y], dim=1))
    return loss(u, zero)

def right_loss(Mat):
    # 右边界点
    Mat = torch.tensor(Mat, dtype=dtype, device=default_device)
    Mat[:, 0:1] = torch.ones_like(Mat[:, 0:1]) * torch.pi * 4
    x = Mat[:, 0:1].float()
    y = Mat[:, 1:2].float()
    t = Mat[:, 2:3].float()
    zero = torch.zeros_like(t).float()
    
    u = pinn(torch.cat([t, x, y], dim=1))
    return loss(u, zero)

def down_loss(Mat):
    # 下边界点
    Mat = torch.tensor(Mat, dtype=dtype, device=default_device)
    Mat[:, 1:2] = torch.zeros_like(Mat[:, 1:2])
    x = Mat[:, 0:1].float()
    y = Mat[:, 1:2].float()
    t = Mat[:, 2:3].float()
    zero = torch.zeros_like(t).float()
    
    u = pinn(torch.cat([t, x, y], dim=1))
    return loss(u, zero)

def up_loss(Mat):
    # 上边界点
    Mat = torch.tensor(Mat, dtype=dtype, device=default_device)
    Mat[:, 1:2] = torch.ones_like(Mat[:, 1:2]) * torch.pi * 4
    x = Mat[:, 0:1].float()
    y = Mat[:, 1:2].float()
    t = Mat[:, 2:3].float()
    zero = torch.zeros_like(t).float()
    
    u = pinn(torch.cat([t, x, y], dim=1))
    return loss(u, zero)

def ini1_loss(Mat):
    # 初始条件1
    Mat = torch.tensor(Mat, dtype=dtype, device=default_device)
    Mat[:, 2:3] = torch.zeros_like(Mat[:, 2:3])
    x = Mat[:, 0:1].float()
    y = Mat[:, 1:2].float()
    t = Mat[:, 2:3].float()
    
    u = pinn(torch.cat([t, x, y], dim=1))
    target = 2 * torch.sin(x/4) * torch.sin(y/4)
    return loss(u, target)

def ini2_loss(Mat):
    # 初始条件2
    Mat = torch.tensor(Mat, dtype=dtype, device=default_device)
    Mat[:, 2:3] = torch.zeros_like(Mat[:, 2:3])
    x = Mat[:, 0:1].float()
    y = Mat[:, 1:2].float()
    t = Mat[:, 2:3].float()
    t.requires_grad_(True)
    
    u = pinn(torch.cat([t, x, y], dim=1))
    target = 2 * torch.sin(x/4) * torch.sin(y/4)
    du_dt = gradients(u, t, order=1)
    return loss(du_dt, torch.zeros_like(du_dt))

# 计算预测波场与绘图函数的设置
class Output_Fun():
    def __init__(self, t_b, t_e, t_num, x_b, x_e, x_num, y_b, y_e, y_num, lamda):
        self.t_b = t_b
        self.t_e = t_e
        self.t_num = t_num
        self.x_b = x_b
        self.x_e = x_e
        self.x_num = x_num
        self.y_b = y_b
        self.y_e = y_e
        self.y_num = y_num
        self.lamda = lamda

    def U_Pred(self, model):
        tc = torch.linspace(self.t_b, self.t_e, self.t_num)
        xc = torch.linspace(self.x_b, self.x_e, self.x_num)
        yc = torch.linspace(self.y_b, self.y_e, self.y_num)
        tm, xm, ym = torch.meshgrid(tc, xc, yc, indexing='ij')
        xx = xm.reshape(-1, 1)
        yy = ym.reshape(-1, 1)
        tt = tm.reshape(-1, 1)
        xy = torch.cat([tt, xx, yy], dim=1)
        
        model.eval()
        with torch.no_grad():
            u_pred = model(xy.to(default_device))
        return u_pred.cpu()

    def U_True(self):
        tc = torch.linspace(self.t_b, self.t_e, self.t_num)
        xc = torch.linspace(self.x_b, self.x_e, self.x_num)
        yc = torch.linspace(self.y_b, self.y_e, self.y_num)
        tm, xm, ym = torch.meshgrid(tc, xc, yc, indexing='ij')
        xx = xm.reshape(-1, 1)
        yy = ym.reshape(-1, 1)
        tt = tm.reshape(-1, 1)
        
        u_true = torch.sin(torch.pi * xx / self.x_e) * torch.sin(torch.pi * yy / self.y_e) * \
                 2 * (torch.cos(self.lamda * tt) + torch.sin(self.lamda * tt))
        return u_true

    def U_Draw(self, true_mat, pred_mat):
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        x = torch.linspace(self.x_b, self.x_e, self.x_num)
        y = torch.linspace(self.y_b, self.y_e, self.y_num)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        
        true_mat = true_mat.detach()
        pred_mat = pred_mat.detach()
        res_mat = true_mat - pred_mat
        
        lenspace = int(self.x_num * self.y_num)
        
        # 绘制预测波场
        for i in range(0, min(self.t_num, 5)):  # 只绘制前5个时间步以节省时间
            u_i = pred_mat[i*lenspace:(i+1)*lenspace].reshape((self.x_num, self.y_num))
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            surf = ax.pcolor(X.numpy(), Y.numpy(), u_i.numpy())
            cb = fig.colorbar(surf, shrink=0.6, aspect=20)
            cb.ax.tick_params(labelsize=12)
            cb.set_label('Wave Field Value', fontsize=14)
            ax.set_xlabel('X', fontsize=14)
            ax.set_ylabel('Y', fontsize=14)
            t = round((self.t_e - self.t_b) * i / (self.t_num - 1), 3)
            ax.set_title(f"Predicted Wave Field at t={t}s", fontsize=16)
            ax.tick_params(labelsize=12)
            plt.tight_layout()
            plt.savefig(f"outputs/wave_fields/predicted_t_{i}.png", dpi=150, bbox_inches='tight')
            plt.close()

        # 绘制真实波场
        for i in range(0, min(self.t_num, 5)):
            u_i = true_mat[i*lenspace:(i+1)*lenspace].reshape((self.x_num, self.y_num))
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            surf = ax.pcolor(X.numpy(), Y.numpy(), u_i.numpy())
            cb = fig.colorbar(surf, shrink=0.6, aspect=20)
            cb.ax.tick_params(labelsize=12)
            cb.set_label('Wave Field Value', fontsize=14)
            ax.set_xlabel('X', fontsize=14)
            ax.set_ylabel('Y', fontsize=14)
            t = round((self.t_e - self.t_b) * i / (self.t_num - 1), 3)
            ax.set_title(f"True Wave Field at t={t}s", fontsize=16)
            ax.tick_params(labelsize=12)
            plt.tight_layout()
            plt.savefig(f"outputs/wave_fields/true_t_{i}.png", dpi=150, bbox_inches='tight')
            plt.close()

        # 绘制误差波场
        for i in range(0, min(self.t_num, 5)):
            u_i = res_mat[i*lenspace:(i+1)*lenspace].reshape((self.x_num, self.y_num))
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            surf = ax.pcolor(X.numpy(), Y.numpy(), u_i.numpy())
            cb = fig.colorbar(surf, shrink=0.6, aspect=20)
            cb.ax.tick_params(labelsize=12)
            cb.set_label('Error Value', fontsize=14)
            ax.set_xlabel('X', fontsize=14)
            ax.set_ylabel('Y', fontsize=14)
            t = round((self.t_e - self.t_b) * i / (self.t_num - 1), 3)
            ax.set_title(f"Error Field at t={t}s", fontsize=16)
            ax.tick_params(labelsize=12)
            plt.tight_layout()
            plt.savefig(f"outputs/wave_fields/error_t_{i}.png", dpi=150, bbox_inches='tight')
            plt.close()

# 参数设置
c = 2
len_x = 4 * torch.pi
len_y = 4 * torch.pi
len_t = torch.pi / (1/8)**(1/2)

t_b, t_e, t_num = 0.0, torch.pi / (1/8)**(1/2), 50  # 减少时间步数以加快计算
x_b, x_e, x_num = 0.0, 4 * torch.pi, 50
y_b, y_e, y_num = 0.0, 4 * torch.pi, 50
lamda = c * (1/8)**(1/2)

# 范围设置
t_range = [t_b, t_e]
x_range = [x_b, x_e]
y_range = [y_b, y_e]
lb = np.asarray([x_range[0], y_range[0], t_range[0]])
ub = np.asarray([x_range[1], y_range[1], t_range[1]])

# 采样点
print("正在生成采样点...")
xyt_in = LHS_Sampling(1000, lb, ub)
xyt_left = LHS_Sampling(100, lb, ub)
xyt_right = LHS_Sampling(100, lb, ub)
xyt_down = LHS_Sampling(100, lb, ub)
xyt_up = LHS_Sampling(100, lb, ub)
xyt_ini1 = LHS_Sampling(100, lb, ub)
xyt_ini2 = LHS_Sampling(100, lb, ub)

# 训练参数
epochs = 1000  # 减少训练轮次以加快演示
loss_i, loss_le, loss_ri, loss_d, loss_u, loss_i1, loss_i2, loss_all = [], [], [], [], [], [], [], []

print("开始训练PINN网络...")
# 训练循环
for i in range(epochs):
    opt.zero_grad()
    
    a1 = interior_loss(xyt_in)
    a2 = left_loss(xyt_left)
    a3 = right_loss(xyt_right)
    a4 = down_loss(xyt_down)
    a5 = up_loss(xyt_up)
    a6 = ini1_loss(xyt_ini1)
    a7 = ini2_loss(xyt_ini2)
    
    l = a1 + a2 + a3 + a4 + a5 + a6 + a7
    
    loss_i.append(a1.item())
    loss_le.append(a2.item())
    loss_ri.append(a3.item())
    loss_d.append(a4.item())
    loss_u.append(a5.item())
    loss_i1.append(a6.item())
    loss_i2.append(a7.item())
    loss_all.append(l.item())
    
    l.backward()
    opt.step()
    
    if i % 100 == 0:
        print(f"训练轮次: {i}, 总损失: {l.item():.6f}")

# 绘制损失函数图像
print("正在绘制损失函数图像...")
plt.figure(figsize=(12, 8))
x = np.linspace(1, epochs, epochs)
plt.semilogy(x, loss_all, 'b-', linewidth=2, label='Total Loss')
plt.xlabel('Training Epochs', fontsize=14)
plt.ylabel('Loss (log scale)', fontsize=14)
plt.title('PINN Training Loss', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("outputs/loss_plots/training_loss.png", dpi=300, bbox_inches='tight')
plt.close()

# 保存模型
print("正在保存模型...")
torch.save(pinn, "outputs/wave_model.pth")

# 求解方程并绘制结果
print("正在计算预测结果并绘制图像...")
sol = Output_Fun(t_b, t_e, t_num, x_b, x_e, x_num, y_b, y_e, y_num, lamda)
mat1 = sol.U_Pred(pinn)
mat2 = sol.U_True()
sol.U_Draw(mat2, mat1)

print("所有计算完成！")
print("结果保存在 outputs/ 目录下：")
print("- 模型文件: outputs/wave_model.pth")
print("- 损失函数图像: outputs/loss_plots/training_loss.png")
print("- 波场图像: outputs/wave_fields/ 目录下")