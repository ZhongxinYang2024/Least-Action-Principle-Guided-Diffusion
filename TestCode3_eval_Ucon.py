import datetime
import torch
from TestCode3 import marginal_prob_std,diffusion_coeff,ScoreNet_MLP
import functools
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
from tools.AutoDiff import compute_grad

def plot_matrix_rows(matrix):
    """
    绘制矩阵的每一行作为单独的子图
    
    参数:
    matrix -- 二维数组或列表，形状为(M, N)
    """
    M = len(matrix)
    N = len(matrix[0])
    
    # 创建子图布局（自动计算行列布局）
    cols = int(np.ceil(np.sqrt(M)))  # 列数
    rows = int(np.ceil(M / cols))    # 行数
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    fig.suptitle(f'Matrix Row Plots ({M} rows x {N} columns)', fontsize=16)
    
    # 如果只有一行子图，将axes转为二维数组便于统一处理
    if M == 1:
        axes = np.array([axes])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(-1, 1)
    
    # 遍历每一行数据
    for i in range(M):
        row = i // cols  # 子图行索引
        col = i % cols   # 子图列索引
        
        # 获取当前子图坐标轴
        ax = axes[row, col] if rows > 1 and cols > 1 else axes.flatten()[i]
        
        # 绘制行数据
        ax.plot(matrix[i], 'o-', linewidth=2, markersize=6)
        ax.set_title(f'Row {i}', fontsize=12)
        ax.set_xlabel('Column Index')
        ax.set_ylabel('Value')
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # 隐藏多余的空子图
    for j in range(M, rows * cols):
        row = j // cols
        col = j % cols
        if rows > 1 and cols > 1:
            axes[row, col].axis('off')
        else:
            axes.flatten()[j].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局
    plt.show()

import torch

def Compute_Saction(x, x_00, x_neg1, T):
    """
    计算轨迹的速度和动能
    
    参数:
    x: 中间时刻位置张量, 形状 (batch_size, num_steps, spatial_dim)
    x_00: 初始位置张量, 形状 (batch_size, spatial_dim)
    x_neg1: 终态位置张量, 形状 (batch_size, spatial_dim)
    T: 总时间
    
    返回:
    kinetic_energy: 动能张量, 形状 (batch_size, num_steps+2)
    v: 速度张量, 形状 (batch_size, num_steps+2, spatial_dim)
    """
    # 确保所有输入都是张量
    assert all(isinstance(t, torch.Tensor) for t in [x, x_00, x_neg1]), "所有输入必须是torch.Tensor"
    
    # 将初始和终态位置拼接到轨迹中
    full_x = torch.cat([
        x_00.unsqueeze(1),  # 添加时间步维度并拼接在开头
        x,
        x_neg1.unsqueeze(1)  # 添加时间步维度并拼接在结尾
    ], dim=1)
    
    # 计算总时间步数 (包括初始和终态)
    total_steps = full_x.shape[1]
    
    # 计算时间间隔 (总时间除以步数间隔)
    dt = T / (total_steps - 1)  # 时间步长
    
    # 计算速度 (中心差分法)
    v = torch.zeros_like(full_x)
    
    # 内部点使用中心差分
    if total_steps > 2:
        v[:, 1:-1] = (full_x[:, 2:] - full_x[:, :-2]) / (2 * dt)
    
    # 边界点使用单侧差分
    v[:, 0] = (full_x[:, 1] - full_x[:, 0]) / dt        # 起始点 (前向差分)
    v[:, -1] = (full_x[:, -1] - full_x[:, -2]) / dt     # 结束点 (后向差分)
    
    # 计算动能 (假设质量为1)
    Kenergy = 0.5 * v**2
    
    # Compute potential energy,
    Penergy = 0.5 * full_x **2
    
    Saction = torch.sum( (Kenergy - Penergy)*dt, dim=1)
    
    return Saction


def CondScore(x,T):
    Batch_size = x.shape[0]

    x00 = torch.ones((Batch_size,),dtype=torch.float32, device=x.device)*0.0
    x_neg1 = torch.ones((Batch_size, ), dtype=torch.float32, device=x.device)*0.0
    fn = functools.partial(Compute_Saction,x_00= x00, x_neg1 = x_neg1,T = T)
    Cgrad = compute_grad(fn, x)
    return Cgrad
    

if __name__ == '__main__':
    signal_to_noise_ratio = 0.16 #@param {'type':'number'}

    ## The number of sampling steps.
    num_steps =  500#@param {'type':'integer'}
    def pc_sampler(score_model, 
                marginal_prob_std,
                diffusion_coeff,
                batch_size=64, 
                num_steps=num_steps, 
                snr=signal_to_noise_ratio,                
                device='cuda',
                eps=1e-3):
        """
        使用预测-校正方法从基于分数的模型生成样本.

        Args:
            score_model: 时间依赖，基于分数的 PyTorch model.
            marginal_prob_std: A function that gives the standard deviation
            of the perturbation kernel.
            diffusion_coeff: A function that gives the diffusion coefficient 
            of the SDE.
            batch_size: 批次大小.
            num_steps: 采样步数， 等价于相当于离散时间步数.    
            device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
            eps: 数值稳定性的最小时间步长.
        
        Returns: 
            采样样本.
        """
        t = torch.ones(batch_size, device=device)
        init_x = torch.randn(batch_size, 128, device=device) * marginal_prob_std(t)[:, None]
        time_steps = np.linspace(1., eps, num_steps)
        step_size = time_steps[0] - time_steps[1]
        
        x = init_x
        with torch.no_grad():
            for time_step in time_steps:      
                batch_time_step = torch.ones(batch_size, device=device) * time_step
                
                # Predictor step (Euler-Maruyama) for discrete SDE
                g = diffusion_coeff(batch_time_step)
                x_mean = x + (g**2)[:, None] * score_model(x, batch_time_step) * step_size
                x = x_mean + torch.sqrt(g**2 * 2 * step_size)[:, None] * torch.randn_like(x)      
                
                for _ in range(2):
                    # Corrector step (Langevin MCMC)
                    grad = score_model(x, batch_time_step) 
                    # 逐样本计算梯度范数 [batch_size]
                    grad_norm_per_sample = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1)
                    noise_norm = np.sqrt(np.prod(x.shape[1:]))
                    # 逐样本计算步长 [batch_size]
                    langevin_step_size = 2 * (snr * noise_norm / grad_norm_per_sample)**2

                    # 确保步长形状匹配 [batch_size, 1, 1,...]
                    step_size_L = langevin_step_size.view(-1, *([1]*(grad.dim()-1)))

                    x_mean =  x + step_size_L * grad
                    x = x_mean + torch.sqrt(2 * step_size_L) * torch.randn_like(x)      

            # The last step does not include any noise
            return x_mean

    ## Load the pre-trained checkpoint from disk.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    sigma=2.5
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    diffusion_coeff_fn = functools.partial(diffusion_coeff,sigma=sigma)

    ckpt = torch.load('ckpt.pth', map_location=device)
    score_model = torch.nn.DataParallel(ScoreNet_MLP(marginal_prob_std=marginal_prob_std_fn))
    score_model = score_model.to(device)
    score_model.load_state_dict(ckpt)


    sample_batch_size = 16 #@param {'type':'integer'}
    sampler = pc_sampler #@param ['Euler_Maruyama_sampler', 'pc_sampler', 'ode_sampler'] {'type': 'raw'}

    print(datetime.datetime.now())
    ## Generate samples using the specified sampler.
    samples = sampler(score_model, 
                    marginal_prob_std_fn,
                    diffusion_coeff_fn, 
                    sample_batch_size, 
                    device=device)
    print(datetime.datetime.now())

    ## Sample visualization.

    plot_matrix_rows(samples.cpu().numpy())