import datetime
import torch
from TestCode3 import marginal_prob_std,diffusion_coeff,ScoreNet_MLP
import functools
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
from tools.AutoDiff import compute_grad
from tools.Animation import animate_curves

def huber_loss(residual, delta=1.0):
    abs_res = torch.abs(residual)
    return torch.where(abs_res < delta, 0.5 * residual**2, delta * (abs_res - 0.5 * delta))

def log_cosh_loss(residual):
    return torch.log(torch.cosh(residual))

def plot_matrix_rows(tt,matrix):
    """
    绘制矩阵的每一行作为单独的子图
    
    参数:
    matrix -- 二维数组或列表，形状为(M, C, N)
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
        ax.plot(tt,matrix[i], 'o-', linewidth=2, markersize=6)
        #ax.plot(tt, np.sin( np.sqrt(3) * np.pi * tt)/np.sin(np.sqrt(4) * np.pi * 2),'--',color='red')
        ax.set_title(f'Row {i}', fontsize=12)
        ax.set_xlabel('Time')
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



def Compute_Saction(x, x_00, x_neg1, T):
    """
    计算轨迹的速度和动能
    
    参数:
    x: 中间时刻位置张量, 形状 (batch_size, num_steps, spatial_dim)
    x_00: 初始位置张量, 形状 (batch_size, spatial_dim)
    x_neg1: 终态位置张量, 形状 (batch_size, spatial_dim)
    T: 总时间
    
    返回:
    Saction: 作用量张量, 形状 (batch_size,)
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
    
    # 修复的关键：避免使用 torch.zeros_like 创建新张量
    # 而是使用 full_x 的差分直接计算速度
    
    # 计算前向差分：x_{t+1} - x_t
    forward_diff = full_x[:, 1:] - full_x[:, :-1]
    
    # 计算中心差分：x_{t+1} - x_{t-1}
    center_diff = full_x[:, 2:] - full_x[:, :-2]
    
    '''
    # 初始化速度张量
    v = torch.empty_like(full_x)
    
    # 边界点使用单侧差分
    v[:, 0] = forward_diff[:, 0] / dt  # 起始点 (前向差分)
    v[:, -1] = forward_diff[:, -1] / dt  # 结束点 (后向差分)
    
    # 内部点使用中心差分
    if total_steps > 2:
        v[:, 1:-1] = forward_diff[:,1:] / (1 * dt)
    
    # 计算动能 (假设质量为1)
    Kenergy = 0.5 * 1 * v ** 2
    
    # 计算势能
    Penergy = 0.5 * (3 * np.pi**2) * full_x ** 2
    '''
    
    v_c = forward_diff /dt
    x_c = (full_x[:, 1:] + full_x[:, :-1])/2
    
    Kenergy = 0.5 * 1 * v_c ** 2

    Penergy = 0.5 * (4 * np.pi**2) * x_c ** 2

    # 计算作用量 (拉格朗日量在时间上的积分)
    Saction = torch.sum( (Kenergy - Penergy) * dt, dim=1)
    
    return Saction

def Compute_PDEloss(x, x_00, x_neg1, T):
    """
    计算轨迹的速度和动能
    
    参数:
    x: 中间时刻位置张量, 形状 (batch_size, num_steps, spatial_dim)
    x_00: 初始位置张量, 形状 (batch_size, spatial_dim)
    x_neg1: 终态位置张量, 形状 (batch_size, spatial_dim)
    T: 总时间
    
    返回:
    Saction: 作用量张量, 形状 (batch_size,)
    """
    # 确保所有输入都是张量
    assert all(isinstance(t, torch.Tensor) for t in [x, x_00, x_neg1]), "所有输入必须是torch.Tensor"
    
    # 将初始和终态位置拼接到轨迹中
    full_x = x
    
    # 计算总时间步数 (包括初始和终态)
    total_steps = full_x.shape[1]
    
    # 计算时间间隔 (总时间除以步数间隔)
    dt = T / (total_steps - 1)  # 时间步长
    
    # 修复的关键：避免使用 torch.zeros_like 创建新张量
    # 而是使用 full_x 的差分直接计算速度
    
    # 计算前向差分：x_{t+1} - x_t
    forward_diff_dx = full_x[:, 1:] - full_x[:, :-1]
    
    # 计算中心差分：x_{t+1} - x_{t-1}
    center_diff_dx = full_x[:, 2:] - full_x[:, :-2]
    

    # 初始化速度张量
    v = torch.empty_like(full_x)*0.0
    
    # 边界点使用单侧差分
    v[:, 0] = forward_diff_dx[:, 0] / dt  # 起始点 (前向差分)
    v[:, -1] = forward_diff_dx[:, -1] / dt  # 结束点 (后向差分)
    
    # 内部点使用中心差分
    if total_steps > 2:
        v[:, 1:-1] = center_diff_dx / (2 * dt)
    

    # 计算前向差分：v_{t+1} - v_t
    forward_diff_dv = v[:, 1:] - v[:, :-1]
    
    # 计算中心差分：v_{t+1} - v_{t-1}
    center_diff_dv = v[:, 2:] - v[:, :-2]

    # 初始化加速度张量
    a = torch.empty_like(full_x) * 0.0
    
    # 边界点使用单侧差分
    a[:, 0] = forward_diff_dv[:, 0] / dt  # 起始点 (前向差分)
    a[:, -1] = forward_diff_dv[:, -1] / dt  # 结束点 (后向差分)
    
    # 内部点使用中心差分
    if total_steps > 2:
        a[:, 1:-1] = center_diff_dv/ (2 * dt)
    
    m = 1.0; k = 4* np.pi**2
    PDEloss_ = torch.sqrt( (m*a + k*full_x)**2 )

    BCloss_ =torch.sqrt( ((x[:,0] - 0.0)**2 + (x[:,-1] - 0.0)**2).unsqueeze(-1) )
    smoothness_ = (x[:,1:] - x[:,:-1]) ** 2
    # 计算作用量 (拉格朗日量在时间上的积分)
    PDEloss = torch.sum(PDEloss_, dim=1) + 1e3 * torch.sum(BCloss_, dim=1) #s+ 1.*torch.sum(smoothness_,dim=1)
    
    return PDEloss

def Get_BC(Batch_size,device):
    x_00 = torch.ones((Batch_size,),dtype=torch.float32, device=device)*0.0
    x_neg1 = torch.ones((Batch_size, ), dtype=torch.float32, device=device)*1.0
    return x_00, x_neg1

def CondScore(x,TotalTime):
    Batch_size = x.shape[0]

    x_00, x_neg1 = Get_BC(Batch_size, x.device)
    Cgrad = -compute_grad(Compute_PDEloss, x, x_00, x_neg1, TotalTime)
    return Cgrad
    

if __name__ == '__main__':
    signal_to_noise_ratio = 0.16 #@param {'type':'number'}
    T = 2
    ## The number of sampling steps.
    num_steps =  500#@param {'type':'integer'}
    def pc_sampler(score_model, 
                marginal_prob_std,
                diffusion_coeff,
                batch_size=64, 
                num_steps=num_steps, 
                snr=signal_to_noise_ratio,                
                device='cuda',
                eps=1e-3,
                T=2):
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
        time_steps = np.linspace(1, eps, num_steps)
        step_size = time_steps[0] - time_steps[1]
        
        x = init_x
        
        allcurves = [];
        for time_now in time_steps:      
            batch_time_now = torch.ones(batch_size, device=device) * time_now
                
            Cfactor = 0.0# 1 - time_now**2.5  #if time_now > 0.5 else 1.0
            # Predictor step (Euler-Maruyama) for discrete SDE
            
            grad_conditional = CondScore(x,T)
            with torch.no_grad():
                grad_unconditional =  score_model(x, batch_time_now)
                grad = (1. - Cfactor) * grad_unconditional + Cfactor * grad_conditional
                g = diffusion_coeff(batch_time_now)
                
                x_mean = x + 1. * (g**2)[:, None] * grad  * step_size
                x = x_mean  + torch.sqrt(g**2 * 2 * step_size)[:, None] * torch.randn_like(x)   
                   
                allcurves.append( (np.linspace(0,T,x.shape[1]),x[0].cpu().numpy()) )
            for _ in range(5):
                # Corrector step (Langevin MCMC)
                
                grad_conditional =  CondScore(x,T)
                with torch.no_grad():
                    grad_unconditional = score_model(x, batch_time_now)
                    grad = (1. - Cfactor) * grad_unconditional + Cfactor * grad_conditional
                    
                    # 逐样本计算梯度范数 [batch_size]
                    grad_norm_per_sample = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1)
                    noise_norm = np.sqrt(np.prod(x.shape[1:]))
                    
                    # 逐样本计算步长 [batch_size]
                    langevin_step_size = 2 * (snr * noise_norm / grad_norm_per_sample)**2

                    # 确保步长形状匹配 [batch_size, 1, 1,...]
                    step_size_L = langevin_step_size.view(-1, *([1]*(grad.dim()-1)))

                    x_mean =  x + step_size_L * grad
                    x = x_mean  + torch.sqrt(2 * step_size_L) * torch.randn_like(x)      

            # The last step does not include any noise
        return x_mean,allcurves

    ## Load the pre-trained checkpoint from disk.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    sigma=25.00
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
    samples,allcurves = sampler(score_model, 
                    marginal_prob_std_fn,
                    diffusion_coeff_fn, 
                    sample_batch_size, 
                    device=device,
                    T=T)
    
    x = samples
    #allcurves = []
    #x = torch.randn(sample_batch_size, 128, device=device) 
    for _ in range(5000):
        # Corrector step (Langevin MCMC)
        
        grad_conditional =  CondScore(x,T)
        with torch.no_grad():
            grad = grad_conditional#.clamp(min=-1e1, max=1e1) 
            
            # 逐样本计算步长 [batch_size]
            langevin_step_size = torch.ones((x.shape[0],))  * 1e-4

            # 确保步长形状匹配 [batch_size, 1, 1,...]
            step_size_L = langevin_step_size.view(-1, *([1]*(grad.dim()-1)))

            x_mean =  x + step_size_L * grad
            x = x_mean #+ torch.sqrt(2 * step_size_L) * torch.randn_like(x)   
            allcurves.append( (np.linspace(0,T,x.shape[1]),x[0].cpu().numpy()) )
            
            
    samples = x;          
    print(datetime.datetime.now())

    ## Sample visualization.

    plot_matrix_rows(np.linspace(0,T,samples.shape[1]),samples.cpu().numpy())
    
    animate_curves(allcurves, interval=1, title="Demostration")
    
    x_00, x_neg1 = Get_BC(sample_batch_size,samples.device)
    Saction = Compute_Saction(samples, x_00, x_neg1, T)
    print(Saction)