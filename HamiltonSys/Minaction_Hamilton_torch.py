#%% 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # 把根目录塞进 sys.path

## 
from tools.AutoDiff import compute_grad
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
#%%
def generate_quadratic_vector(n):
    if n < 2:
        raise ValueError("n must be at least 2 to have a maximum value of 1")
    x = torch.linspace(0, n, n + 1)
    a = -4.0 / (n ** 2)
    h = n / 2.0
    vector = a * (x - h) ** 2 + 1
    
    vector[0] = 0.0
    vector[-1] = 0.0
    return vector
    

from torch.optim import Optimizer

class PureGradientDescent(Optimizer):
    """
    完全纯粹的梯度下降优化器
    不使用动量、自适应学习率或任何其他技巧
    """
    
    def __init__(self, params, lr):
        """
        初始化纯梯度下降优化器
        
        参数:
            params (iterable): 需要优化的参数
            lr (float): 学习率
        """
        if lr < 0.0:
            raise ValueError(f"无效的学习率: {lr}")
            
        defaults = dict(lr=lr)
        super(PureGradientDescent, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """
        执行单次优化步骤
        
        参数:
            closure (callable, optional): 重新计算损失的闭包函数
            
        返回:
            loss (float, optional): 如果提供了闭包函数，则返回损失值
        """
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            learning_rate = group['lr']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                # 纯粹的梯度下降更新规则: θ = θ - η * ∇J(θ)
                p.data.add_(-learning_rate, p.grad.data)
                
        return loss
    

def Plot_phase3D(x, y, z, labels, ax3d, fig, color=None, cmap=None, linewidth=2, auto_scale=True ,margin_ratio=0.1):
    """
    绘制三维相图，曲线颜色可自定义
    
    参数:
    x, y, z: 三个维度的数据
    labels: 坐标轴标签列表 [x_label, y_label, z_label]
    ax3d: 3D坐标轴对象
    fig: 图形对象
    color: 曲线颜色 (可选)
        - 字符串: 如 'blue', 'red', '#FF00FF'
        - RGB元组: 如 (0.2, 0.4, 0.8)
        - None: 使用z值映射颜色 (默认)
    cmap: 当color为None时使用的颜色映射 (默认为蓝紫色渐变)
    linewidth: 曲线宽度 (默认2)
    """
    # 准备线段数据
    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # 处理颜色参数
    if color is not None:
        # 使用固定颜色
        lc = Line3DCollection(segments, color=color, linewidth=linewidth)
    else:
        # 使用z值映射颜色
        if cmap is None:
            # 创建蓝紫色渐变作为默认颜色映射
            cmap = ListedColormap(plt.cm.Blues(np.linspace(0.3, 0.9, 256)))
        
        lc = Line3DCollection(segments, cmap=cmap, linewidth=linewidth)
        lc.set_array(z)  # 根据z值设置颜色
    
    # 将线段添加到3D坐标轴
    ax3d.add_collection(lc)
    
    # 设置坐标轴标签
    ax3d.set_xlabel(labels[0])
    ax3d.set_ylabel(labels[1])
    ax3d.set_zlabel(labels[2])
    ax3d.set_title('Phase Graph')
    
    if auto_scale:
        # 计算数据范围
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        z_min, z_max = np.min(z), np.max(z)
        
        # 计算数据范围大小
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        
        # 添加比例自适应边距
        margin_x = margin_ratio * x_range if x_range > 0 else 0.1
        margin_y = margin_ratio * y_range if y_range > 0 else 0.1
        margin_z = margin_ratio * z_range if z_range > 0 else 0.1
        
        # 设置坐标轴范围（含自适应边距）
        ax3d.set_xlim(x_min - margin_x, x_max + margin_x)
        ax3d.set_ylim(y_min - margin_y, y_max + margin_y)
        ax3d.set_zlim(z_min - margin_z, z_max + margin_z)
        

    ax3d.set_zlim([np.min(z), np.max(z)])
    
    # 设置视角
    ax3d.view_init(elev=30, azim=-60)
    
    # 添加颜色条（仅当使用颜色映射时）
    if color is None:
        cbar = fig.colorbar(lc, ax=ax3d, pad=0.1)
        cbar.set_label('time')
    
    return lc  # 返回线段集合对象以便进一步定制
    


class PhySystem:
    def __init__(self, init_position, init_momentum, t_end, n_split, IF_dissipative=False, device='cpu'):
        self.device = device
        
        initial_pos = np.array(init_position) 
        initial_mom = np.array(init_momentum)
        self.num_of_obj, self.dimensionality = np.array(init_position).shape
        
        self.t_end = t_end # simulated time
        self.n_split     = n_split # split t_end into n parts
        
        self.tt = np.linspace(0, self.t_end, self.n_split+1)
        self.dt = self.tt[1] - self.tt[0]
        
        # create store array [obj, dim, timesteps]
        self.pos = np.zeros((self.num_of_obj, self.dimensionality, n_split + 1))
        self.mom = np.zeros((self.num_of_obj, self.dimensionality, n_split + 1))
        
        # set inital state
        self.pos[:, :, 0] = initial_pos
        self.mom[:, :, 0] = initial_mom

        self.IF_dissipative = IF_dissipative
        
    def _Cal_tt(self):
        '''
        Args:
            for processing input: [num_obj, num_dim, n_split]
            !!! no parallel !!!
        '''
        decay = 0.0
        dq = self.pos[:,:,1:] - self.pos[:,:,:-1] # dq: [num_obj, num_dim, n_split]
        dp = self.mom[:,:,1:] - self.mom[:,:,:-1] # dq: [num_obj, num_dim, n_split]
        
        q_mid = 0.5 * (self.pos[:,:,1:] + self.pos[:,:,:-1])
        p_mid = 0.5 * (self.mom[:,:,1:] + self.mom[:,:,:-1])
    
        self.tt[0] = 0.0
        for i_step in range(self.n_split):
            dt = []
            for i_obj in range(self.num_of_obj):
                for i_dim in range(self.dimensionality):
                    dt_ = dq[i_obj, i_dim, i_step] / self._dHdp(q_mid[:,:, i_step], 
                                                                p_mid[:,:, i_step],
                                                                i_obj,
                                                                i_dim).cpu().numpy()
                    dt.append(dt_)
                    dt_ = -dp[i_obj, i_dim, i_step] / self._dHdq(q_mid[:,:, i_step], 
                                                                 p_mid[:,:, i_step],
                                                                 i_obj,
                                                                 i_dim).cpu().numpy()
                    dt.append(dt_)
            
            tt_now = self.tt[i_step] + np.mean( np.array(dt) )
            self.tt[i_step+1] = decay * self.tt[i_step+1] + (1 - decay) * tt_now 
    
    def _Cal_tt_torch(self, q, p, dt_expect):
        filtered_mean = lambda data: (d := np.array(data))[np.abs(d-d.mean()) <= (1.618) * d.std()].mean()
        
        
        dq = q[..., 1:] - q[..., :-1] # dq: [num_obj, num_dim, n_split]
        dp = p[..., 1:] - p[..., :-1] # dq: [num_obj, num_dim, n_split]
        
        
        v =   self._dHdp_multi(q, p)
        F =  -self._dHdq_multi(q, p)
        v_mid = (v[...,1:] + v[...,:-1]) / 2.
        F_mid = (F[...,1:] + F[...,:-1]) / 2.
        
        dt_q = dq / (v_mid + 1e-20)
        dt_p = dp / (F_mid + 1e-20)
        
        
        dt_p[dt_p < 0.1 * dt_expect] =  0.1 * dt_expect
        dt_q[dt_q < 0.1 * dt_expect] =  0.1 * dt_expect
        dt_p[dt_p > 10.0 * dt_expect] = 10.0 * dt_expect
        dt_q[dt_q > 10.0 * dt_expect] = 10.0 * dt_expect       

        tt_q = torch.cumsum(dt_q, dim = -1 )
        tt_p = torch.cumsum(dt_p, dim = -1 )
        tt_q = torch.cat( (torch.zeros( (self.num_of_obj, self.dimensionality), device=q.device).unsqueeze(-1), tt_q), dim=-1 )
        tt_p = torch.cat( (torch.zeros( (self.num_of_obj, self.dimensionality), device=q.device).unsqueeze(-1), tt_p), dim=-1 )     
        tt = torch.mean(tt_q, dim=(-3,-2))  # [num_samples, n_split+1] here is for case 1, 2, 3 because the q_end points are fixed
        
        ## method 1:
        # t_total = tt[-1]
        
        ## method 2:
        t_total =   filtered_mean(dt_q.cpu().detach().numpy())   *  (q.shape[-1] - 1) 
        
        ## method 3 only for case 4:
        # t_total =   (filtered_mean(dt_q.cpu().detach().numpy()) + filtered_mean(dt_p.cpu().detach().numpy()) )/2.  * (q.shape[-1] - 1) 
        
        
        return tt, 0, torch.tensor(t_total, device=q.device) # for determine the t-series whether physics

    
    def _dHdq(self, q, p, i_obj, i_dim, t_now=0.0):
        if not torch.is_tensor(q):
            q = torch.from_numpy(q)
        if not torch.is_tensor(p):
            p = torch.from_numpy(p)
        def wrapped_f(q_Iobj_Id):
            q_full = q.clone()
            q_full[i_obj, i_dim] = q_Iobj_Id
            return self.HamiltonFnc_torch(q_full, p, t_now)
    

        q_Iobj_Id = q[i_obj, i_dim].detach()  # original momentum
        dHdq_i = compute_grad(wrapped_f, q_Iobj_Id)
        return dHdq_i
    
    
    def _dHdp(self, q, p, i_obj, i_dim, t_now=0.0):
        # create package function, (q, t_now, i_obj, i_dim is fixed) only derivative for p[i_obj, i_dim]
        if not torch.is_tensor(q):
            q = torch.from_numpy(q)
        if not torch.is_tensor(p):
            p = torch.from_numpy(p)
        
        def wrapped_f(p_Iobj_Id):
            p_full = p.clone()
            p_full[i_obj, i_dim] = p_Iobj_Id
            return self.HamiltonFnc_torch(q, p_full, t_now)
    

        p_Iobj_Id = p[i_obj, i_dim].detach()  # original momentum
        dHdp_i = compute_grad(wrapped_f, p_Iobj_Id)
        return dHdp_i

    
    def _dHdq_multi(self, q, p, t_now=0.0):
        # 确保输入为张量并移动到正确设备
        if not torch.is_tensor(q):
            q = torch.as_tensor(q, dtype=torch.float64, device=self.device)
        if not torch.is_tensor(p):
            p = torch.as_tensor(p, dtype=torch.float64, device=self.device)

        q = q.detach().requires_grad_(True)
        H_value = self.HamiltonFnc_torch(q, p ,t_now)
        
        dHdq = torch.autograd.grad(
                outputs=H_value,
                inputs=q,
                grad_outputs=torch.ones_like(H_value),  # 确保标量输出
                create_graph=True,  # 保留计算图
                retain_graph=True,  # 保留中间结果
                only_inputs=True
            )[0]

        return dHdq
    
    def _dHdp_multi(self, q, p, t_now=0.0):
        # 确保输入为张量并移动到正确设备
        if not torch.is_tensor(q):
            q = torch.as_tensor(q, dtype=torch.float64, device=self.device)
        if not torch.is_tensor(p):
            p = torch.as_tensor(p, dtype=torch.float64, device=self.device)
        
        # 确保 p 需要梯度
        p = p.detach().requires_grad_(True)
        
        # 计算 H(q, p) 并保留计算图
        H_value = self.HamiltonFnc_torch(q, p, t_now)
        
        # 计算 dH/dp 并创建计算图
        dHdp = torch.autograd.grad(
            outputs=H_value,
            inputs=p,
            grad_outputs=torch.ones_like(H_value),  # 确保标量输出
            create_graph=True,  # 保留计算图
            retain_graph=True,  # 保留中间结果
            only_inputs=True
        )[0]
        return dHdp
    
    
    def dHdq_dissipative(self, q, p):
        pass
    
    def Ed_dissipative(self, q, p):
        pass
            
    def generate_perturbations(self,
                               perturbation_shape,
                               perturbation_type,
                               eps_q,
                               eps_p):

        Tlen = perturbation_shape[-1]; num_samples3 = perturbation_shape[0]
        match perturbation_type:
            case 'white_noise':
                deltaq = torch.randn(perturbation_shape, device=self.device, dtype=torch.float64) * eps_q
                deltap = torch.randn(perturbation_shape, device=self.device, dtype=torch.float64) * eps_p
                
                deltaq[0::3, ...] =  torch.abs(deltaq[0::3, ...])
                deltaq[1::3, ...] = -torch.abs(deltaq[1::3, ...])
                deltap[0::3, ...] =  torch.abs(deltap[0::3, ...])
                deltap[1::3, ...] = -torch.abs(deltap[1::3,... ])
                deltaq[...,0]= 0.0; deltaq[...,-1] = 0.0

            case 'sinwhite':
                base = torch.linspace(0, 1, steps=Tlen, device=self.device, dtype=torch.float64)                     #shape:[n_split+1]
                end_values = (torch.arange(1, num_samples3 + 1, device=self.device, dtype=torch.float64) * torch.pi) #shape[num_samples]
                t_per_sample = end_values[:, None] * base[None, :]
                
                T = t_per_sample.view(num_samples3, 1, 1, Tlen)
                phase_q = torch.zeros((num_samples3,
                                       perturbation_shape[1], 
                                       perturbation_shape[2],
                                       1), device=self.device)
                deltaq = torch.sin(T + phase_q) * eps_q * (-1.0)**torch.randint(0,2,(1, perturbation_shape[1]), device=self.device).unsqueeze(-1).unsqueeze(-1)
                deltaq[...,0]= 0.0; deltaq[...,-1] = 0.0
                
                deltap = torch.randn(perturbation_shape, device=self.device, dtype=torch.float64) * eps_p
                deltap[0::3, ...] =  torch.abs(deltap[0::3, ...])
                deltap[1::3, ...] = -torch.abs(deltap[1::3,... ])
                
            case 'sin' | 'sincos':
                base = torch.linspace(0, 1, steps=Tlen, device=self.device, dtype=torch.float64)                     #shape:[n_split+1]
                end_values = (torch.arange(1, num_samples3 + 1, device=self.device, dtype=torch.float64) * torch.pi) #shape[num_samples]
                t_per_sample = end_values[:, None] * base[None, :]
                
                T = t_per_sample.view(num_samples3, 1, 1, Tlen)
                phase_q = torch.zeros((num_samples3,
                                       perturbation_shape[1], 
                                       perturbation_shape[2],
                                       1), device=self.device)
                phase_p = torch.zeros((num_samples3,
                                       perturbation_shape[1], 
                                       perturbation_shape[2],
                                       1), device=self.device) * 0.0
                
                deltaq = torch.sin(T + phase_q) * eps_q * (-1.0)**torch.randint(0,2,(1, perturbation_shape[1]), device=self.device).unsqueeze(-1).unsqueeze(-1)
                deltap = torch.cos(T + phase_p) * eps_p * (-1.0)**torch.randint(0,2,(1, perturbation_shape[1]), device=self.device).unsqueeze(-1).unsqueeze(-1)  #+  torch.randn(phase_p.shape).repeat(1,1,Tlen,1) * eps_p
                deltaq[...,0]= 0.0; deltaq[...,-1] = 0.0

            case 'sincosp1':
                base = torch.linspace(0, 1, steps=Tlen, device=self.device, dtype=torch.float64)                     #shape:[n_split+1]
                end_values = (torch.arange(1, num_samples3 + 1, device=self.device, dtype=torch.float64) * torch.pi) #shape[num_samples]
                t_per_sample = end_values[:, None] * base[None, :]
                
                T = t_per_sample.view(num_samples3, 1, 1, Tlen)
                phase_q = torch.zeros((num_samples3,
                                       perturbation_shape[1], 
                                       perturbation_shape[2],
                                       1), device=self.device)
                phase_p = torch.ones((num_samples3,
                                       perturbation_shape[1], 
                                       perturbation_shape[2],
                                       1), device=self.device)
                
                deltaq = torch.sin(T + phase_q) * eps_q * (-1.0)**torch.randint(0,2,(1, perturbation_shape[1]), device=self.device).unsqueeze(-1).unsqueeze(-1)
                deltap = torch.cos(T + phase_p) * eps_p * (-1.0)**torch.randint(0,2,(1, perturbation_shape[1]), device=self.device).unsqueeze(-1).unsqueeze(-1)  #+  torch.randn(phase_p.shape).repeat(1,1,Tlen,1) * eps_p
                deltaq[...,0]= 0.0; deltaq[...,-1] = 0.0
                
            case 'sinsin':
                base = torch.linspace(0, 1, steps=Tlen, device=self.device, dtype=torch.float64)                     #shape:[n_split+1]
                end_values = (torch.arange(1, num_samples3 + 1, device=self.device, dtype=torch.float64) * torch.pi) #shape[num_samples]
                t_per_sample = end_values[:, None] * base[None, :]
                
                T = t_per_sample.view(num_samples3, 1, 1, Tlen)
                phase_q = torch.zeros((num_samples3,
                                       perturbation_shape[1], 
                                       perturbation_shape[2],
                                       1), device=self.device)
                phase_p = torch.zeros((num_samples3,
                                       perturbation_shape[1], 
                                       perturbation_shape[2],
                                       1), device=self.device)
                
                deltaq = torch.sin(T + phase_q) * eps_q * (-1.0)**torch.randint(0, 2, (1, perturbation_shape[1]), device=self.device).unsqueeze(-1).unsqueeze(-1)
                deltap = torch.sin(T + phase_p) * eps_p * (-1.0)**torch.randint(0, 2, (1, perturbation_shape[1]), device=self.device).unsqueeze(-1).unsqueeze(-1) 
                deltaq[...,0]= 0.0; deltaq[...,-1] = 0.0
            
            case 'sincosm1':
                base = torch.linspace(0, 1, steps=Tlen, device=self.device, dtype=torch.float64)                     #shape:[n_split+1]
                end_values = (torch.arange(1, num_samples3 + 1, device=self.device, dtype=torch.float64) * torch.pi) #shape[num_samples]
                t_per_sample = end_values[:, None] * base[None, :]
                
                T = t_per_sample.view(num_samples3, 1, 1, Tlen)
                phase_q = torch.zeros((num_samples3,
                                       perturbation_shape[1], 
                                       perturbation_shape[2],
                                       1), device=self.device)
                phase_p = - torch.ones((num_samples3,
                                       perturbation_shape[1], 
                                       perturbation_shape[2],
                                       1), device=self.device)
                
                deltaq = torch.sin(T + phase_q) * eps_q * (-1.0)**torch.randint(0,2,(1, perturbation_shape[1]), device=self.device).unsqueeze(-1).unsqueeze(-1)
                deltap = torch.cos(T + phase_p) * eps_p * (-1.0)**torch.randint(0,2,(1, perturbation_shape[1]), device=self.device).unsqueeze(-1).unsqueeze(-1)  #+  torch.randn(phase_p.shape).repeat(1,1,Tlen,1) * eps_p
                deltaq[...,0]= 0.0; deltaq[...,-1] = 0.0

            case _:
                print(f'ERROR in noise_type:{perturbation_type}, there is no {perturbation_type}!')
        
        

        
        return deltaq, deltap

        
    
    
    def Action_torch(self, q, p, tt): # for optimize trace
        '''
        Params: q: [b, num_obj, num_dim, n_split]
                p: [b, num_obj, num_dim, n_split]
                t: [b, n_split]
        '''
        dq = q[...,1:] - q[...,:-1]
        dt = tt[...,1:] - tt[...,:-1]
        
        p_mid = (p[...,1:] + p[...,:-1])/2.
        H = self.HamiltonFnc_torch(q, p)
        H_mid = (H[...,1:] + H[...,:-1])/2.
        
        S_qp = torch.sum(p_mid*dq,dim=(-3,-2,-1))
        S_Ht = torch.sum(-H_mid * dt, dim=(-1))
        
        if not self.IF_dissipative:
            return S_qp + S_Ht
        else:
            Ed = self.Ed_dissipative(q, p)
            Ed_mid = (Ed[...,1:] + Ed[...,:-1])/2.
            S_Edt = torch.sum( - Ed_mid * dt, dim=(-1))
            return S_qp + S_Ht + S_Edt

        
        
    def Solver(self, eps_q, eps_p, n_integral, num_samples, lr, num_iterations,perturbation_type, IF_guess=True):
        for i_step in range(self.n_split):
            print('=*' * 30)
            print(f'now! doing {i_step}/{self.n_split} steps')
            print('=*' * 30)
            for i_obj in range(self.num_of_obj):
                for i_dim in range(self.dimensionality):
                    
                    prev_q = torch.from_numpy(self.pos[:,:,i_step]).to(self.device) # [num_obj, num_dim]
                    prev_p = torch.from_numpy(self.mom[:,:,i_step]).to(self.device) # [num_obj, num_dim]
                    prev_qi = prev_q[i_obj, i_dim]
                    prev_pi = prev_p[i_obj, i_dim]
                    
                    curr_q = prev_q.clone().detach() # [num_obj, num_dim]
                    curr_p = prev_p.clone().detach() # [num_obj, num_dim]
                    curr_qi = (prev_qi + prev_pi/self.mass[i_obj] * self.dt).clone().detach() # just Gauss
                    curr_pi = (prev_pi).clone().detach()
                    curr_q[i_obj, i_dim] = curr_qi
                    curr_p[i_obj, i_dim] = curr_pi
                    
                    
                    for _ in range(2):
                        ## optimise p_i
                        curr_pi = curr_pi.clone().detach().requires_grad_(True)
                        optimizer = optim.Adam([curr_pi], lr=lr)
                        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                                        optimizer, 
                                        mode='min',       # 监控损失最小化
                                        factor=0.5,       # 学习率乘以0.5
                                        patience=5,       # 5个epoch没有改善则降低学习率
                                        min_lr=1e-8       # 最小学习率
                                    )
                        
                        for i in range(num_iterations):
                            if IF_guess:
                                curr_qi = (prev_qi + curr_pi/self.mass[i_obj] * self.dt).clone().detach()
                            else:
                                curr_qi = (prev_qi).clone().detach()
                            curr_q[i_obj, i_dim] = curr_qi.clone().detach()
                            optimizer.zero_grad()
                            
                            perturbation_shape = ( 3 * num_samples, self.num_of_obj, self.dimensionality, n_integral+1)
                            deltaq, _ = self.generate_perturbations(perturbation_shape=perturbation_shape,
                                                                    perturbation_type=perturbation_type,
                                                                    eps_p=eps_p,
                                                                    eps_q=eps_q)
                            
                            deltaq  = deltaq[:, i_obj,i_dim, :]
                            
                            DeltaS_overdp = deltaq.sum(dim=-1) * ((curr_pi - prev_pi) / n_integral )
                            
                            mid_q = (curr_q + prev_q)/2
                            mid_p = (curr_p + prev_p)/2
                            dHdq_val = self._dHdq_multi(mid_q, mid_p)[i_obj, i_dim]
                            
                            if self.IF_dissipative:
                                dHdp_val += self.dHdq_dissipative(mid_q, mid_p)[i_obj, i_dim]
                                
                            DeltaS_overdt = deltaq.sum(dim=-1) * (self.dt/n_integral) * dHdq_val
                            
                            loss = torch.mean( torch.square(DeltaS_overdp + DeltaS_overdt) )  / torch.tensor(eps_q**2)
                            if loss.item() < 1e-60:
                                break
                            loss.backward()
                            optimizer.step()
                            scheduler.step(loss)
                            
                            curr_p[i_obj, i_dim] = curr_pi.clone().detach()
                            
                            
                            # 打印进度
                            if i % (num_iterations//5) == 0:
                                print(f'optimize p_{i_obj}_{i_dim}   | Iteration {i:3d} | p_i: {curr_pi.item():.6f} | Loss: {loss.item():.16f}')
                        
                        ## optimise q_i
                        curr_qi = curr_qi.clone().detach().requires_grad_(True)
                        optimizer = optim.Adam([curr_qi], lr=lr)
                        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                                        optimizer, 
                                        mode='min',       # 监控损失最小化
                                        factor=0.5,       # 学习率乘以0.5
                                        patience=5,       # 5个epoch没有改善则降低学习率
                                        min_lr=1e-8       # 最小学习率
                                    )
                                                    
                        for i in range(num_iterations):
                            optimizer.zero_grad()
                            
                            perturbation_shape = ( 3 * num_samples, self.num_of_obj, self.dimensionality, n_integral+1)
                            _, deltap = self.generate_perturbations(perturbation_shape=perturbation_shape,
                                                                    perturbation_type=perturbation_type,
                                                                    eps_p=eps_p,
                                                                    eps_q=eps_q)
                            deltap  = deltap[:, i_obj,i_dim, :]
                            
                            DeltaS_overdq = deltap.sum(dim=-1) * ((curr_qi - prev_qi) / n_integral )
                            mid_q = (curr_q + prev_q)/2
                            mid_p = (curr_p + prev_p)/2
                            dHdp_val = self._dHdp_multi(mid_q, mid_p)[i_obj, i_dim]
                            DeltaS_overdt = deltap.sum(dim=-1) * (self.dt/n_integral) * (-dHdp_val)

                            loss =  torch.mean( torch.square(DeltaS_overdq + DeltaS_overdt) ) / torch.tensor(eps_p**2)
                            if loss.item() < 1e-60:
                                break
                            loss.backward()
                            optimizer.step()
                            scheduler.step(loss)
                            
                            curr_q[i_obj, i_dim] = curr_qi.clone().detach()
                            
                            # 打印进度
                            if i % (num_iterations//5) == 0:
                                print(f'optimize q_{i_obj}_{i_dim}  | Iteration {i:3d} | q_i: {curr_qi.item():.6f} | Loss: {loss.item():.16f}') 
                        
                    self.pos[i_obj, i_dim, i_step+1] = curr_qi.cpu().item()
                    self.mom[i_obj, i_dim, i_step+1] = curr_pi.cpu().item()
                            
    def Optimize_NewParams(self, pos_end_cond, mom_end_cond ,
                           eps_q, eps_p,                     
                           lambda_q, lambda_p,
                           num_samples, IF_progressive,
                           reduce_op_type ,lr,  lr_scheduler,
                           lr_updatet, lr_decay, 
                           num_iterations, dt_expect, 
                           t_min,  t_end_given=None, period_updatet=25, 
                           IF_uniform_dt=True, decay_updatet=0.995, 
                           lambda_smooth_q_1st=[0.0], lambda_smooth_p_1st=[0.0],
                           lambda_smooth_q_2nd=[0.0], lambda_smooth_p_2nd=[0.0], 
                           lambda_physics = 0e0, IF_weight=False, IF_autoscale_loss=False,
                           variation_method='direct_integral', # or 'indirect_substract'
                           perturbation_type='sincos', optimizers=['SGDM']):      
        
        if not isinstance(lambda_smooth_p_1st, list):
            lambda_smooth_p_1st = [lambda_smooth_p_1st]
        if not isinstance(lambda_smooth_q_1st, list):
            lambda_smooth_q_1st = [lambda_smooth_q_1st]
        if not isinstance(lambda_smooth_p_2nd, list):
            lambda_smooth_p_2nd = [lambda_smooth_p_2nd]
        if not isinstance(lambda_smooth_q_2nd, list):
            lambda_smooth_q_2nd = [lambda_smooth_q_2nd]
           
        reduce_op = torch.sum if reduce_op_type == 'sum' else torch.mean
         
        #add time dimention
        pos_end_cond = torch.from_numpy(pos_end_cond).unsqueeze(-1).to(device=self.device)
        mom_end_cond = torch.from_numpy(mom_end_cond).unsqueeze(-1).to(device=self.device)
        
        best_loss = 1e10
        loss_lst = []   
        lr_lst = []
        all_curves = [] 
          
        pos_start = torch.from_numpy(self.pos[:,:,[0]]).to(self.device).requires_grad_(False).to(device=self.device)
        mom_start = torch.from_numpy(self.mom[:,:,[0]]).to(self.device).requires_grad_(False).to(device=self.device)
        
        pos_op = torch.from_numpy(self.pos[:,:,1:-1]).to(self.device).requires_grad_(True).to(device=self.device)
        mom_op = torch.from_numpy(self.mom[:,:,1:-1]).to(self.device).requires_grad_(True).to(device=self.device)
        
        pos_end = torch.from_numpy(self.pos[:,:,[-1]]).to(self.device)
        mom_end = torch.from_numpy(self.mom[:,:,[-1]]).to(self.device)
        
        pos_fixed_mask = torch.zeros_like(pos_end, dtype=torch.bool).to(device=self.device)
        mom_fixed_mask = torch.zeros_like(mom_end, dtype=torch.bool).to(device=self.device)
        
        Tlen = pos_start.shape[-1] + pos_op.shape[-1] + pos_end.shape[-1]
        for i_obj in range(self.num_of_obj):
            for i_dim in range(self.dimensionality):
                if pos_end_cond[i_obj, i_dim,...] != torch.inf:
                    pos_end[i_obj, i_dim,...] = pos_end_cond[i_obj, i_dim,...]
                    pos_fixed_mask[i_obj, i_dim,...] = True
                if mom_end_cond[i_obj, i_dim,...] != torch.inf:
                    mom_end[i_obj, i_dim,...] = mom_end_cond[i_obj, i_dim,...]
                    mom_fixed_mask[i_obj, i_dim,...] = True
        
        pos_end.to(device=self.device).requires_grad_(True)
        mom_end.to(device=self.device).requires_grad_(True)
             
             
        gamma = decay_updatet
        t_total_prev = torch.tensor(self.t_end, device=self.device, requires_grad=False); 
        tt_prev  = torch.linspace(0, self.t_end, self.tt.shape[0], dtype=torch.float64, device= self.device, requires_grad=False)
            
            
        for optimizer_type in optimizers:
            match optimizer_type.lower(): 
                case 'adam':
                    optimizer = optim.Adam(
                        [{'params': pos_op, 'lr': lr[0]},
                        {'params': mom_op, 'lr': lr[0]},
                        {'params': pos_end,'lr': lr[0]},
                        {'params': mom_end,'lr': lr[0]}
                        ])
                    iter_range = range(num_iterations) 
                    
                case 'adamw':
                    optimizer = optim.AdamW(
                        [{'params': pos_op, 'lr': lr[0]},
                        {'params': mom_op, 'lr': lr[0]},
                        {'params': pos_end, 'lr': lr[0]},
                        {'params': mom_end, 'lr': lr[0]}
                        ],
                        weight_decay=0.01  # AdamW特有的权重衰减参数
                    )
                    iter_range = range(num_iterations) 
                
                case 'gd':
                    all_params = list(pos_op) + list(mom_op) + list(pos_end) + list(mom_end) if isinstance(pos_op, (list, tuple)) else [pos_op, mom_op, pos_end, mom_end]
                    optimizer = PureGradientDescent(all_params, lr=lr[0])
                    iter_range = range(num_iterations)
                    
                case 'sgd':  # 新增SGD优化器
                    optimizer = optim.SGD(
                        [{'params': pos_op, 'lr': lr[0]},
                        {'params': mom_op, 'lr': lr[0]},
                        {'params': pos_end, 'lr': lr[0]},
                        {'params': mom_end, 'lr': lr[0]}
                        ],
                        weight_decay=0.0,  # 可以根据需要添加权重衰减
                    )
                    iter_range = range(num_iterations)
                
                case 'sgdm':  # 带动量的SGD (SGDM)
                    optimizer = optim.SGD(
                        [{'params': pos_op, 'lr': lr[0]},
                            {'params': mom_op, 'lr': lr[0]},
                            {'params': pos_end, 'lr': lr[0]},
                            {'params': mom_end, 'lr': lr[0]}],
                        momentum=0.9,  # 关键：设置动量系数，通常0.9
                        weight_decay=0.0,
                        nesterov=True  # 可选：使用Nesterov加速梯度，效果更好
                        )
                    iter_range = range(num_iterations)
                       
                case 'lbfgs':
                    all_params = list(pos_op) + list(mom_op) + list(pos_end) + list(mom_end) if isinstance(pos_op, (list, tuple)) else [pos_op, mom_op, pos_end, mom_end]
                    optimizer = optim.LBFGS(
                        all_params,
                        lr=lr[1],          # 关键：降低学习率！
                        max_iter=20,                # 减少单步迭代
                        max_eval=25,                # 减少评估次数
                        history_size=100,            # 降低历史记录
                        tolerance_grad=1e-11,        # 放宽梯度容差
                        tolerance_change=1e-11,      # 放宽函数变化容差
                        line_search_fn='strong_wolfe'  # 保留线搜索
                    )
                    iter_range = range(num_iterations//2)      
                case _:
                    raise ValueError(f'optimizer_type {optimizer_type} not recognized.')
            
            match lr_scheduler:
                case 'Onplateau':                  
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer, 
                            mode='min',       # 监控损失最小化
                            factor=lr_decay,       # 学习率乘以0.5
                            patience=lr_updatet,       # keep patience
                            min_lr=1e-10      # 最小学习率
                        )
                case 'Cos':
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            T_max=num_iterations//2,  # 最大迭代次数（半个周期的长度）
                            eta_min=lr[0]/1e2         # 最小学习率
                        )
                case _:
                    raise ValueError(f'lr_scheduler {lr_scheduler} not recognized.')
            
            
            perturbation_shape = (3 * num_samples, self.num_of_obj, self.dimensionality, Tlen)
            mask_progressive = torch.zeros(perturbation_shape, device=self.device)
            for i_step in iter_range:
                lambda_smooth_p_1st_ = lambda_smooth_p_1st[min(i_step//(num_iterations//len(lambda_smooth_p_1st)), len(lambda_smooth_p_1st)-1)]
                lambda_smooth_q_1st_ = lambda_smooth_q_1st[min(i_step//(num_iterations//len(lambda_smooth_q_1st)), len(lambda_smooth_q_1st)-1)]
                lambda_smooth_p_2nd_ = lambda_smooth_p_2nd[min(i_step//(num_iterations//len(lambda_smooth_p_2nd)), len(lambda_smooth_p_2nd)-1)]
                lambda_smooth_q_2nd_ = lambda_smooth_q_2nd[min(i_step//(num_iterations//len(lambda_smooth_q_2nd)), len(lambda_smooth_q_2nd)-1)]
                
                if i_step % (self.num_of_obj * self.dimensionality * period_updatet) == 0:
                    q_full = torch.cat([pos_start, pos_op, pos_end], dim=-1)
                    p_full = torch.cat([mom_start, mom_op, mom_end], dim=-1)

                    
                    tt_tensor, counts, t_total = self._Cal_tt_torch(q_full.detach().clone(), p_full.detach().clone(), dt_expect)
                    t_total = t_total if t_end_given == None else torch.tensor(t_end_given, device=self.device) 

                    
                    t_total_curr = gamma * t_total_prev + (1. - gamma) * t_total.clone().detach() 
                    tt_curr = gamma * tt_prev + (1. - gamma) * tt_tensor.clone().detach()
                    
                    
                    t_total_prev = t_total_curr
                    tt_prev = tt_curr
                    

                    
                    if IF_uniform_dt:
                        tt_tensor = torch.linspace(
                            0.0, torch.max(t_total_prev, torch.tensor(t_min,device=self.device)), 
                            Tlen, device=self.device
                            )
                    else:
                        tt_tensor = tt_curr
                    tt_tensor = tt_tensor.requires_grad_(False).to(device=self.device)
                

                deltaq, deltap = self.generate_perturbations(perturbation_shape=perturbation_shape,
                                                            perturbation_type=perturbation_type,
                                                            eps_p=eps_p,
                                                            eps_q=eps_q,
                                                            )
                
                if  IF_progressive:
                    change_T = num_iterations // ( 3 * num_samples)
                    openOne = min(i_step // change_T, 3*num_samples -1) 
                    mask_progressive[openOne,...] = 1.0
                    deltaq = deltaq * mask_progressive
                    deltap = deltap * mask_progressive
                
                
                # create masks for each dq_i and dp_i
                masks = torch.zeros(2 * self.num_of_obj * self.dimensionality, *perturbation_shape, device=self.device)
                for i, (i_obj, i_dim, is_dp) in enumerate([(i_obj, i_dim, is_dp) 
                                                            for is_dp in [False, True]
                                                            for i_obj in range(self.num_of_obj) 
                                                            for i_dim in range(self.dimensionality) ] ):
                    if not is_dp:  # dq mask
                        masks[i, ..., i_obj, i_dim, :] = 1.0
                    else:  # dp mask : store in the last half 
                        masks[i, ..., i_obj, i_dim, :] = 1.0

                # split into two partu
                dq_masks, dp_masks = masks.chunk(2, dim=0)
                
                
                deltaq_ =  deltaq.repeat(self.num_of_obj*self.dimensionality, *([1] * (deltaq.dim() - 1))) * dq_masks.reshape(-1, *perturbation_shape[1:])
                deltap_ =  deltap.repeat(self.num_of_obj*self.dimensionality, *([1] * (deltap.dim() - 1))) * dp_masks.reshape(-1, *perturbation_shape[1:])
                
                if IF_weight:
                    linear_n = 1.0 + self.mu * (torch.linspace(1, Tlen, Tlen, device=self.device, requires_grad=False)).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(3 * num_samples * self.num_of_obj * self.dimensionality, self.num_of_obj, self.dimensionality, 1)
                
                    deltaq_ = deltaq_ * linear_n / linear_n.mean()
                    deltap_ = deltap_ * linear_n / linear_n.mean()
                
                IF_qpseperate = False
                if IF_qpseperate:
                    deltaq_ = deltaq_ * max(0,  (-1)**(i_step//1000     ))
                    deltap_ = deltap_ * max(0,  (-1)**(i_step//1000  + 1))

                def closure(tt=tt_tensor,  #shape: [n_split+1]
                            pos_fixed_mask=pos_fixed_mask, 
                            mom_fixed_mask=mom_fixed_mask,
                            pos_end_cond = pos_end_cond,
                            mom_end_cond = mom_end_cond): # tt_tensor would't change in one-step L-BFGS
                    optimizer.zero_grad()
                    
                    q_full = torch.cat([pos_start, pos_op, pos_end], dim=-1).to(device=self.device)
                    p_full = torch.cat([mom_start, mom_op, mom_end], dim=-1).to(device=self.device)
                    
                    
                    reg_loss_dq_1st =  torch.mean(torch.square( q_full[...,1:] - q_full[...,:-1]  ))
                    reg_loss_dp_1st =  torch.mean(torch.square( p_full[...,1:] - p_full[...,:-1]  ))

                    reg_loss_dq_2nd =  torch.mean( torch.square(q_full[...,2:] - 2 * q_full[...,1:-1] + q_full[...,:-2]))
                    reg_loss_dp_2nd =  torch.mean( torch.square(p_full[...,2:] - 2 * p_full[...,1:-1] + p_full[...,:-2]))
                    
                    p_full_mid  = (p_full[...,  :-1] + p_full[...,  1:  ])/2.
                    dq_full_mid = (q_full[..., 1:  ] - q_full[...,   :-1])/2.
                    reg_loss_physics = torch.mean(torch.relu(-p_full_mid * dq_full_mid)) * lambda_physics 
                    

                    reg_loss_smooth = reg_loss_dq_1st * lambda_smooth_q_1st_ +\
                                        reg_loss_dp_1st * lambda_smooth_p_1st_ +\
                                            reg_loss_dq_2nd * lambda_smooth_q_2nd_ +\
                                                reg_loss_dp_2nd * lambda_smooth_p_2nd_
                                         
                    
                    if variation_method == 'indirect_substract':
                        tt_ = tt.unsqueeze(0).repeat(3 * num_samples * self.num_of_obj * self.dimensionality,     1)
                        
                        S =  self.Action_torch(q_full, 
                                            p_full, 
                                            tt).to(device=self.device)
                        
                        Sbar_dq = self.Action_torch(q_full.unsqueeze(0) + deltaq_, 
                                                    p_full.unsqueeze(0) + deltap_ * 0.0, 
                                                    tt_ ).to(device=self.device)
                        
                        Sbar_dp = self.Action_torch(q_full.unsqueeze(0) + deltaq_ * 0.0, 
                                                    p_full.unsqueeze(0) + deltap_, 
                                                    tt_ ).to(device=self.device)
                                
                        loss_dq = reduce_op(   torch.square((Sbar_dq - S) / eps_q)   ) * lambda_q
                        
                        if IF_autoscale_loss:
                            loss_dp = reduce_op(   torch.square((Sbar_dp - S) / eps_p)   ) * lambda_p
                            # lambda_p_ = 1./ loss_dp.item()
                            # loss_dp = loss_dp * lambda_p_
                            loss_dq = reduce_op(   torch.square((Sbar_dq - S) / eps_q)   ) * lambda_q
                            lambda_q_ = loss_dp.item() / loss_dq.item()
                            loss_dq = reduce_op(   torch.square((Sbar_dq - S) / eps_q)   ) * lambda_q * lambda_q_
                        else:
                            loss_dp = reduce_op(   torch.square((Sbar_dp - S) / eps_p)   ) * lambda_p
                            loss_dq = reduce_op(   torch.square((Sbar_dq - S) / eps_q)   ) * lambda_q
                
                    elif variation_method == 'direct_integral':
                        q_full_extend = q_full.unsqueeze(0).repeat(3 * num_samples * self.num_of_obj * self.dimensionality, 1,1,1)
                        p_full_extend = p_full.unsqueeze(0).repeat(3 * num_samples * self.num_of_obj * self.dimensionality, 1,1,1)

                        dq_extend = q_full_extend[..., 1:] - q_full_extend[..., :-1]
                        dp_extend = p_full_extend[..., 1:] - p_full_extend[..., :-1]
                        
                        dHdq =  self._dHdq_multi(q_full_extend, p_full_extend).to(device=self.device)
                        dHdp =  self._dHdp_multi(q_full_extend, p_full_extend).to(device=self.device)
                        dHdq_mid = (dHdq[...,:-1] + dHdq[...,1:])/2.
                        dHdp_mid = (dHdp[...,:-1] + dHdp[...,1:])/2.
                             
                        deltaq_mid = 0.5 * (deltaq_[...,1:] + deltaq_[...,:-1])
                        deltap_mid = 0.5 * (deltap_[...,1:] + deltap_[...,:-1])
                    
                        dt4 = (tt[1:] - tt[:-1]).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(3 * num_samples * self.num_of_obj * self.dimensionality, self.num_of_obj, self.dimensionality, 1)
                        if self.IF_dissipative:
                            dHdq_dissipative = self.dHdq_dissipative(q_full_extend, p_full_extend).to(device=self.device)
                            dHdq_mid = dHdq_mid + (dHdq_dissipative[...,1:] + dHdq_dissipative[...,:-1]) / 2.
                        
                        deltaS_over_deltap = torch.einsum('ijkl,ijkl->i', deltap_mid, dq_extend - dHdp_mid * dt4)
                        deltaS_over_deltaq = torch.einsum('ijkl,ijkl->i', deltaq_mid, dp_extend + dHdq_mid * dt4) 
                        
                        if IF_autoscale_loss:
                            loss_dp = reduce_op(   torch.square(deltaS_over_deltap / eps_p)   ) * lambda_p
                            # lambda_p_ = 1./ loss_dp.item()
                            # loss_dp = loss_dp * lambda_p_
                            loss_dq = reduce_op(   torch.square(deltaS_over_deltaq / eps_q)   ) * lambda_q
                            lambda_q_ = loss_dp.item() / loss_dq.item()
                            loss_dq = reduce_op(   torch.square(deltaS_over_deltaq / eps_q)   ) * lambda_q * lambda_q_
                        else:
                            loss_dp = reduce_op( torch.square(deltaS_over_deltap / eps_p)) * lambda_p
                            loss_dq = reduce_op( torch.square(deltaS_over_deltaq / eps_q)) * lambda_q
                        
                    else:
                        raise ValueError(f'variation_method {variation_method} not recognized.')
                    
                    loss = reg_loss_smooth + reg_loss_physics +  loss_dq + loss_dp
                    loss_lst.append([reg_loss_smooth.item(), reg_loss_physics.item(), loss_dq.item(), loss_dp.item(), loss.item() ])
                    lr_lst.append(optimizer.param_groups[0]['lr'])
                    loss.backward() 
                    

                    
                    match optimizer_type.lower():
                        case 'lbfgs':
                            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                        case 'adam' | 'adamw' | 'sgd' | 'sgdm' | 'gd':
                            pass
                        case _: 
                            raise ValueError(f'optimizer_type {optimizer_type} not recognized.')
                        
                    with torch.no_grad():
                        pos_end.grad[pos_fixed_mask] = 0.0
                        mom_end.grad[mom_fixed_mask] = 0.0
                        pos_end[pos_fixed_mask] = pos_end_cond[pos_fixed_mask]
                        mom_end[mom_fixed_mask] = mom_end_cond[mom_fixed_mask]
                        
                        if IF_qpseperate:
                            mom_end.grad = mom_end.grad * max(0,  (-1)**(i_step//1000     ))
                            pos_end.grad = pos_end.grad * max(0,  (-1)**(i_step//1000  + 1))
                            mom_op.grad = mom_op.grad * max(0,  (-1)**(i_step//1000     ))
                            pos_op.grad = pos_op.grad * max(0,  (-1)**(i_step//1000  + 1))
                            
                    return loss
                
                match optimizer_type.lower():
                    case 'adam' | 'adamw' | 'sgd' | 'sgdm' | 'gd':
                        loss = closure()
                        loss_value = loss.item()  
                        optimizer.step()
                    case 'lbfgs':
                        loss = optimizer.step(closure)
                        loss_value = loss.item() if isinstance(loss, torch.Tensor) else loss
                    case _:
                        raise ValueError(f'optimizer_type {optimizer_type} not recognized.')
                    
                if loss_value < 1e-30:
                    print(f"Early stopping at step {i_step}, loss={loss.item():.4e}")
                    break
                
                match optimizer_type.lower():
                    case 'adam' | 'adamw' | 'sgd' | 'sgdm' | 'gd':
                        if lr_scheduler == 'Cos':
                            scheduler.step()
                        else:
                            scheduler.step(loss_value)
                    case 'lbfgs': 
                        if i_step % 10 == 0:  # 减少调度器更新频率
                            scheduler.step(loss_value)
                    case _:
                        raise ValueError(f'optimizer_type {optimizer_type} not recognized.')
                

                
                # print process
                if i_step % 25 == 0 or i_step == num_iterations - 1:
                    print('=*' * 30)
                    print(f'now! doing {i_step}/{num_iterations} steps, optimizer:{optimizer_type}')
                    print(f"Step {i_step}: Loss = {loss.item():.6e}, LR = {optimizer.param_groups[0]['lr']:.8e}, t_total = {t_total_prev.item():.6f}")
                    print('=*' * 30)
                    print('\n')
                
                if i_step % (num_iterations // 1000 + 1) == 0:
                    all_curves.append( (tt_tensor.clone().detach().cpu().numpy(), p_full[0,0,:].clone().detach().cpu().numpy()) )
                    if loss.item() < best_loss:
                        self.pos_best = torch.cat([pos_start, pos_op, pos_end], dim=-1).clone().detach().cpu().numpy()
                        self.mom_best = torch.cat([mom_start, mom_op, mom_end], dim=-1).clone().detach().cpu().numpy()
                        best_loss = loss.item()
                
        print(f'pos_end : {pos_end}')
        print(f'mom_end : {mom_end}')
        self.loss_reparams = loss_lst
        self.lr_reparams = lr_lst
        self.tt = tt_tensor.clone().detach().cpu().numpy()
        self.t_end = self.tt[-1]
        self.n_split = len(self.tt) - 1 # the len(dt) = len(t) - 1
        self.pos = torch.cat([pos_start, pos_op, pos_end], dim=-1).clone().detach().cpu().numpy()
        self.mom = torch.cat([mom_start, mom_op, mom_end], dim=-1).clone().detach().cpu().numpy()
        
        ## add lowpass filter for pos and mom
        for i_obj in range(self.num_of_obj):
            for i_dim in range(self.dimensionality):
                self.pos[i_obj, i_dim, :] = Lowpass_filter(self.pos[i_obj, i_dim, :], order=4)
                self.mom[i_obj, i_dim, :] = Lowpass_filter(self.mom[i_obj, i_dim, :], order=4)
                self.pos_best[i_obj, i_dim, :] = Lowpass_filter(self.pos_best[i_obj, i_dim, :], order=4)
                self.mom_best[i_obj, i_dim, :] = Lowpass_filter(self.mom_best[i_obj, i_dim, :], order=4)                                       
                
        return all_curves

def Lowpass_filter(data, cutoff_freq=None, sampling_freq=None, order=4):
    from scipy.signal import butter, filtfilt
    
    """
    Arg:
    data: input data
    cutoff_freq:  (Hz)
    sampling_freq:  (Hz)
    order:  (默认4)
    
    return:
    data after filtering
    """
    if cutoff_freq is None and sampling_freq is None:
        normal_cutoff = 0.5  # 默认截止频率为采样频率的一半
    else:
        nyquist_freq = 0.5 * sampling_freq
        normal_cutoff = cutoff_freq / nyquist_freq
    
    # design butter lowpass filter
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    filtered_data = filtfilt(b, a, data)
    
    return filtered_data

if __name__ == "__main__":
    x = np.linspace(0, 10, 1000)
    original_signal = np.sin(2 * np.pi * 0.5 * x) + 0.5 * np.sin(2 * np.pi * 2 * x)
    high_freq_noise = 0.3 * np.sin(2 * np.pi * 15 * x) + 0.2 * np.random.normal(size=len(x))
    y_noisy = original_signal + high_freq_noise
    y = Lowpass_filter(y_noisy , order=4)
    plt.plot(y_noisy, label='Noisy Signal')
    plt.plot(y, label='Filtered Signal', linewidth=2)
    plt.plot(original_signal, label='Original Signal', linestyle='--')
    plt.legend()
    plt.show()

    


