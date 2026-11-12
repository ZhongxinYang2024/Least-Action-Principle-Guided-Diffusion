import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import tqdm
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# 自定义数据集类
class CustomDataset_SIN(Dataset):
    def __init__(self, num_samples=1000, num_features=128, T=1):
        super().__init__()
        # 
        data = torch.zeros(num_samples, num_features)
        labels = torch.zeros(num_samples)
        for i in range(num_samples):
            tbegin = i/(T*num_samples)
            labels[i] = tbegin;
            tt = tbegin + torch.linspace(0,T,num_features)
            data[i] = torch.sin(2 * torch.pi * tt)
        self.data = data
        self.labels = labels

        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 返回元组格式 (特征, 标签) - 更安全的数据格式
        return self.data[idx], self.labels[idx]
    

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # 在初始化期间随机采样权重。 这些权重是固定的 
    # 在优化期间并且不可训练
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False) # W.shape = (embend_dim//2, )
  def forward(self, x): # x.shape = (B,)
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi # x_proj.shape= (B,embed_dim//2)
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1) # shape(B, embed_dim)

class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)

class ScoreNet_MLP(nn.Module):
    '''Initailize a time dependent score-based MLP'''
    
    def __init__(self, marginal_prob_std, layers=[256,512,256], input_dim=128, embed_dim=128):
        '''
        Args: 
        marginal_prob_std: offer deviation at given time t
        layers: the number of every layer
        embed_dim: do time embeding
        '''
        super().__init__()
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
                                   nn.Linear(embed_dim,embed_dim))
        
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std
        
        self.layers = nn.ModuleList()
        
        prev_size = input_dim + embed_dim
        for size in layers:  # 不包括输出层
            self.layers.append(nn.Linear(prev_size, size))
            prev_size = size
        
        # 添加输出层
        self.layers.append(nn.Linear(prev_size, input_dim))

    
    
    def forward(self,x,t): # x: [B, n_features], t: [B,]
        embed = self.act(self.embed(t)) #embed: [B, embed_dim]
        h = torch.cat([x, embed], dim=-1) #h: [B, n_features+embed_dim]
        for i, layer in enumerate(self.layers[:-1]):
            h = self.act(layer(h))
        h = self.layers[-1](h)  # 输出层
        h = h / self.marginal_prob_std(t)[:, None]
        return h

def marginal_prob_std(t, sigma):
    '''
    Compute the mean and standard deviation of p(x(t) | x(0))
    Args:
        t: A vector of time steps (tensor)
        sigma: The σ parameter in our SDE (scalar)
    '''
    # 直接在原始设备上操作，避免不必要的复制
    t_scaled = t.detach().clone()
    # 使用 torch.log 保持张量计算
    return torch.sqrt((sigma**(2 * t_scaled) - 1.) / (2. * torch.log(torch.tensor(sigma, device=t.device))))

def diffusion_coeff(t, sigma):
    '''
    Compute the diffusion coefficient of SDE
    Args:
        t: A vector of time steps (tensor)
        sigma: The σ parameter in our SDE (scalar)
    '''
    # 直接返回计算结果，自动继承设备信息
    return sigma ** t.detach().clone()
    
if __name__ == '__main__': 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    sigma = 25.0
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)


    def loss_fn(model, x, marginal_prob_std, eps=1e-5):
        """The loss function for training score-based generative models.

        Args:
            model: 时间依赖，基于分数的 PyTorch model.
            x: A mini-batch of training data.    
            marginal_prob_std: A function that gives the standard deviation of 
            the perturbation kernel.
            eps: A tolerance value for numerical stability.
        """
        random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps # random_t: [B,]
        # t from [eps, 1-eps]  
        z = torch.randn_like(x) #x: [B, n_features]
        std = marginal_prob_std(random_t)
        perturbed_x = x  +   z * std[:,None]
        score = model(perturbed_x, random_t)
        loss = torch.mean( torch.sum((score * std[:, None] + z)**2, dim=(1)) )
        return loss


    score_model = torch.nn.DataParallel(ScoreNet_MLP(marginal_prob_std=marginal_prob_std_fn))
    score_model = score_model.to(device)

    n_epochs =   500#@param {'type':'integer'}
    ## size of a mini-batch
    batch_size =  64 #@param {'type':'integer'}
    ## learning rate
    lr=1e-3 #@param {'type':'number'}

    dataset = CustomDataset_SIN(num_samples=10000, num_features=128, T=2)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    optimizer = Adam(score_model.parameters(), 
                     lr=lr,
                     betas=(0.9, 0.999),
                     eps=1e-8,
                     weight_decay=0)
    
    ### scheduler 
    Type_scheduler = 'None'
    match Type_scheduler:
        case 'Geometric':
            # 添加学习率调度器 (动态调整)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min',        # 监控损失最小值
                factor=0.5,        # 学习率衰减因子
                patience=10,       # 10个epoch无改善则降低LR
                min_lr=1e-8        # 最小学习率下限
            )
        case 'Exponential':
            # 指数衰减 (每10个epoch衰减50%)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
        case 'CosAnneal':
            # 余弦退火 (T_max周期后重置LR)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        case 'None':
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                                    optimizer, 
                                    lr_lambda=lambda epoch: 1.0  # 学习率乘数恒为1.0
                                    )
        case _:
            print(f'Some error in Scheduler: {Type_scheduler}')
    
    # 添加变量跟踪学习率变化
    last_lr = lr
    lr_changed = False

    epoch_bar = tqdm(range(n_epochs), desc='Training')
    for epoch in epoch_bar:
        avg_loss = 0.
        num_items = 0
        for x, y in data_loader:
            x = x.to(device)    
            loss = loss_fn(score_model, x, marginal_prob_std_fn)
            optimizer.zero_grad()
            loss.backward()    
            optimizer.step()
            avg_loss += loss.item() * x.shape[0] 
            num_items += x.shape[0]
        
        epoch_avg_loss = avg_loss / num_items
        
        # 更新学习率 (基于本epoch平均损失)
        scheduler.step(epoch_avg_loss)
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 检查学习率是否变化
        if current_lr != last_lr:
            lr_changed = True
            print(f"\nEpoch {epoch+1}: Learning rate reduced to {current_lr:.2e}")
        else:
            lr_changed = False
            
        last_lr = current_lr
        
        # 在进度条中显示损失和当前学习率
        epoch_bar.set_postfix(avg_loss=epoch_avg_loss, lr=current_lr)
        torch.save(score_model.state_dict(), 'ckpt_adam.pth')
    

  
  