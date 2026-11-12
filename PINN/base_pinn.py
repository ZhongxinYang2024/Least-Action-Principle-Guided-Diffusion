'''
@Author: Zhongxin Yang (zhongxinyang@stu.pku,edu.cn)

'''
#%% 
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.autograd.functional import jacobian
import time
import os
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

#%%
class PhysicsInformedNN(nn.Module):
    """
    A general framework
    """
    def __init__(self, layers, activation=nn.Tanh()):
        """
        initail
        
        Arg:
            layers: list, [input_dim, hidden1, hidden2, ..., output_dim]
            activation: act function, default is Tanh
        """
        super(PhysicsInformedNN, self).__init__()
        
        self.act = activation
        self.layers = nn.ModuleList()
        
        # construct NN
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.act(layer(x))
        x = self.layers[-1](x)
        return x


class PINNDataset(Dataset):
    """
    Custom dataset, for load and manage training data
    """
    def __init__(self, domain_pos, 
                 bc_pos=None, bc_value=None,
                 ic_pos=None, ic_value=None,
                 pde_pos=None, pde_value=None):
        """
        Arg:
            domain_data: [n_domain, dim]
            bc_data:    [n_bc, dim]
            ic_data:    [n_ic, dim]
            pde_data:   [n_pde, dim]
        """
        self.domain_pos = torch.FloatTensor(domain_pos) if domain_pos is not None else None
        self.bc_pos = torch.FloatTensor(bc_pos) if bc_pos is not None else None
        self.bc_value = torch.FloatTensor(bc_value) if bc_value is not None else None
        self.ic_pos = torch.FloatTensor(ic_pos) if ic_pos is not None else None
        self.ic_value = torch.FloatTensor(ic_value) if ic_value is not None else None
        self.pde_pos = torch.FloatTensor(pde_pos) if pde_pos is not None else None
        self.pde_value = torch.FloatTensor(pde_value) if pde_value is not None else None
        
        self.n_domain = len(domain_pos) if domain_pos is not None else 0
        self.n_bc = len(bc_pos) if bc_pos is not None else 0
        self.n_ic = len(ic_pos) if ic_pos is not None else 0
        self.n_pde = len(pde_pos) if pde_pos is not None else 0
        
    def __len__(self):
        return max(self.n_domain, self.n_bc, self.n_ic, self.n_pde)
       
    def __getitem__(self, idx):
        item = {}
        
        if self.n_domain > 0:
            domain_idx = idx % self.n_domain
            item['domain'] = self.domain_pos[domain_idx]
        
        if self.n_bc > 0:
            bc_idx = idx % self.n_bc
            item['bc_pos'] = self.bc_pos[bc_idx]
            item['bc_val'] = self.bc_value[bc_idx]
        
        if self.n_ic > 0:
            ic_idx = idx % self.n_ic
            item['ic_pos'] = self.ic_pos[ic_idx]
            item['ic_val'] = self.ic_value[ic_idx]
        
        if self.n_pde > 0:
            pde_idx = idx % self.n_pde
            item['pde_pos'] = self.pde_pos[pde_idx]
            item['pde_val'] = self.pde_value[pde_idx]        
        return item

def pinn_collate_fn(batch):
    result = {'domain': [], 
              'bc_pos': [], 'bc_val':[],
              'ic_pos': [], 'ic_val':[],
              'pde_pos': [], 'pde_val':[]}
    
    for sample in batch:
        for key in result:
            if key in sample:
                result[key].append(sample[key])
    
    # 将列表转换为张量
    for key in result:
        if result[key]:  # 如果列表不为空
            result[key] = torch.stack(result[key])
        else:
            result[key] = None
    
    return result

class PINNTrainer:
    def __init__(self, model, pde_func, bc_func=None, ic_func=None, device='cpu'):
        """
        Inital trainer
        
        Arg:
            model: PINN model
            pde_func: PDE residual function
            bc_func: BC function
            ic_func: IC function
            device: （'cpu' or 'cuda'）
        """
        self.model = model.to(device)
        self.pde_func = pde_func
        self.bc_func = bc_func
        self.ic_func = ic_func
        self.device = device
        
        # history
        self.history = {
            'total_loss': [],
            'pde_loss': [],
            'bc_loss': [],
            'ic_loss': [],
            'data_loss': []
        }
    
    def compute_pde_loss(self, x):
        """
        Compute pde loss
        """
        x.requires_grad_(True)
        
        u = self.model(x)
        pde_residual = self.pde_func(x, u)
        pde_loss = torch.mean(pde_residual**2)
        
        return pde_loss
    
    def compute_bc_loss(self, x_bc, u_bc_true):
        """
        Arg:
            x_bc: location of bc
            u_bc_true: true value
        """
        u_bc_pred = self.model(x_bc)
        if self.bc_func is not None:
            bc_residual = self.bc_func(self.model, x_bc, u_bc_pred, u_bc_true)
            bc_loss = torch.mean(bc_residual**2)
        else:
            bc_loss = torch.mean((u_bc_pred - u_bc_true)**2)
            
        return bc_loss
    
    def compute_ic_loss(self, x_ic, u_ic_true):
        """
        Arg:
            x_ic: inital pnts
            u_ic_true: true value
        """
        u_ic_pred = self.model(x_ic)
        
        if self.ic_func is not None:
            ic_residual = self.ic_func(x_ic, u_ic_pred, u_ic_true)
            ic_loss = torch.mean(ic_residual**2)
        else:
            ic_loss = torch.mean((u_ic_pred - u_ic_true)**2)
            
        return ic_loss
    
    def compute_data_loss(self, x_data, u_data_true):
        """
        Arg:
            x_data: data points
            u_data_true: true value
        """
        u_data_pred = self.model(x_data)
        data_loss = torch.mean( (u_data_pred - u_data_true)**2/(u_data_true + 1e-10)**2 )
        
        return data_loss
    
    def train(self, dataloader, optimizer, scheduler=None, epochs=100, 
            pde_weight=1.0, bc_weight=1.0, ic_weight=1.0, data_weight=1.0,
            verbose=True, print_every=100, save_path='./'):
        """
        Arg:
            dataloader: 
            optimizer: 
            epochs: 
            pde_weight: PDE loss weight
            bc_weight: B.C. loss weight
            ic_weight: I.C loss weight
            data_weight: data loss weight
            verbose: If print info
            print_every: print interval
        """
        self.model.train()
        start_time = time.time()
        
        for epoch in range(epochs):
            total_loss_epoch = 0
            pde_loss_epoch = 0
            bc_loss_epoch = 0
            ic_loss_epoch = 0
            data_loss_epoch = 0
            
            for i, batch in enumerate(dataloader):
                
                # 定义closure函数用于LBFGS优化器
                def closure():
                    optimizer.zero_grad()
                    total_loss = 0
                    
                    # interior PDE loss
                    if batch['domain'] is not None:
                        x_domain = batch['domain'].to(self.device)
                        pde_loss = self.compute_pde_loss(x_domain)
                        total_loss += pde_weight * pde_loss
                    
                    # B.C loss
                    if batch['bc_pos'] is not None:
                        x_bc = batch['bc_pos'].to(self.device)
                        u_bc_true = batch['bc_val'].to(self.device)
                        bc_loss = self.compute_bc_loss(x_bc, u_bc_true)
                        total_loss += bc_weight * bc_loss
                    
                    # I.C loss
                    if batch['ic_pos'] is not None:
                        x_ic = batch['ic_pos'].to(self.device)
                        u_ic_true = batch['ic_val'].to(self.device)
                        ic_loss = self.compute_ic_loss(x_ic, u_ic_true)
                        total_loss += ic_weight * ic_loss
                    
                    # data loss (supervisor)
                    if batch['pde_pos'] is not None:
                        x_data = batch['pde_pos'].to(self.device)
                        u_data_true = batch['pde_val'].to(self.device)
                        data_loss = self.compute_data_loss(x_data, u_data_true)
                        total_loss += data_weight * data_loss
                    
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    return total_loss
                
                # 对于LBFGS优化器使用closure，其他优化器使用标准流程
                if isinstance(optimizer, torch.optim.LBFGS):
                    total_loss = optimizer.step(closure)
                    total_loss = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
                else:
                    total_loss = closure()
                    optimizer.step()
                    total_loss = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
                
                if scheduler != None:
                    scheduler.step(total_loss)
                
                total_loss_epoch += total_loss
                # 这里需要重新计算各个loss的值用于记录
                # 可以调用closure()重新计算，或者修改closure函数返回各个loss
                
                if i == 1:
                    break
            
            # 记录平均损失
            n_batches = len(dataloader)
            # record mean loss
            n_batches = len(dataloader)
            self.history['total_loss'].append(total_loss_epoch / n_batches)
            self.history['pde_loss'].append(pde_loss_epoch / n_batches)
            self.history['bc_loss'].append(bc_loss_epoch / n_batches)
            self.history['ic_loss'].append(ic_loss_epoch / n_batches)
            self.history['data_loss'].append(data_loss_epoch / n_batches)
            
            # printf
            if verbose and (epoch % print_every == 0 or epoch == epochs - 1):
                print(f'Epoch {epoch}/{epochs}, Total Loss: {self.history["total_loss"][-1]:.12f}')

            
            if epoch % (epochs // 20) == 0:
                torch.save(self.model, f"{save_path}/check_pnt_{epoch}.pth")
        
        end_time = time.time()
        print(f'Training completed in {end_time - start_time:.2f} seconds')
    
    def plot_training_history(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(len(self.history['total_loss']))
        
        ax.semilogy(epochs, self.history['total_loss'], label='Total Loss', linewidth=2)
        ax.semilogy(epochs, self.history['pde_loss'], label='PDE Loss', linestyle='--')
        ax.semilogy(epochs, self.history['bc_loss'], label='BC Loss', linestyle='--')
        ax.semilogy(epochs, self.history['ic_loss'], label='IC Loss', linestyle='--')
        ax.semilogy(epochs, self.history['data_loss'], label='Data Loss', linestyle='--')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training History')
        ax.legend()
        ax.grid(True, which="both", ls="--")
        
        plt.show()

def gradients(u, x, order=1):
    u.requires_grad_(True)
    x.requires_grad_(True)
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,
                                   only_inputs=True, )[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)
    
    
def batched_jacobian(batched_y: torch.Tensor, 
                     batched_x: torch.Tensor, 
                     need_higher_grad: bool = True) -> torch.Tensor:
    """
    计算批处理输入下的雅可比矩阵
    
    Args:
        batched_y (torch.Tensor): N x y_shape 的输出张量
        batched_x (torch.Tensor): N x x_shape 的输入张量
        need_higher_grad (bool, optional): 是否需要计算高阶梯度，默认为 True
    
    Returns:
        torch.Tensor: N x y_shape x x_shape 的雅可比矩阵
    """
    batch_size = batched_x.shape[0]
    
    # 初始化雅可比矩阵
    y_shape = batched_y.shape[1:]
    x_shape = batched_x.shape[1:]
    J = torch.zeros(batch_size, *y_shape, *x_shape, device=batched_x.device)
    
    # 对每个输出维度计算梯度
    for i in range(batched_y.shape[1]):
        # 计算第i个输出维度对输入的梯度
        grad_outputs = torch.zeros_like(batched_y)
        grad_outputs[:, i] = 1.0
        
        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=batched_y,
            inputs=batched_x,
            grad_outputs=grad_outputs,
            create_graph=need_higher_grad,
            retain_graph=True,
            allow_unused=False
        )[0]
        
        # 将梯度存入雅可比矩阵
        J[:, i] = gradients
    
    return J
    
#%%
if __name__ == "__main__":
    # 
    N, y_dim, x_dim = 32, 3, 4
    batched_x = torch.randn(N, x_dim, requires_grad=True)
    batched_y = batched_x ** 2  # 简单变换

    # 计算雅可比矩阵
    J = batched_jacobian(batched_y, batched_x)
    print(J.shape)  # 应输出: torch.Size([32, 3, 4])