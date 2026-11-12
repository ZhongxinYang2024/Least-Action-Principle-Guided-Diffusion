import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.utils.weight_norm as weight_norm
from torch.utils.data import DataLoader, TensorDataset
import os, sys
sys.path.append(f'{os.getcwd()}')
from data.CustomDataLoader import CustomDataset_Foil
from base_pinn import batched_jacobian

def compute_unit_normal_vectors_torch(xx, yy):
    """
    使用PyTorch计算曲线上的单位法向量 - 完全向量化
    
    参数:
    xx: [B, n]  x坐标张量
    yy: [B, n]  y坐标张量
    
    返回:
    nx: [B, n]  单位法向量的x分量
    ny: [B, n]  单位法向量的y分量
    """
    # 确保是浮点类型
    xx = xx.float()
    yy = yy.float()
    
    # 计算切线向量 - 使用梯度计算
    # 对于离散点，我们可以使用差分来近似导数
    dx = xx[:, 2:] - xx[:, :-2]  # 中心差分
    dy = yy[:, 2:] - yy[:, :-2]
    
    # 处理边界点 - 使用前向/后向差分
    dx_left = xx[:, 1] - xx[:, 0]  # 左边界
    dy_left = yy[:, 1] - yy[:, 0]
    dx_right = xx[:, -1] - xx[:, -2]  # 右边界
    dy_right = yy[:, -1] - yy[:, -2]
    
    # 组合完整的导数张量
    dx_full = torch.cat([
        dx_left.unsqueeze(1), 
        dx, 
        dx_right.unsqueeze(1)
    ], dim=1)
    
    dy_full = torch.cat([
        dy_left.unsqueeze(1), 
        dy, 
        dy_right.unsqueeze(1)
    ], dim=1)
    
    # 计算切线长度
    tangent_length = torch.sqrt(dx_full**2 + dy_full**2 + 1e-10)
    
    # 单位切线向量
    unit_tx = dx_full / tangent_length
    unit_ty = dy_full / tangent_length
    
    # 单位法向量 (逆时针旋转90度)
    nx = - unit_ty
    ny =   unit_tx
    
    return nx, ny


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class PINN_NS(nn.Module):
    def __init__(self, layers):
        super(PINN_NS, self).__init__()
        self.num_layers = len(layers)
        temp = [];
        for l in range(1, self.num_layers):
            temp.append(weight_norm(torch.nn.Linear(layers[l-1], layers[l]), dim = 0))
            torch.nn.init.normal_(temp[l-1].weight)
        self.layers = torch.nn.ModuleList(temp)
        self.act = Swish()
        
    def forward(self, x):
        for i in range(0, self.num_layers-1):
            h = self.layers[i](x)
            if i == 0:
                x = self.act(h)
            elif i < self.num_layers-2:
                h = self.act(h)
                x = x + h
            else:
                return h

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.constant_(m.bias, 0.0)




class NS_PINN_Trainer:
    def __init__(self, grid_data, flow_data, label_data, hidden_layers=[64, 64, 64]):
        """
        grid_data: [B, 2, Nx, Ny] 网格数据
        flow_data: [B, 3, Nx, Ny] 流场数据 (u, v, p)
        label_data: [B, num_cond] condition data
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 处理数据
        self.grid_data = grid_data.to(self.device)
        self.flow_data = flow_data.to(self.device)
        self.label_data = label_data.to(self.device)
        self.B, _, self.Nx, self.Ny = grid_data.shape
        
        
        # 初始化网络
        input_dim = 2 + label_data.shape[1]  # (x, y)
        output_dim = 3  # (u, v, p)
        layers = [input_dim] + hidden_layers + [output_dim]
        self.model = PINN_NS(layers).to(self.device).apply(init_weights)
                


    def compute_pde_loss(self, x, u):
        """计算NS方程残差损失"""
        # 计算梯度
        
        dudx = batched_jacobian(u,x)[:, 0, 0].unsqueeze(-1)
        dudy = batched_jacobian(u,x)[:, 0, 1].unsqueeze(-1)
        dvdx = batched_jacobian(u,x)[:, 1, 0].unsqueeze(-1)
        dvdy = batched_jacobian(u,x)[:, 1, 1].unsqueeze(-1)
        dpdx = batched_jacobian(u,x)[:, 2, 0].unsqueeze(-1) 
        dpdy = batched_jacobian(u,x)[:, 2, 1].unsqueeze(-1)

        
        
        # 连续性方程: du/dx + dv/dy = 0
        continuity_eq = dudx + dvdy
        
        # 动量方程 (无粘性)
        # x-momentum: u*du/dx + v*du/dy + dp/dx = 0
        x_momentum = u[:, 0:1] * dudx + u[:,1:2] * dudy + dpdx 
        
        # y-momentum: u*dv/dx + v*dv/dy + dp/dy = 0
        y_momentum = u[:, 0:1] * dvdx + u[:,1:2] * dvdy + dpdy 
        
        pde_loss = (torch.mean(continuity_eq**2) + 
                   torch.mean(x_momentum**2) + 
                   torch.mean(y_momentum**2))
        
        return pde_loss
    
    def compute_bc_loss(self, pred_flow, batch_X, batch_y, shape, mask_type='no_penetration'):
        """计算边界条件损失"""
        pred_flow_reshape = pred_flow.reshape(-1,shape[2], shape[3], 3).permute(0,3,1,2)
        batch_X_reshape =   batch_X.reshape(-1,shape[2], shape[3], 6).permute(0,3,1,2)
        #batch_y_reshape =   batch_y.reshape(-1,shape[2], shape[3], 3).permute(0,3,1,2)
        if mask_type == 'no_penetration':
            
            # 无穿透边界: 法向速度为零 (u·n = 0)
            u_pred = pred_flow_reshape[:, 0, 0, :]
            v_pred = pred_flow_reshape[:, 1, 0, :]
            
            # 获取法向量
            normal_x, normal_y = compute_unit_normal_vectors_torch(batch_X_reshape[:, 0, 0,:], batch_X_reshape[:, 1, 0, :])
            
            # 法向速度
            normal_velocity = u_pred * normal_x + v_pred * normal_y
            
            bc_loss = torch.mean(normal_velocity**2)
            
        elif mask_type == 'far_field':
            # 远场边界: 使用数据作为边界条件
            u_pred = pred_flow_reshape[:, 0, -1, :]
            v_pred = pred_flow_reshape[:, 1, -1, :]
            p_pred = pred_flow_reshape[:, 2, -1, :]
            
            u_true = torch.cos(batch_X_reshape[:, 2, -1, : ] * 10.0 / 180.0 * torch.pi) # sin(aoa)
            v_true = torch.sin(batch_X_reshape[:, 2, -1, : ] * 10.0 / 180.0 * torch.pi) # cos(aoa)
            p_true = torch.full_like(p_pred, 0.0) # 0.0
            
            bc_loss = (torch.mean((u_pred - u_true)**2) + 
                      torch.mean((v_pred - v_true)**2) + 
                      torch.mean((p_pred - p_true)**2) )
        
        return bc_loss
    
    def train(self, optimizer, scheduler, epochs=10000, bs=3, print_every=1, optimizer_type = 'lbfgs'):
        """训练模型"""
        losses = []
        optimizer_old = optimizer
        
        self.optimizer = optimizer
        self.scheduler = scheduler 
        
        for epoch in range(epochs):
            def closure():
                self.optimizer.zero_grad()
                
                rd = np.random.randint(0, self.B)
                std = (rd) % self.B
                edd = (bs + rd) % self.B
                if edd < std:
                    edd = self.B - 1
                raw_shape  = self.grid_data[ std: edd   ,...].shape
                batch_Xg = self.grid_data[ std: edd   ,...]
                batch_Xg = batch_Xg.permute(0, 2, 3, 1).reshape(-1, 2)
                batch_Xcond = self.label_data[std: edd, ...  ].unsqueeze(-1).unsqueeze(-1).repeat(1,  1,  raw_shape[2], raw_shape[3]).permute(0, 2, 3, 1).reshape(-1, 4)
                batch_X = torch.cat([batch_Xg, batch_Xcond], dim=-1)
                batch_X.requires_grad_(True)
                batch_y = self.flow_data[  std: edd  ,...]
                batch_y = batch_y.permute(0, 2, 3, 1).reshape(-1, 3)
                
                # 前向传播
                pred_flow = self.model(  batch_X  )
                
                # 数据损失
                data_loss = torch.mean((pred_flow - batch_y)**2)
                
                # PDE损失 (只在内部点计算)
                pde_loss = self.compute_pde_loss(batch_X, pred_flow)
                
                # 边界条件损失
                bc_no_penetration = self.compute_bc_loss(pred_flow, batch_X, batch_y,  raw_shape, 'no_penetration')
                bc_far_field = self.compute_bc_loss(pred_flow, batch_X, batch_y, raw_shape, 'far_field')
                bc_loss = bc_no_penetration + bc_far_field
                
                # 总损失
                total_loss = (self.lambda_data * data_loss + 
                            self.lambda_pde * pde_loss + 
                            self.lambda_bc * bc_loss)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        
                if epoch % print_every == 0:
                    print(f'Epoch {epoch}/{epochs}, Total Loss: {total_loss.item():.12f}, '
                      f'Data Loss: {data_loss.item():.12f}, PDE Loss: {pde_loss.item():.12f}, '
                      f'BC Loss: {bc_loss.item():.12f}')
                return total_loss
            
            # 反向传播
    
            match optimizer_type.lower():
                case 'adam' | 'adamw' | 'sgd' | 'sgdm' | 'gd':
                    loss = closure()
                    loss_value = loss.item()  
                    losses.append(loss_value)
                    self.optimizer.step()
                case 'lbfgs':                            
                    loss = self.optimizer.step(closure)
                    loss_value = loss.item() if isinstance(loss, torch.Tensor) else loss
                    losses.append(loss_value)
                    
                    self.optimizer = optimizer_old
                case _:
                    raise ValueError(f'optimizer_type {optimizer_type} not recognized.')
            if np.isnan(loss_value):
                self.model = torch.load(  f'./PINN/log/foil/ckpnt_{last_save}_{optimizer_type}.pth', weights_only=False)
                print('Failed, Load previous model')
                continue
            
            losses.append(loss.item())
            
            if epoch % max((epochs // 100 ),1) == 0:
                last_save = epoch
                torch.save(self.model, f'./PINN/log/foil/ckpnt_{epoch}_{optimizer_type}.pth')
        
        return losses

def eval(ckt_path, aoa_preset = 0.0, creal_preset = -0.17241, cimag_preset = 0.0, radius_preset = 1.2268):
    from HamiltonSys.ZhukovskyTransfomation import GetAFoil
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = torch.load( ckt_path, weights_only=False)
    model.eval()
    with torch.no_grad():
        # create test grid
        

        _, Z_Mesh = GetAFoil(aoa_preset, creal_preset, cimag_preset, radius_preset)
        XX, YY = Z_Mesh.real, Z_Mesh.imag

        XX -=  (creal_preset - radius_preset) + 1./(creal_preset - radius_preset)
        XX /= (2. * radius_preset - 2. * radius_preset/ (creal_preset**2 - radius_preset**2 )   )
        YY /= (2. * radius_preset - 2. * radius_preset/ (creal_preset**2 - radius_preset**2 )   )
        
        x_test, y_test = XX.reshape(-1,1) / 25.0, YY.reshape(-1,1)/25.0
        aoa_test = np.full_like(x_test, aoa_preset) / 10.0
        creal_test = np.full_like(x_test, creal_preset)
        cimag_test = np.full_like(x_test, cimag_preset)
        radius_test = np.full_like(x_test,radius_preset)
        
        # perdict
        test_points = np.concatenate([x_test, y_test, aoa_test, creal_test, cimag_test, radius_test], axis=-1)
        test_tensor = torch.FloatTensor(test_points).to(device)
        U_pred = model(test_tensor).cpu().numpy()
        UU, VV, PP = U_pred[:,0].reshape(XX.shape),  U_pred[:,1].reshape(XX.shape), U_pred[:,2].reshape(XX.shape) 
        # plot
        fig = plt.figure(figsize=(15, 5))  # 建议增加图形宽度以容纳colorbar

        ax1 = fig.add_subplot(1,3,1)
        ax2 = fig.add_subplot(1,3,2)
        ax3 = fig.add_subplot(1,3,3)

        # 绘制等高线图并保存返回对象
        cf1 = ax1.contourf(XX, YY, UU)
        cf2 = ax2.contourf(XX, YY, VV)
        cf3 = ax3.contourf(XX, YY, PP)

        # 为每个子图添加colorbar
        fig.colorbar(cf1, ax=ax1)
        fig.colorbar(cf2, ax=ax2)
        fig.colorbar(cf3, ax=ax3)

        # 可选：添加标题
        ax1.set_title('UU')
        ax2.set_title('VV')
        ax3.set_title('PP')

        plt.tight_layout()  # 自动调整子图间距
        plt.show()

    return XX, YY, UU, VV, PP

# 主训练流程
if __name__ == "__main__":
    
    yzx = 0
    if yzx:
        
        # 生成非均匀网格示例数据
        print("生成非均匀网格示例数据...")
        dataset = CustomDataset_Foil(96, 128, IFall=True)
        dataset.IfScaleFoil2One()
        dataset.IfReverseScalar()
        grid_data = dataset.labels_distributed
        flow_data = dataset.flowdata
        label_data = dataset.labels
        grid_data = grid_data / 25. # x, y /25.0
        flow_data[:,2,...] = ( flow_data[:,2,...] - 1. / 1.4 / (0.1)**2 ) 
        label_data[:, 0] /= 10.0 # aoa / 10.0
        
        # 创建训练器
        print("初始化PINN训练器...")
        trainer = NS_PINN_Trainer(grid_data, flow_data,  label_data, hidden_layers=[128, 128, 128, 128, 128, 128, 128, 128, 128])
        

        # 训练模型
        print("开始训练...")
        trainer.model = torch.load(f'./PINN/log/foil/ckpnt_13500_adam.pth', weights_only=False)

        trainer.lambda_data, trainer.lambda_pde, trainer.lambda_bc = 2e4, 1, 1e3
        optimizer = optim.Adam(trainer.model.parameters(), lr=2e-4, weight_decay=1e-5)
        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5000, factor=0.7)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
        losses = trainer.train(optimizer, scheduler, epochs=50000,  bs=3, print_every=1, optimizer_type='adam')
        
        
        # trainer.lambda_data, trainer.lambda_pde, trainer.lambda_bc = 2e4, 1, 1e3
        # optimizer = torch.optim.LBFGS( trainer.model.parameters(),
        #             lr = 2e-3,
        #             max_iter=1000,
        #             history_size=1000,
        #             tolerance_grad=1e-6,
        #             tolerance_change=1e-10,
        #             line_search_fn='strong_wolfe')
        # scheduler = None
        # losses = trainer.train(optimizer, scheduler, epochs=10000, bs=2, print_every=1, optimizer_type = 'lbfgs')
    
        # 绘制训练损失
        plt.figure(figsize=(10, 6))
        plt.semilogy(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('PINN Training Loss')
        plt.grid(True)
        plt.show()
    else:
        import scipy.io as sio
        # # 测试预测
        # print("测试预测...")
        aoa = 0.0;
        creal = -0.17241;
        cimag = 0.0
        radius = 1.2268
        X, Y, U, V, P = eval(f'./PINN/log/foil/ckpnt_49500_bst_adam6.pth', aoa, creal, cimag, radius)
         
        data_dict = {
                'XX': X,
                'YY': Y,
                'UU': U,
                'VV': V,
                'PP': P, 
                'aoa': aoa, 
                'creal': creal,
                'cimag': cimag,
                'R': radius
                    }
        
        save_path = './PINN/log/foil'
        filename  = f'{save_path}/Aoa{aoa:.1f}_cr{creal:.1f}_ci{cimag:.1f}_r{radius:.1f}_pinn.mat'
        sio.savemat(filename, data_dict)