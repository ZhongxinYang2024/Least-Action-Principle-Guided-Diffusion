
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


##
from base_pinn import PINNDataset, PhysicsInformedNN, pinn_collate_fn, PINNTrainer
from base_pinn import batched_jacobian,gradients

if __name__ == '__main__':    
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    dtype=torch.float32
    torch.set_default_dtype(dtype)
    
    save_path = './log/wave_eqn'
    os.makedirs(save_path, exist_ok=True)
    mode = 'eval'
    

    # 1. define PDE residual function
    def burgers_residual(x, u):
        
        u_t = gradients(u,x)[:, 0:1]
        
        u_tt = gradients(u_t,x)[:, 0:1]
        
        u_x = gradients(u,x)[:, 1:2]
        
        u_xx = gradients(u_x,x)[:, 1:2]
        
        
        residual = u_tt - 1.0 * u_xx 
        return residual
    
    # 2. generate training data
    T_simulation = 1.0
    x_max = 1.0
    # interior data
    n_domain = 1000
    xx_domain = np.random.uniform(0, x_max, (n_domain, 1))  # [t, x]
    tt_domain = np.random.uniform(0, T_simulation, (n_domain,1))
    x_domain = np.concatenate([xx_domain, tt_domain], axis=-1)
    
    # B.C. data 
    n_bc = 100
    x_bc_left = np.ones((n_bc//2, 2)) * 0.0
    x_bc_right = np.ones((n_bc//2, 2)) * x_max
    x_bc_left[:,0] = np.linspace(0, T_simulation, n_bc//2)
    x_bc_right[:,0] = np.linspace(0, T_simulation, n_bc//2)
    x_bc = np.concatenate([x_bc_left, x_bc_right], axis=0)
    u_bc = np.zeros((n_bc, 1))
    
    # I.C. data
    n_ic = 100
    x_ic = np.random.uniform(0, x_max, (n_ic, 2)); x_ic[:,0] = 0.0
    u_ic = -np.sin(np.pi * x_ic[:, 1]).reshape(-1, 1)  # 初始条件: u(0, x) = -sin(πx)
    
    # 3. 创建数据集和数据加载器
    dataset = PINNDataset(domain_pos=x_domain, 
                        bc_pos=x_bc, bc_value=u_bc,
                        ic_pos=x_ic, ic_value=u_ic)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=pinn_collate_fn)
    
    # 4. create the model
    layers = [2, 20, 20, 20, 1]  # 输入维度为2 (t, x)，输出维度为1 (u)
    model = PhysicsInformedNN(layers)

    match mode.lower():
        case 'train':
            # 5. create traner
            trainer = PINNTrainer(model, pde_func=burgers_residual, device=device)
            
            # 6. set optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            # 7. train the model
            trainer.train(dataloader, 
                        optimizer, epochs=1000, 
                        pde_weight=1.0, bc_weight=1.0, ic_weight=1.0,
                        verbose=True, print_every=100)
            
            # 8. plot the history
            torch.save(model, f"{save_path}/check_pnt.pth")
            trainer.plot_training_history()
        case 'eval':
            model = torch.load( f"{save_path}/check_pnt.pth", weights_only=False)
            model.eval()
            with torch.no_grad():
                # create test grid
                t_test = np.linspace(0, T_simulation, 50)
                x_test = np.linspace(0, x_max, 50)
                t_grid, x_grid = np.meshgrid(t_test, x_test)
                
                # perdict
                test_points = np.vstack([t_grid.flatten(), x_grid.flatten()]).T
                test_tensor = torch.FloatTensor(test_points).to(device)
                u_pred = model(test_tensor).cpu().numpy().reshape(t_grid.shape)
                
                # plot
                plt.figure(figsize=(10, 6))
                plt.contourf(t_grid, x_grid, u_pred, 50, cmap='jet')
                plt.colorbar()
                plt.xlabel('t')
                plt.ylabel('x')
                plt.show()
        case _:
            raise ValueError(f'mode {mode} not regonized.')