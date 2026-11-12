import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import sys

##
from base_pinn import PINNDataset, PhysicsInformedNN, pinn_collate_fn, PINNTrainer
from base_pinn import batched_jacobian,gradients

sys.path.append(f'{os.getcwd()}')
from data.CustomDataLoader import CustomDataset_FreeFall



#%%
if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    dtype=torch.float32
    torch.set_default_dtype(dtype)
    
    save_path = './PINN/log/freefall'
    os.makedirs(save_path, exist_ok=True)
    mode = 'eval'
    

    # 1. define PDE residual function
    def residual_fun(x, u):
        '''
        x[t,g]
        u[q,v]
        Arg: 
        x.shape :(num, ch_in)
        u.shape :(num, ch_out)
        batch_jacobian(u,x).shape :(num, ch_out, ch_in)
        '''
        
        dxdt = batched_jacobian(u,x)[:, 0, 0].unsqueeze(-1)
        dvdt = batched_jacobian(u,x)[:, 1, 0].unsqueeze(-1)
        
        
        residual = torch.sqrt( (dxdt - u[:,[1]])**2 + (dvdt - x[:,[1]])**2 )
        return residual
    
    # 2. generate training data
    T_simulation = 2.0
    g_begin = 5.; g_end = 15.0
    ch_in = 2; ch_out = 2
    
    # interior data
    n_domain = 1000
    tt_domain = np.random.uniform(0, T_simulation, (n_domain,1))
    gg_domain = np.random.uniform(g_begin, g_end, (n_domain,1))
    x_domain = np.concatenate([tt_domain,gg_domain], axis=-1)
    
    # B.C. data 
    pass

    # I.C. data
    n_ic = 100
    x_ic = np.ones((n_ic, ch_in)) * 0.0; x_ic[:,1] = np.random.uniform(g_begin, g_end, (n_ic,))
    u_ic = np.ones((n_ic, ch_out)) * 0.0 
    
    # PDE data
    resolution = 128; num_g = 1000
    dataset = CustomDataset_FreeFall(resolution=resolution)
    dataset.IfReverseScalar()
    u_pde = dataset.data.squeeze(-1).permute(0,2,1).reshape(-1, ch_out)
    cond_pde = dataset.labels.unsqueeze(-1).repeat(1,1,resolution).permute(0,2,1).reshape(-1, dataset.num_cond)
    tt_pde =   torch.linspace(0, T_simulation, resolution).unsqueeze(0).unsqueeze(0).repeat(num_g,1,1).permute(0,2,1).reshape(-1, 1)
    x_pde = torch.cat([tt_pde, cond_pde], dim=-1).numpy()
    
    
    # 3. create dataset
    dataset = PINNDataset(domain_pos=x_domain, 
                        ic_pos=x_ic, ic_value=u_ic,
                        pde_pos=x_pde, pde_value=u_pde)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=pinn_collate_fn)
    
    # 4. create the model
    layers = [ch_in, 20, 20, 20, ch_out]  #
    model = PhysicsInformedNN(layers)

    match mode.lower():
        case 'train':
            # 5. create traner
            trainer = PINNTrainer(model, pde_func=residual_fun, device=device)
            
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
                g_preset =10.0; T_tot = T_simulation * 2.0
                t_test = np.linspace(0, T_tot, 50).reshape(-1,1)
                t_grid = t_test
                g_test = np.full_like(t_test,g_preset)
                
                # perdict
                test_points = np.concatenate([t_test, g_test], axis=-1)
                test_tensor = torch.FloatTensor(test_points).to(device)
                u_pred = model(test_tensor).cpu().numpy()
                
                # plot
                plt.figure(figsize=(10, 6))
                plt.plot(t_grid, u_pred[:,0],label='q')
                plt.plot(t_grid, u_pred[:,1],label='p')
                plt.xlabel('t')
                plt.title(f'G = {g_preset:.2f}')
                plt.show()
            np.savetxt(f'{save_path}/G{g_preset:.1f}_T{T_tot:.1f}.dat', np.concatenate([test_points, u_pred],axis=-1), 
                       fmt='%.6e', header='t    g    q    p',  
                       delimiter='\t')

        case _:
            raise ValueError(f'mode {mode} not regonized.')