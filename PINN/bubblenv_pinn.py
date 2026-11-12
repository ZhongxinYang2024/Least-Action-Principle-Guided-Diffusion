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
from data.CustomDataLoader import CustomDataset_BubbleNV



#%%
if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    dtype=torch.float32
    torch.set_default_dtype(dtype)
    
    save_path = './PINN/log/bubblenv'
    os.makedirs(save_path, exist_ok=True)
    mode = 'train'
    

    # 1. define PDE residual function
    def residual_fun(x, u):
        '''
        x[t,q0,p0,k,m]
        u[q,p]
        Arg: 
        x.shape :(num, ch_in)
        u.shape :(num, ch_out)
        batch_jacobian(u,x).shape :(num, ch_out, ch_in)
        '''
        
        dqdt = batched_jacobian(u,x)[:, 0, 0].unsqueeze(-1)
        dpdt = batched_jacobian(u,x)[:, 1, 0].unsqueeze(-1)
        
        residual1 = (dqdt - u[:,1:2] / x[:,4:5])**2  # dqdt - p/m = 0
        residual2 = (dpdt + u[:,0:1] * x[:,3:4] )**2  # dpdt + kq = 0
        residual = torch.sqrt(residual1 + residual2)
        return residual
    
    # 2. generate training data
    num_T = 1
    ch_in = 5; ch_out = 2
    qq_begin = -1.; qq_end = 1.
    pp_begin = -1.; pp_end = 1
    kk_begin = 0.5; kk_end = 1.5
    mm_begin = 1.0; mm_end = 1.
        
    # interior data
    n_domain = 8000 * 128
    qq_domain = np.random.uniform(qq_begin, qq_end, (n_domain,1))
    pp_domain = np.random.uniform(pp_begin, pp_end, (n_domain,1))
    kk_domain = np.random.uniform(kk_begin, kk_end, (n_domain,1))    
    mm_domain = np.random.uniform(mm_begin, mm_end, (n_domain,1))
    base_domain = np.random.uniform(0, 2. * np.pi * num_T / 1.0, (n_domain,1))
    omega_domain = np.sqrt(kk_domain / mm_domain)
    tt_domain = base_domain / omega_domain
    x_domain  = np.concatenate([tt_domain, qq_domain, pp_domain, kk_domain, mm_domain], axis=-1)

    # B.C. data 
    pass

    # I.C. data #[t,q0,p0,kk,mm]
    n_ic = 8000 * 1
    x_ic = np.ones((n_ic, ch_in)) * 0.0; 
    x_ic[:,1] = np.random.uniform(qq_begin, qq_end, (n_ic,))
    x_ic[:,2] = np.random.uniform(pp_begin, pp_end, (n_ic,))
    x_ic[:,3] = np.random.uniform(kk_begin, kk_end, (n_ic,))
    x_ic[:,4] = np.random.uniform(mm_begin, mm_end, (n_ic,))
    u_ic = np.ones((n_ic, ch_out)) * 0.0 
    u_ic[:,0] = x_ic[:,1]
    u_ic[:,1] = x_ic[:,2]
    
    
    # PDE data
    resolution = 128
    dataset = CustomDataset_BubbleNV(resolution=resolution)
    u_pde = dataset.data.squeeze(-1).permute(0,2,1).reshape(-1, ch_out) # [B, C, H, 1]
    
    base_pde =   torch.linspace(0, 2. * np.pi * num_T / 1.0, resolution)
    tt_pde = ((1. / torch.sqrt(dataset.labels[:,2]/mm_begin)[:, None]) * base_pde[None,:]).unsqueeze(1).permute(0,2,1).reshape(-1,1)
    cond_pde = dataset.labels.unsqueeze(-1).repeat(1,1,resolution).permute(0,2,1).reshape(-1, dataset.num_cond) # [B, 3] 3:[qq,pp,kk]


    mm_pde = torch.full_like(tt_pde, mm_begin)
    x_pde = torch.cat([tt_pde, cond_pde, mm_pde], dim=-1).numpy()
    
    
    # 3. create dataset
    dataset = PINNDataset(domain_pos=x_domain, 
                        ic_pos=x_ic, ic_value=u_ic,
                        pde_pos=x_pde, pde_value=u_pde)
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=True, collate_fn=pinn_collate_fn)
    
    # 4. create the model
    layers = [ch_in, 32, 64, 64, 32, ch_out]  #
    model = PhysicsInformedNN(layers)

    match mode.lower():
        case 'train':
            # 5. create traner
            trainer = PINNTrainer(model, pde_func=residual_fun, device=device)
            
            # 6. set optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            # 7. train the model
            trainer.train(dataloader, 
                        optimizer, epochs=5000, 
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
                q0_preset = 0.0; p0_preset = 1.0; kk_preset = 1.0; mm_preset = 1.0; T_preset = 2. * np.pi * num_T / np.sqrt(kk_preset/mm_preset) * 1.20
                t_test = np.linspace(0,  T_preset, resolution).reshape(-1,1)
                q0_test = np.full_like(t_test,q0_preset)
                p0_test = np.full_like(t_test,p0_preset)
                kk_test = np.full_like(t_test,kk_preset)
                mm_test = np.full_like(t_test,mm_preset)
                
                # perdict
                test_points = np.concatenate([t_test, q0_test, p0_test, kk_test, mm_test], axis=-1)
                test_tensor = torch.FloatTensor(test_points).to(device)
                u_pred = model(test_tensor).cpu().numpy()
                
                # plot
                plt.figure(figsize=(10, 6))
                plt.plot(t_test, u_pred[:,0],label='q')
                plt.plot(t_test, u_pred[:,1],label='p')
                plt.xlabel('t')
                plt.title(f'q0{q0_preset:.2f}_p0{p0_preset:.2f}_kk{kk_preset:.2f}_mm{mm_preset:.2f}')
                plt.show()
            np.savetxt(f'{save_path}/q0{q0_preset:.1f}_p0{p0_preset:.1f}_kk{kk_preset:.1f}_mm{mm_preset:.1f}_T{T_preset:.1f}.dat', np.concatenate([test_points, u_pred],axis=-1), 
                       fmt='%.6e', header='t    q0    p0    kk    mm    q    p',  
                       delimiter='\t')

        case _:
            raise ValueError(f'mode {mode} not regonized.')