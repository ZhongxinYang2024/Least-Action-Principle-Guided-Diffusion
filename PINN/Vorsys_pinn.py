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
from data.CustomDataLoader import CustomDataset_VortexSys



#%%
if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    dtype=torch.float32
    torch.set_default_dtype(dtype)
    
    save_path = './PINN/log/vorsys'
    os.makedirs(save_path, exist_ok=True)
    mode = 'eval'
    

    # 1. define PDE residual function
    def residual_fun(x, u):
        '''
        x[t,q0_2x,gamma2]
        u[x1,y1,x2,y2]
        Arg: 
        x.shape :(num, ch_in)
        u.shape :(num, ch_out)
        batch_jacobian(u,x).shape :(num, ch_out, ch_in)
        '''
        
        dx1dt = batched_jacobian(u,x)[:, 0, 0].unsqueeze(-1)
        dy1dt = batched_jacobian(u,x)[:, 1, 0].unsqueeze(-1)
        dx2dt = batched_jacobian(u,x)[:, 2, 0].unsqueeze(-1)
        dy2dt = batched_jacobian(u,x)[:, 3, 0].unsqueeze(-1)
        
        Dx = ( u[:,0:1]-  u[:,2:3] )
        Dy = ( u[:,1:2] - u[:,3:4] )
        rr = torch.sqrt( Dx**2. + Dy**2. )
        residual1 = (dx1dt - (- 0.5 / np.pi *  x[:,2:3] / rr * (Dy/rr)  ) )**2   # dq1 - pHpp1
        residual2 = (dy1dt - (+ 0.5 / np.pi *  x[:,2:3] / rr * (Dx/rr)  ) )**2   # dp1 + pHpq1
        residual3 = (dx2dt - (- 0.5 / np.pi *      1.0  / rr * (-Dy/rr)  ) )**2   # dq2 - pHpp2
        residual4 = (dy2dt - (+ 0.5 / np.pi *      1.0  / rr * (-Dx/rr)  ) )**2   # dp1 + pHpq1
        residual = torch.sqrt(residual1 + residual2 + residual3 + residual4)
        return residual
    
    # 2. generate training data
    num_T = 1
    ch_in = 3; ch_out = 4
    q_2x_begin = 0.5; q_2x_end = 2.0
    gamma2_begin = -0.5; gamma2_end = 2.0
        
    # interior data
    n_domain = 2000 * 128
    q_2x_domain = np.random.uniform(q_2x_begin, q_2x_end, (n_domain,1))
    gamma2_domain = np.random.uniform(gamma2_begin, gamma2_end, (n_domain,1))
    base_domain = np.random.uniform(0, 2. * np.pi * num_T / 1.0, (n_domain,1))
    omega_domain = (gamma2_domain + 1.0) / 2.0 / np.pi / q_2x_domain**2
    tt_domain = base_domain / omega_domain
    x_domain  = np.concatenate([tt_domain, q_2x_domain, gamma2_domain], axis=-1)

    # B.C. data 
    pass

    # I.C. data #[t, q0_2x, gamma2]
    n_ic = 2000 * 1
    x_ic = np.ones((n_ic, ch_in)) * 0.0; 
    x_ic[:,1] = np.random.uniform(q_2x_begin, q_2x_end, (n_ic,))
    x_ic[:,2] = np.random.uniform(gamma2_begin, gamma2_end, (n_ic,))
    u_ic = np.ones((n_ic, ch_out)) * 0.0  # [q_1x,p_1y,q_2x,p_2y]
    u_ic[:,2] = x_ic[:,1]    
    
    
    # PDE data
    resolution = 128
    dataset = CustomDataset_VortexSys(resolution=resolution)
    data_unprocessed = dataset.data
    # data_processed = data_unprocessed
    # data_processed[:,0,...] *= 1.0 # q_1x
    # data_processed[:,1,...] *= 1.0 # p_1y
    # data_processed[:,2,...] *= torch.sqrt(torch.abs(dataset.labels[:,1])).unsqueeze(-1).unsqueeze(-1).repeat(1, resolution, 1)
    # data_processed[:,3,...] *= (torch.sign(dataset.labels[:,1])*torch.sqrt(torch.abs(dataset.labels[:,1]))).unsqueeze(-1).unsqueeze(-1).repeat(1, resolution, 1)
    u_pde = data_unprocessed.squeeze(-1).permute(0,2,1).reshape(-1, ch_out) # [B, C, H, 1] -> [B*H, C]
    
    base_pde =   torch.linspace(0, 2. * np.pi * num_T / 1.0, resolution)
    tt_pde = ( ( 1. /( (dataset.labels[:,1] + 1.0)/ 2. / torch.pi / dataset.labels[:,0]**2 ) )[:, None] * base_pde[None,:]).unsqueeze(1).permute(0,2,1).reshape(-1,1)
    cond_pde = dataset.labels.unsqueeze(-1).repeat(1,1,resolution).permute(0,2,1).reshape(-1, dataset.num_cond) # [B, 2] 2:[q0_2x,gamma2]
    
    x_pde = torch.cat([tt_pde, cond_pde], dim=-1).numpy()
    
    
    # 3. create dataset
    dataset = PINNDataset(domain_pos=x_domain, 
                        ic_pos=x_ic, ic_value=u_ic,
                        pde_pos=x_pde, pde_value=u_pde)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True, collate_fn=pinn_collate_fn)
    
    # 4. create the model
    layers = [ch_in, 32, 64, 64, 32, ch_out]  #
    model = PhysicsInformedNN(layers)

    match mode.lower():
        case 'train':
            # 5. create traner
            trainer = PINNTrainer(model, pde_func=residual_fun, device=device)
            
            # 6. set optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
            
            # 7. train the model
            trainer.train(dataloader, 
                        optimizer, epochs=5000, 
                        pde_weight=1.0, bc_weight=1.0, ic_weight=1.0,
                        verbose=True, print_every=10)
            
            # 8. plot the history
            torch.save(model, f"{save_path}/check_pnt.pth")
            trainer.plot_training_history()
        case 'eval':
            model = torch.load( f"{save_path}/check_pnt.pth", weights_only=False)
            model.eval()
            with torch.no_grad():
                # create test grid
                q0_2x_preset = 1.0; gamma2_preset = 1.0; T_preset = 2. * np.pi * num_T / ((gamma2_preset+1.0)/2./np.pi/q0_2x_preset**2) * 1.20
                t_test = np.linspace(0,  T_preset, resolution).reshape(-1,1)
                q0_2x_test = np.full_like(t_test,q0_2x_preset)
                gamma2_test = np.full_like(t_test,gamma2_preset)
                
                # perdict
                test_points = np.concatenate([t_test, q0_2x_test, gamma2_test], axis=-1)
                test_tensor = torch.FloatTensor(test_points).to(device)
                u_pred = model(test_tensor).cpu().numpy()
                
                # plot
                plt.figure(figsize=(10, 6))
                plt.plot(t_test, u_pred[:,0],label='x1')
                plt.plot(t_test, u_pred[:,1],label='y1')
                plt.plot(t_test, u_pred[:,2],label='x2')
                plt.plot(t_test, u_pred[:,3],label='y2')

                plt.xlabel('t')
                plt.title(f'q20{q0_2x_preset:.2f}_gamma2{gamma2_preset:.2f}')
                plt.show()
            np.savetxt(f'{save_path}/q20{q0_2x_preset:.1f}_gamma2{gamma2_preset:.1f}_T{T_preset:.1f}.dat', np.concatenate([test_points, u_pred],axis=-1), 
                       fmt='%.6e', header='t    q0_2x    gamma2    x1    y1    x2    y2',  
                       delimiter='\t')

        case _:
            raise ValueError(f'mode {mode} not regonized.')