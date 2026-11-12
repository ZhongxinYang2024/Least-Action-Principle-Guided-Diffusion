import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import sys
import torch.optim as optim

##
from base_pinn import PINNDataset, PhysicsInformedNN, pinn_collate_fn, PINNTrainer
from base_pinn import batched_jacobian,gradients

sys.path.append(f'{os.getcwd()}')
from data.CustomDataLoader import CustomDataset_Foil

def eval(ckt_path):
    from HamiltonSys.ZhukovskyTransfomation import GetAFoil
    
    model = torch.load( ckt_path, weights_only=False)
    model.eval()
    with torch.no_grad():
        # create test grid
        aoa_preset = 0.0; creal_preset = -0.17241; cimag_preset = 0.0; radius_preset = 1.2268; 

        _, Z_Mesh = GetAFoil(aoa_preset, creal_preset, cimag_preset, radius_preset)
        XX, YY = Z_Mesh.real, Z_Mesh.imag

        XX -=  (creal_preset - radius_preset) + 1./(creal_preset - radius_preset)
        XX /= (2. * radius_preset - 2. * radius_preset/ (creal_preset**2 - radius_preset**2 )   )
        YY /= (2. * radius_preset - 2. * radius_preset/ (creal_preset**2 - radius_preset**2 )   )
        
        x_test, y_test = XX.reshape(-1,1), YY.reshape(-1,1)
        aoa_test = np.full_like(x_test, aoa_preset)
        creal_test = np.full_like(x_test, creal_preset)
        cimag_test = np.full_like(x_test, cimag_preset)
        radius_test = np.full_like(x_test,radius_preset)
        
        # perdict
        test_points = np.concatenate([x_test, y_test, aoa_test, creal_test, cimag_test, radius_test], axis=-1)
        test_tensor = torch.FloatTensor(test_points).to(device)
        U_pred = model(test_tensor).cpu().numpy()
        UU, VV, PP = U_pred[:,0].reshape(XX.shape),  U_pred[:,2].reshape(XX.shape), U_pred[:,2].reshape(XX.shape)
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

    # np.savetxt(f'{save_path}/q0{q0_preset:.1f}_p0{p0_preset:.1f}_kk{kk_preset:.1f}_mm{mm_preset:.1f}_T{T_preset:.1f}.dat', np.concatenate([test_points, u_pred],axis=-1), 
    #            fmt='%.6e', header='t    q0    p0    kk    mm    q    p',  
    #            delimiter='\t')
        plt.show()
    from HamiltonSys.Mesh import DrawNonuniformGrid
    # DrawNonuniformGrid(Z_Mesh.real, Z_Mesh.imag)

#%%
if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    dtype=torch.float32
    torch.set_default_dtype(dtype)
    
    save_path = './PINN/log/foil'
    os.makedirs(save_path, exist_ok=True)
    mode = 'eval'
    

    # 1. define PDE residual function
    def residual_fun(x, u):
        '''
        x[x, y, aoa, creal, cimag, radius]
        u[u, v, p]
        Arg: 
        x.shape :(num, ch_in)
        u.shape :(num, ch_out)
        batch_jacobian(u,x).shape :(num, ch_out, ch_in)
        '''
        
        dudx = batched_jacobian(u,x)[:, 0, 0].unsqueeze(-1)
        dudy = batched_jacobian(u,x)[:, 0, 1].unsqueeze(-1)
        dvdx = batched_jacobian(u,x)[:, 1, 0].unsqueeze(-1)
        dvdy = batched_jacobian(u,x)[:, 1, 1].unsqueeze(-1)
        dpdx = batched_jacobian(u,x)[:, 2, 0].unsqueeze(-1) * (1./ 1.4 / (0.1)**2)
        dpdy = batched_jacobian(u,x)[:, 2, 1].unsqueeze(-1) * (1./ 1.4 / (0.1)**2)
                
        residual1 = torch.abs(dudx + dvdy)  # dudx + dvdy = 0
        residual2 = torch.abs(u[:,0:1] * dudx + u[:,1:2] * dudy + dpdx) # ududx + vdudy = - dpdx
        residual3 = torch.abs(u[:,0:1] * dvdx + u[:,1:2] * dvdy + dpdy)  # udvdx + vdvdy = - dpdy
        residual =  residual1 + residual2 + residual3   
        residual *= 25.0/torch.sqrt( (x[:,0:1] - 0.5)**2 + (x[:, 1:2])**2 + 1e-4) 
        return residual
    
    def bc_loss(model, x_far, u_bc_pred, u_bc_truth):
        '''
        x_far: [x, y ,aoa, creal, cimag, radius]
        '''
        loss_far = torch.abs( (u_bc_pred - u_bc_truth)/(u_bc_truth+ 1e-10) )
        
        n_pnts = len(x_far)
        
        center = torch.complex(x_far[:,3:4], x_far[:,4:5])
        
        phi = torch.rand((n_pnts,1), device=x_far.device) * 2. * torch.pi 
        
        zetab_inCircle = center + x_far[:,5:6] * torch.complex(torch.cos(phi), torch.sin(phi))
        zeta_inCircle = zetab_inCircle
        ### for test only
        # center = torch.ones((n_pnts,1), device=x_far.device) *-0.5
        # zeta_inCircle = center  + 2.0 * torch.complex(torch.cos(phi), torch.sin(phi))
        
        z_inCircle = zeta_inCircle + 1.**2 / zeta_inCircle
        
        xx_foil = (z_inCircle.real) - (x_far[:,3:4] - x_far[:,5:6]) - 1./(x_far[:,3:4] - x_far[:,5:6])
        xx_foil /= (2. * x_far[:,5:6] - 2. * x_far[:,5:6]/ (x_far[:,3:4]**2 - x_far[:,5:6]**2 )   )
        yy_foil = (z_inCircle.imag)/(2. * x_far[:,5:6] - 2. * x_far[:,5:6]/ (x_far[:,3:4]**2 - x_far[:,5:6]**2 )   )
        x_foil = torch.cat([xx_foil, yy_foil, x_far[:,2:6] ], axis=(-1) )       
        
        
        
        ####
        u_foil = model(x_foil)
        n_zeta = zeta_inCircle - center
        dzdzeta = 1.0 - 1.0**2/zeta_inCircle**2
        n_z = dzdzeta * n_zeta
        nx = n_z.real; ny = n_z.imag
        nx = nx/ torch.sqrt(nx**2 + ny**2 + 1e-10); ny = ny/ torch.sqrt(nx**2 + ny**2 + 1e-10)
        loss_wall = torch.abs( (u_foil[:,0:1] * nx + u_foil[:,1:2] * ny)/torch.sqrt(u_foil[:, 0:1]**2 +  u_foil[:, 1:2]**2 + 1e-10 )  )
        
        # dudx = batched_jacobian(u_foil,x_foil)[:, 0, 0].unsqueeze(-1)
        # dudy = batched_jacobian(u_foil,x_foil)[:, 0, 1].unsqueeze(-1)
        # dvdx = batched_jacobian(u_foil,x_foil)[:, 1, 0].unsqueeze(-1)
        # dvdy = batched_jacobian(u_foil,x_foil)[:, 1, 1].unsqueeze(-1)
        
        return loss_far + loss_wall
    
    # 2. generate training data
    aoa_begin = 0.0; aoa_end = 10.0
    creal_begin = -1.0; creal_end = 0.0
    cimag_begin = 0.0; cimag_end = 0.0
    radius_begin = 1.0 + 2e-2; radius_end = 2.0 + 2e-2
    ch_in = 6; ch_out = 3
    nr = 96; nt = 128   
    
    if mode == 'train':
        # PDE data
        
        dataset = CustomDataset_Foil(nr, nt)
        dataset.IfScaleFoil2One()
        x_pde_dis = dataset.labels_distributed
        x_pde_con = dataset.labels.unsqueeze(-1).unsqueeze(-1).repeat(1,1, nr, nt)
        x_pde = torch.cat([x_pde_dis, x_pde_con],axis=(1)).permute(0,2,3,1).reshape(-1,ch_in)
        u_pde = dataset.flowdata.permute(0,2,3,1).reshape(-1, ch_out)
        
        
        # interior data
        x_domain = x_pde

        # B.C. data 
        R_far = 25.0
        n_bc = 128 * 32 * 32 * 11
        rr = np.ones((n_bc,1)) * R_far
        tt = np.random.uniform(0.0, 2. * np.pi, (n_bc,1))
        xx_bc = rr * np.cos(tt)
        yy_bc = rr * np.sin(tt)
        aoa_bc = np.random.uniform(aoa_begin, aoa_end, (n_bc,1))
        creal_bc = np.random.uniform(creal_begin, creal_end, (n_bc,1))
        cimag_bc = np.random.uniform(cimag_begin, cimag_end, (n_bc,1))    
        radius_bc = np.random.uniform(radius_begin, radius_end, (n_bc,1))
        radius_bc[radius_bc < np.sqrt(creal_bc**2 + cimag_bc**2) +1.0] = (np.sqrt(creal_bc**2 + cimag_bc**2) +1.0)[radius_bc < np.sqrt(creal_bc**2 + cimag_bc**2) +1.0] 
        bc_pos = np.concatenate([xx_bc, yy_bc, aoa_bc, creal_bc, cimag_bc, radius_bc],axis=-1)
        u_bc = 1.0 * np.cos(aoa_bc * np.pi / 180.)
        v_bc = 1.0 * np.sin(aoa_bc * np.pi / 180.)
        p_bc = np.full_like(aoa_bc, 1.0) 
        bc_value = np.concatenate([u_bc, v_bc, p_bc], axis=-1)

        # I.C. data 
        pass 
        
        
        # 3. create dataset
        dataset = PINNDataset(domain_pos=x_domain, 
                            bc_pos=bc_pos, bc_value=bc_value,
                            pde_pos=x_pde, pde_value=u_pde)
        dataloader = DataLoader(dataset, batch_size= nr * nt * 3, shuffle=True, collate_fn=pinn_collate_fn)
    
    # 4. create the model
    layers = [ch_in, 128, 128, 128, 128, 128, 128, 128, 128, ch_out]  #
    model = PhysicsInformedNN(layers)

    match mode.lower():
        case 'train':
            #model = torch.load( f"{save_path}/check_pnt_120.pth", weights_only=False)
            # 5. create traner
            trainer = PINNTrainer(model, 
                                  pde_func=residual_fun,
                                  bc_func=bc_loss, 
                                  device=device)
            
            # 6. set optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.7,
                patience=10,
                min_lr = 1e-10
            )
                        
            # optimizer = torch.optim.LBFGS(model.parameters(), 
            #                     lr=2e-1, 
            #                     max_iter=20, 
            #                     history_size=50, 
            #                     line_search_fn='strong_wolfe')
            # scheduler = None


            # 7. train the model
            trainer.train(dataloader, 
                        optimizer, scheduler, epochs=10000, 
                        pde_weight=1e-4, bc_weight=1e-3, ic_weight=0.0,
                        verbose=True, print_every=1, save_path = save_path)
            
            # 8. plot the history
            torch.save(model, f"{save_path}/check_pnt.pth")
            trainer.plot_training_history()
        case 'eval':
            eval(f"{save_path}/check_pnt_37000_adam.pth")
        case _:
            raise ValueError(f'mode {mode} not regonized.')