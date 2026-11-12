import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

##
from HamiltonSys.ZhukovskyTransfomation import Joukowski_airfoil, GetAFoil
# from ZhukovskyTransfomation import Joukowski_airfoil, GetAFoil

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def Cal_Ga(xx, yy, uu, vv):
    """
    向量化版本的环量计算函数，效率更高
    """
    
    xx = np.asarray(xx)
    yy = np.asarray(yy)
    uu = np.asarray(uu)
    vv = np.asarray(vv)
    
    # 计算所有线段的dx, dy
    dx = np.diff(xx)
    dy = np.diff(yy)
    
    # 计算线段中点速度
    u_mid = 0.5 * (uu[:-1] + uu[1:])
    v_mid = 0.5 * (vv[:-1] + vv[1:])
    
    # 计算环量
    circulation = - np.sum(u_mid * dx + v_mid * dy)
    
    return circulation


def Foil_reparam(aoa, creal, cimag, radius, Hconfig,
                 XX=None, YY=None, UU=None, VV=None, PP=None):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    ## def model
    center = torch.tensor(creal, dtype=torch.complex64, device=device) + torch.tensor(cimag, dtype=torch.complex64, device=device) * 1j
    try:
        initial_gamma = Cal_Ga(XX[0,:], YY[0,:], UU[0,:], VV[0,:])
        print(f'inital_gamma = {initial_gamma:.6f}')
    except (TypeError, ValueError):
        initial_gamma = 0.0
    
    
    # inital gamma as parameter needed to optimize
    gamma = torch.nn.Parameter(torch.tensor(initial_gamma, requires_grad=True, device= device))
    optimizer = torch.optim.SGD([gamma], 
                                lr=Hconfig['lr'],
                                momentum=0.9,
                                weight_decay=0.0, 
                                nesterov=True)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min',       # 监控损失最小化
                factor=Hconfig['lr_decay'],       # 学习率乘以0.5
                patience=Hconfig['lr_updatet'],      
                min_lr=1e-10       # 最小学习率
            )
                    
    eps = 5e-4    
    gamma_tp = []
    S_tp = []
    gamma_bst = 0.0; loss_bst = 1e10
    for epoch in range(Hconfig['num_iterations']):
        optimizer.zero_grad()
    
        params0 = {
            'degree_of_foreaft_asymmetry':  0.0,  # tau=0 for ellipses
            'degree_of_smoothness_trailing':  0.0, # D=0 for sharp trailing edge, D=1 for identity map 
            'transform_param':              1.0,  # b
            'half_chord_length':         1.0,
            'center_cylinder':       center,
            'radius_cylinder': radius,
            'free_flow': 1.0,
            'angle_of_attack': aoa / 180 * torch.pi,
            'gamma': gamma,  # 使用传入的 gamma 参数
            'IF_modified': False,   
        } 
    
        foil0 = Joukowski_airfoil(params0)
        S = foil0.integral_QoI_in_zeta_torch([96 * 4, 128 * 2], R_max=100.0)
        
        names = locals()
        n_samples = 10
        
        Sbar = torch.zeros((n_samples,), device=device)
        for i in range(1, n_samples+1):
            names['params'+str(i)] = {
                'degree_of_foreaft_asymmetry':  0.0,  # tau=0 for ellipses
                'degree_of_smoothness_trailing':  0.0, # D=0 for sharp trailing edge, D=1 for identity map 
                'transform_param':              1.0,  # b
                'half_chord_length':         1.0,
                'center_cylinder':       center,
                'radius_cylinder': radius,
                'free_flow': 1.0,
                'angle_of_attack': aoa / 180 * torch.pi,
                'gamma': gamma + torch.randn((1,), device=device) * eps,  # 使用传入的 gamma 参数
                'IF_modified': False,   
            } 
    
            names['foil'+str(i)] = Joukowski_airfoil(names['params'+str(i)])
            Sbar[i-1] = names['foil'+str(i)].integral_QoI_in_zeta_torch([96 * 4, 128 * 2], R_max=100.0)
        
        
    
        loss = torch.mean( (Sbar - S)**2 )/ eps**2
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        gamma_tp.append(gamma.item())
        S_tp.append(S.item())
        
        if epoch % 1 == 0:
            print(f'Epoch {epoch}: gamma = {gamma.item():.9f}, S = {S.item():.9f}, Loss = {loss.item():.9f}, LR = {optimizer.param_groups[0]["lr"]:.8f}')
            if loss.item() < loss_bst:
                gamma_bst = gamma.item()
                loss_bst = loss.item()
    
    gamma_fine = gamma_bst
    center = creal + cimag * 1j; 
    
    params = {'degree_of_foreaft_asymmetry':  0.0, #  e.g. tau=0 for ellipses
            'degree_of_smoothness_trailing':  0.0,# e.g. D=0 for sharp trailing edge D=1 for identity map 
            'transform_param':              1.0, # b
            'half_chord_length':         1.0,
            'center_cylinder':       center,
            'radius_cylinder': radius,
            'free_flow': 1.0,
            'angle_of_attack': aoa / 180 * np.pi,
            'gamma': gamma_fine,
            'IF_modified': False,   
            } 

    
    foil_std = Joukowski_airfoil(params)
    foil_std.Get_Zetab_cylinder(num_points=200)
    foil_std.Joukowski_transform()
    
    _, Z_Mesh = GetAFoil(aoa, creal, cimag, radius)
    W = foil_std.Get_W_Inz(Z_Mesh)
    
    plt.scatter(gamma_tp, S_tp)
    plt.show()
    
    return Z_Mesh.real, Z_Mesh.imag, W.real, W.imag
    

    
    
if __name__ == '__main__':
    from Hconfigs.Q5_C2_config import get_config
    Hconfig = get_config()
        
    XX, YY, UU, VV =  Foil_reparam(20.0, -0.1724137921, 0.0, 1.2268965517, Hconfig=Hconfig)

    
    fig = plt.figure()
    ax4 = fig.add_subplot(3, 3, 4)
    contourf_plot = ax4.contourf(XX, YY, UU, levels=20)
    cbar = plt.colorbar(contourf_plot, ax=ax4)
    cbar.set_label('Velocity Magnitude')
    cl = ax4.contour(XX, YY, UU, levels=10, colors='black', alpha=0.3)
    ax4.clabel(cl, inline=True, fontsize=8, fmt='%.2f')
    plt.show()





