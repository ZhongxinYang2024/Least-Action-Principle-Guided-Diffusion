#%% 
import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


from . import Minaction_Hamilton_torch
# import Minaction_Hamilton_torch

##
PhySystem = Minaction_Hamilton_torch.PhySystem
Plot_phase3D =  Minaction_Hamilton_torch.Plot_phase3D

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

#%% user def function
def plot(item, fig, ax1, ax2, ax3, IF_answer=False, IFplot_best=False):
    pos_x, pos_y = reverse_scalar(item.pos, item.mom, item.mass)
    ax1.plot(item.tt, pos_x[0,0,:],label='x1', color='blue', linestyle='-',marker='s')
    ax1.plot(item.tt, pos_x[1,0,:],label='x2', color='green', linestyle='-',marker='s')
    ax1.legend()    
    ax1.set_xlabel('t')
    ax1.set_ylabel('x')
    

    ax2.plot(item.tt, pos_y[0,0,:],label='y1', color='blue', linestyle='-',marker='^')
    ax2.plot(item.tt, pos_y[1,0,:],label='y2', color='green', linestyle='-',marker='^')
    ax2.legend()
    ax2.set_xlabel('t')
    ax2.set_ylabel('y')
    
    
    z = item.tt
    Plot_phase3D(pos_x[0,0,:], pos_y[0,0,:], z, ['x', 'y', 't'], ax3, fig, color='blue')
    Plot_phase3D(pos_x[1,0,:], pos_y[1,0,:], z, ['x', 'y', 't'], ax3, fig, color='green')

    if IFplot_best:
        pos_x_bst, pos_y_bst = reverse_scalar(item.pos_best, item.mom_best, item.mass)
        ax1.plot(item.tt, pos_x_bst[0,0,:], linestyle=':',color='green')
        ax1.plot(item.tt, pos_x_bst[1,0,:], linestyle=':',color='green')
        ax2.plot(item.tt, pos_y_bst[0,0,:], linestyle=':',color='green')
        ax2.plot(item.tt, pos_y_bst[1,0,:], linestyle=':',color='green')
        
    if IF_answer:
        xc = np.sum(item.mass * pos_x[:,0,0]) / np.sum(item.mass)
        Omega = np.sum(item.mass) / (2. * np.pi  * ( (pos_x[1,0,0] - pos_x[0,0,0])**2 + (pos_y[1,0,0] - pos_y[0,0,0])**2 ) ) 
        


        x1_target = xc * (1. - np.cos(Omega * item.tt))
        y1_target = -xc * np.sin(Omega * item.tt)
        x2_target = xc + (pos_x[1,0,0] - xc) * np.cos(Omega * item.tt)
        y2_target = (pos_x[1,0,0] - xc) * np.sin(Omega * item.tt)
        ax1.plot(item.tt, x1_target, color='red', linestyle='--')
        ax1.plot(item.tt, x2_target, color='red', linestyle='--')
        ax2.plot(item.tt, y1_target, color='red', linestyle='--')
        ax2.plot(item.tt, y2_target, color='red', linestyle='--')
    
        Plot_phase3D(x1_target, y1_target, z, ['x', 'y', 't'], ax3, fig, color='red')
        Plot_phase3D(x2_target, y2_target, z, ['x', 'y', 't'], ax3, fig, color='red')  


def scalar(x, y, mass): # x.shape: [num_obj, 1, ...]
    mass_matrix =  mass.reshape(-1, 1, *([1] * (x.ndim - 2)))
    q =  x * np.sqrt( np.abs( mass_matrix ))
    p =  y * np.sqrt( np.abs( mass_matrix )) * np.sign(mass_matrix)
    return q,p
    
    
def reverse_scalar(q, p, mass):
    mass_matrix =  mass.reshape(-1, 1, *([1] * (q.ndim - 2)))
    x =  q / np.sqrt( np.abs( mass_matrix ))
    y =  p / np.sqrt( np.abs( mass_matrix )) / np.sign(mass_matrix)
    return x,y


def scalar_torch(x, y, mass):
    # x.shape: [num_obj, 1, ...]
    mass_matrix = mass.view(-1, 1, *([1] * (x.dim() - 2)))
    abs_mass = torch.abs(mass_matrix)
    sqrt_abs_mass = torch.sqrt(abs_mass)
    sign_mass = torch.sign(mass_matrix)
    
    q = x * sqrt_abs_mass
    p = y * sqrt_abs_mass * sign_mass
    return q, p

def reverse_scalar_torch(q, p, mass):
    mass_matrix = mass.view(-1, 1, *([1] * (q.dim() - 2)))
    abs_mass = torch.abs(mass_matrix)
    sqrt_abs_mass = torch.sqrt(abs_mass)
    sign_mass = torch.sign(mass_matrix)
    
    x = q / sqrt_abs_mass
    y = p / sqrt_abs_mass / sign_mass
    return x, y

class PhySystem_PointVortex_2bodies_2d(PhySystem):
    # only one mass and one spring
    mass = 0.0
    def __init__(self, init_position, init_momentum, t_end, n_split , mass,IF_dissipative=False, device='cpu'):
        super().__init__(init_position, init_momentum, t_end, n_split, IF_dissipative, device)
        
        #  Vortex
        self.mass = np.array(mass)
        
        # 检查质量参数维度
        if self.mass.size != self.num_of_obj:
            raise ValueError("Mass array size must match number of objects")

    def HamiltonFnc_np(self, q, p, t_now=0.0):
        H = np.sum( np.sum(p**2, axis=1)/(2 * self.mass))
        for i_obj in range(self.num_of_obj):
            for j_obj in range(i_obj+1, self.num_of_obj):
                rr = 0.0
                for i_dim in range(self.dimensionality):
                    rr += np.square(q[i_obj, i_dim] - q[j_obj,i_dim])
                rr = np.sqrt(rr)
                H += self.G * self.mass[i_obj] * self.mass[j_obj] /rr 
        return H
    
    def HamiltonFnc_torch(self, q, p, t_now=0.0):
        '''
        Arg: q: [batch_size, num_obj, num_dim, n_split] or [num_obj, num_dim, n_split] or [num_obj, num_dim]

        '''
        ndim = q.dim()
        
        if ndim == 4:
            q_permuted = q.permute(1, 2, 0, 3) # new shape: [num_obj, num_dim, batch_size, n_split]
            p_permuted = p.permute(1, 2, 0, 3) # new shape: [num_obj, num_dim, batch_size, n_split]
        else:
            q_permuted = q  # 对于 ndim=3 或 2，无需置换
            p_permuted = p
        new_shape = q_permuted.shape

        mass_tensor = torch.as_tensor(self.mass, dtype=p.dtype, device=self.device).view([len(self.mass)] + [1])
        
        x_permuted, y_permuted = reverse_scalar_torch(q_permuted, p_permuted, mass_tensor )
        Z_permuted = torch.cat([x_permuted, y_permuted], dim=(1)) 

        # compute all distance seperation
        # q_i: [num_of_obj, 1, dims, ...], q_j: [1, num_of_obj, dims, ...]
        Z_i = Z_permuted.unsqueeze(1)  # 形状 [n, 1, dims, ...]
        Z_j = Z_permuted.unsqueeze(0)  # 形状 [1, n, dims, ...]
        Dq = Z_i - Z_j        # 形状 [n, n, dims]
    
        # compute distance
        distances = torch.sqrt(torch.sum(Dq**2, dim=2) + 1e-20)  # 形状 [n, n, ...]
    

        # compute potential term (ignore i=j)
        # use mask (diag ignored)
        mask = torch.ones(self.num_of_obj, self.num_of_obj, 
                    dtype=torch.float64, device=self.device)
        mask = mask - torch.eye(self.num_of_obj, 
                                dtype=torch.float64, device=self.device)
    
        mass_weight = (mass_tensor * mass_tensor.T).view([len(self.mass), len(self.mass)] + [1]*(q.dim() - 2 ))
        potential =  -(1./4./torch.pi) * torch.log(distances) * mass_weight
        
        H = torch.einsum('ij...,ij->...',[potential, mask])

        return H #shape: [n_split, b]

def Vortex_reparam(t_end,n_split,
                   mass,
                   known_pos, # obj, dim, time
                   known_mom,
                   init_position, # obj, dim
                   init_momentum,
                   pos_end_cond, # obj, dim
                   mom_end_cond,
                   Hconfig):
        Vor_another = PhySystem_PointVortex_2bodies_2d(init_position=init_position,
                                        init_momentum=init_momentum,
                                        t_end = t_end,
                                        n_split = n_split,
                                        mass= mass,
                                        device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # transfer to new system
        Vor_another.pos = copy.deepcopy(known_pos)
        Vor_another.mom = copy.deepcopy(known_mom)   
        

        Vor_another.pos[:,:,0] = init_position
        Vor_another.mom[:,:,0] = init_momentum
        
        pos_end_cond = np.array(pos_end_cond)
        mom_end_cond = np.array(mom_end_cond)   
        
        eps = 1e-12
        curves =  Vor_another.Optimize_NewParams( 
                pos_end_cond= pos_end_cond, 
                mom_end_cond= mom_end_cond,
                eps_q = eps,
                eps_p = eps,   
                lambda_q = 1.0,
                lambda_p = 1.0,
                num_samples = (n_split + 1) // 2 * 2 // 3 ,  # should  slightly larger than  n_split // 2 * 2 // 3  the first 2 is maximun frequency, the second 2 is Nyquist thoerm, the third 3 is samples times 
                IF_progressive= False,
                reduce_op_type='mean',
                lr = Hconfig['lr'], #case2:4e-1; case3: 4e-1 case4: 5e-1
                lr_scheduler= 'Onplateau',
                lr_updatet=  Hconfig['lr_updatet'],  #case2:1000; case3:1000; case4:1000
                lr_decay=    Hconfig['lr_decay'], #case2: 0.7
                num_iterations= Hconfig['num_iterations'], #case2:50000; case3:10000; case4:180000
                dt_expect= t_end /  n_split, # Minaction_Hamilton_torch line 245 need changes, and the total time
                t_min = 0.0, 
                t_end_given =  None,
                IF_uniform_dt=  True,
                period_updatet = Hconfig['period_updatet'],  #case2:25; case3:25; case4:25;
                decay_updatet= Hconfig['decay_updatet'], #case2:0.99; case3:0.995; case4:0.999;
                lambda_smooth_q_1st= Hconfig['lambda_smooth_q_1st'], 
                lambda_smooth_p_1st= Hconfig['lambda_smooth_p_1st'], #case2:[1e-2, 5e-3, 1e-3, 5e-4]; case3:[1e-2, 5e-3, 1e-3, 5e-4]; case4:[1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 5e-3, 5e-3, 5e-3, 2.5e-3, 2e-3, 2e-3, 2e-3, 2e-3, 2e-3, 2e-3, 5e-4, 5e-4, 5e-4]
                perturbation_type= Hconfig['perturbation_type'], # 'sincos', 'sinsin', 'white_noise', 'sincosp1' #case2:'sincos'; case3:'sincos'; case4:'subcosp1'
                variation_method='indirect_substract', 
                optimizers=['SGDM'])
        
        return Vor_another, curves
#%%
if __name__ == "__main__":
    # test params
    num_T = 1.
    preprocess_q0_2x = 1.0;  target_q0_2x = 1.0;
    preprocess_gamma2 = 2.0; target_gamma2 = 4.0;
    
    xc = (0.0 * 1.0 + target_gamma2 * target_q0_2x)/ (1.0 + target_gamma2 + 1e-20)
    target_omega = np.abs(target_gamma2 + 1.0) / 2. / np.pi / target_q0_2x**2
    
    preprocess_T = 2.0 * np.pi * num_T / ( np.abs(preprocess_gamma2 + 1.0) / 2. / np.pi / preprocess_q0_2x**2 )
    target_T = 2.0 * np.pi * num_T / target_omega
        
    mass = np.array([1.0, preprocess_gamma2])
    init_x = np.array([[0.0],  [preprocess_q0_2x]])
    init_y = np.array([[0.0],  [0.0]])
    init_position, init_momentum = scalar(init_x, init_y, mass)
    t_end = preprocess_T
    n_split = 127

    Vor = PhySystem_PointVortex_2bodies_2d(init_position=init_position,
                                      init_momentum=init_momentum,
                                      t_end = t_end,
                                      n_split = n_split,
                                      mass=mass) # vortex intensity instead    
    

    eps = 1e-12
    IFsolve = False
    if IFsolve:
         Vor.Solver(
                eps_q = eps, 
                eps_p = eps, 
                n_integral = 10, 
                num_samples = 5, 
                lr = 5e-4, 
                num_iterations=200,
                perturbation_type='sin',
                IF_guess=False)
    else:
        item = Vor
        xc = np.sum(item.mass * init_x[:,0]) / np.sum(item.mass)
        yc = np.sum(item.mass * init_y[:,0]) / np.sum(item.mass)
        R = np.sqrt( (init_x[1,0] - init_x[0,0])**2 + (init_y[1,0] - init_y[0,])**2 )
        r1 = item.mass[1] / np.sum(item.mass) * R; r2 = item.mass[0] / np.sum(item.mass) * R;  
        V1 = item.mass[1] / (2.0 * np.pi * R); V2 = item.mass[0] / (2.0 * np.pi * R)
        Omega = np.sum(item.mass) / (2. * np.pi * R**2)
        x1_target = xc * (1. - np.cos(Omega * item.tt))
        y1_target = -xc * np.sin(Omega * item.tt)
        x2_target = xc + (init_x[1,0] - xc) * np.cos(Omega * item.tt)
        y2_target = (init_x[1,0] - xc) * np.sin(Omega * item.tt)
        Vor.pos, Vor.mom  =  scalar(np.concatenate((x1_target.reshape(1,1,-1), x2_target.reshape(1,1,-1)), axis=(0)),
                                                  np.concatenate((y1_target.reshape(1,1,-1), y2_target.reshape(1,1,-1)), axis=(0)),
                                                  mass)
        

    IFreparams = True
    if IFreparams:
        from Hconfigs.Q4_C3_config import get_config
        Hconfig = get_config()
        
                    
        # f = np.loadtxt('./log/vortexsys/q201.0_gamma21.0_T23.7.dat')
        # known_pos = np.concatenate((f[:,3].reshape(1,1,-1), f[:,5].reshape(1,1,-1)), axis=0)
        # known_mom = np.concatenate((f[:,4].reshape(1,1,-1), f[:,6].reshape(1,1,-1)), axis=0)
        known_pos = Vor.pos
        known_mom = Vor.mom
        xc = np.sum(mass * init_x[:,0]) / np.sum(mass)
        print(preprocess_T)
        Vortex_another, curves = Vortex_reparam(preprocess_T,   # just a guess
                                            127,
                                            [1.0,  target_gamma2],
                                            known_pos, # obj, dim, time
                                            known_mom,
                                            np.array([[0.0], [target_q0_2x * np.sqrt(np.abs(target_gamma2))   ]]),
                                            np.array([[0.0], [0.0]]),
                                            np.array([[0.0], [target_q0_2x * np.sqrt(np.abs(target_gamma2))]]),
                                            np.array([[np.inf], [np.inf]]),
                                            Hconfig)     

            

            
#%% 
    fig = plt.figure()
    ax1 = fig.add_subplot(2,3,1)
    ax2 = fig.add_subplot(2,3,2)
    ax3 = fig.add_subplot(2,3,3, projection='3d')
    ax4 = fig.add_subplot(2,3,4)
    ax5 = fig.add_subplot(2,3,5)
    ax6 = fig.add_subplot(2,3,6, projection='3d')
    
    plot( Vor, fig, ax1, ax2, ax3, IF_answer=True)
    
    if IFreparams:
        plot( Vortex_another, fig, ax4, ax5, ax6, IF_answer=True, IFplot_best=True)
        
        fig2 = plt.figure()
        ax2_1 = fig2.add_subplot(2,3,1)
        ax2_1.semilogy(np.linspace(0,1, len( Vortex_another.loss_reparams)), Vortex_another.loss_reparams)
    
    plt.legend()
    plt.show()
    
    #from tools.Animation import animate_curves
    #animate_curves(all_curves, interval=0.005, title='demonstration')
    



# %%
