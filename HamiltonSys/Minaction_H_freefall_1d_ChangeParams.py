#%% 
import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
import os
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


##
from . import Minaction_Hamilton_torch

PhySystem = Minaction_Hamilton_torch.PhySystem
Plot_phase3D =  Minaction_Hamilton_torch.Plot_phase3D


#%% user def function
def plot(item, fig, ax1, ax2, ax3):
    q_target = -0.5 * item.g * item.tt**2;
    p_target = -item.g * item.tt * item.mass[0]
    ax1.plot(item.tt, item.pos_best[0,0,:],label='Pos', color='blue', linestyle='-',marker='s')
    ax1.plot(item.tt, q_target, color='red', linestyle='--')
    ax1.set_xlabel('t')
    ax1.set_ylabel('q')
    

    ax2.plot(item.tt, item.mom_best[0,0,:],label='Mom', color='blue', linestyle='-',marker='^')
    ax2.plot(item.tt, p_target, color='red', linestyle='--')
    ax2.set_xlabel('t')
    ax2.set_ylabel('p')
    
    x = item.pos[0,0,:]; y = item.mom[0,0,:]; z = item.tt
    Plot_phase3D(x, y, z, ['q', 'p', 't'], ax3, fig)
    Plot_phase3D(q_target, p_target, z, ['q', 'p', 't'], ax3, fig, color='red')
    
class PhySystem_FreeFall(PhySystem):
    # only one mass and one spring
    mass = 0.0
    g = 0.0
    def __init__(self, init_position, init_momentum, t_end, n_split ,mass, g, IF_dissipative = False, device='cpu'):
        super().__init__(init_position, init_momentum, t_end, n_split, IF_dissipative, device)
        
        #  FreeFall specific
        self.mass = np.array(mass)
        self.g = np.array(g)
        
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

        mass_tensor = torch.as_tensor(self.mass, dtype=p.dtype, device=self.device).view([len(self.mass)] + [1]*(ndim - 2))
        
        kinetic = torch.sum(p_permuted**2, dim=1) / (2 * mass_tensor)
        H = torch.sum(kinetic, dim=0) # sum over objects, shape: [b, n_split] or [n_split] or scalar
        
        # compute distance
        mask = torch.zeros(new_shape, device=self.device); 
        mask[:, 0, ...] = 1.0
        h  = torch.sum(q_permuted * mask, dim=1)
            
        potential =  h * mass_tensor * torch.tensor(self.g, device=self.device) 
        H += torch.sum(potential,dim=0)
            

        return H #shape: [b, n_split] or [n_split] or scalar
    
def FreeFall_reparam(t_end, 
                    n_split,
                     g,
                    known_pos,
                    known_mom,
                    init_position,
                    init_momentum,
                    pos_end_cond,
                    mom_end_cond,
                    Hconfig):

        FreeFall_another = PhySystem_FreeFall(init_position=init_position,
                                    init_momentum=init_momentum,
                                    t_end = t_end,
                                    n_split = n_split,
                                    mass= [1.0],
                                    g =  g,
                                    device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # transfer to new system
        FreeFall_another.pos = copy.deepcopy(known_pos)
        FreeFall_another.mom = copy.deepcopy(known_mom)   
        
        if known_pos[:,:,0] != init_position:
            FreeFall_another.pos[:,:,0] = init_position
        if known_mom[:,:,0] != init_momentum:
            FreeFall_another.mom[:,:,0] = init_momentum

        pos_end_cond = np.array(pos_end_cond)
        mom_end_cond = np.array(mom_end_cond)
         
        eps = 1e-12   
        curves =  FreeFall_another.Optimize_NewParams( 
                pos_end_cond= pos_end_cond, 
                mom_end_cond= mom_end_cond,
                eps_q = eps,
                eps_p = eps,                                     
                lambda_q = 1.0,
                lambda_p = 1.0,
                num_samples = (n_split+1) // 2 * 2 // 3 ,  # should  slightly larger than  n_split // 2 * 2 // 3  the first 2 is maximun frequency, the second 2 is Nyquist thoerm, the third 3 is samples times 
                IF_progressive = False, # add noise to different freedom seperately.
                reduce_op_type= 'mean',
                lr = Hconfig['lr'], # ; case2:8e-2; case3:2e-1;
                lr_scheduler='Onplateau',
                lr_updatet = Hconfig['lr_updatet'], # #case2:500; #case3:500; 
                lr_decay = Hconfig['lr_decay'],
                num_iterations= Hconfig['num_iterations'], # case2:10000; case3:20000;
                dt_expect= t_end /  n_split,
                IF_uniform_dt=True,
                t_min = 0.00,
                t_end_given=None,
                period_updatet = Hconfig['period_updatet'], # case2:50； case3:50;
                decay_updatet = Hconfig['decay_updatet'], # case2-3: 0.999; case4:0.995
                lambda_smooth_q_1st=Hconfig['lambda_smooth_q_1st'], 
                lambda_smooth_p_1st=Hconfig['lambda_smooth_p_1st'] , # case2:5e-2; case3:[5e-2, 1e-2, 5e-3]; 
                perturbation_type= Hconfig['perturbation_type'], #perturbation ; sincos; sincos; 
                variation_method='indirect_substract', #'direct_integral', # or 'direct_integral'
                optimizers=['Adam']) # 
        
        return FreeFall_another, curves
#%%
if __name__ == "__main__":
    # test params
    mass = [1.0]
    g = 20.0
    init_position = np.array([[0.0]])
    init_momentum = np.array([[0.0]])
    t_end = 2.0
    n_split = 127

    FreeFall = PhySystem_FreeFall(init_position=init_position,
                                      init_momentum=init_momentum,
                                      t_end = t_end,
                                      n_split = n_split,
                                      mass=mass,
                                      g = g)    
    
    eps = 1e-12
    IFsolve = False
    if IFsolve:
         FreeFall.Solver(
                eps_q = eps, 
                eps_p = eps, 
                n_integral = 20, 
                num_samples = 5, 
                lr = 2e-1, 
                num_iterations=100,
                perturbation_type='white_noise')
    else:
        FreeFall.pos[0,0,:] =  -0.5 * FreeFall.g * FreeFall.tt**2 + np.random.randn(n_split+1)
        FreeFall.pos[0,0,0] = 0.0
        FreeFall.mom[0,0,:]  = -FreeFall.g * FreeFall.tt * FreeFall.mass[0] + np.random.randn(n_split+1)
        FreeFall.mom[0,0,0] = 0.0

    IFreparams = True
    if IFreparams:
        FreeFall_another, curves = FreeFall_reparam(t_end,
                                                    n_split,
                                                     g,
                                                    FreeFall.pos,
                                                    FreeFall.mom,
                                                    init_position,
                                                    init_momentum,
                                                    pos_end_cond=[[- 0.5 * g * t_end**2]],
                                                    mom_end_cond=[[ np.inf   ]])


    fig = plt.figure()
    ax1 = fig.add_subplot(2,3,1)
    ax2 = fig.add_subplot(2,3,2)
    ax3 = fig.add_subplot(2,3,3, projection='3d')
    ax4 = fig.add_subplot(2,3,4)
    ax5 = fig.add_subplot(2,3,5)
    ax6 = fig.add_subplot(2,3,6, projection='3d')
    
    plot( FreeFall, fig, ax1, ax2, ax3)
    
    if IFreparams:
        plot( FreeFall_another, fig, ax4, ax5, ax6)
        
        fig2 = plt.figure()
        ax2_1 = fig2.add_subplot(2,3,1)
        ax2_1.semilogy(np.linspace(0,1, len( FreeFall_another.loss_reparams)), FreeFall_another.loss_reparams)
    
    plt.tight_layout()
    plt.legend()
    plt.show()
    
    from tools.Animation import animate_curves
    animate_curves(curves, interval=0.02, title='demonstration')
    



# %%
