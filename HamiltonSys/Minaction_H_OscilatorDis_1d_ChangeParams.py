#%% 
import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

##
# import Minaction_Hamilton_torch
from . import Minaction_Hamilton_torch

##
PhySystem = Minaction_Hamilton_torch.PhySystem
Plot_phase3D =  Minaction_Hamilton_torch.Plot_phase3D

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

#%% user def function
def plot(item, fig, ax1, ax2, ax3, IFplot=True, IFplot_best=False):
    delta = (item.mu/item.mass[0])**2 - 4 * item.k / item.mass[0]
    if delta > 0:
        print('overdamping')
        r1 = (- item.mu/item.mass[0] + np.sqrt(delta)) / 2
        r2 = (- item.mu/item.mass[0] - np.sqrt(delta)) / 2
        A = (item.mom[0,0,0]/item.mass[0] - r2 * item.pos[0,0,0]) / (r1 - r2)
        B = item.pos[0,0,0] - A 
        q_target = A * np.exp(r1 * item.tt) + B * np.exp(r2 * item.tt)
        p_target = ( A * r1 * np.exp(r1 * item.tt) + B * r2 * np.exp(r2 * item.tt) ) * item.mass[0]
    elif delta == 0:
        print('critical damping')
        r = - item.mu/item.mass[0] / 2
        A = item.pos[0,0,0]
        B = item.mom[0,0,0]/item.mass[0] - r * A
        q_target = (A + B * item.tt) * np.exp(r * item.tt)
        p_target = ( B * np.exp(r * item.tt) + (A + B * item.tt) * r * np.exp(r * item.tt) ) * item.mass[0]
    else:
        print('underdamping')
        A = item.pos[0,0,0]
        B = (item.mom[0,0,0]/item.mass[0] + item.mu/(2*item.mass[0]) * item.pos[0,0,0]) / np.sqrt(item.k/item.mass[0] - (item.mu/item.mass[0])**2/4)   
        a = - item.mu/(2*item.mass[0])
        w = np.sqrt(item.k/item.mass[0] - (item.mu/item.mass[0])**2/4)
        q_target = np.exp(a * item.tt) * ( A * np.cos(w * item.tt) + B * np.sin(w * item.tt) )
        p_target = (  a * np.exp(a * item.tt) * ( A * np.cos(w * item.tt) + B * np.sin(w * item.tt) ) + np.exp(a * item.tt) * ( - A * w * np.sin(w * item.tt) + B * w * np.cos(w * item.tt) ) ) * item.mass[0]
    if not IFplot:
        return q_target, p_target

    ax1.plot(item.tt, item.pos[0,0,:],label='Pos', color='blue', linestyle='-',marker='s')
    ax1.plot(item.tt, q_target, color='red', linestyle='--')
    
    ax1.set_xlabel('t')
    ax1.set_ylabel('q')
    

    ax2.plot(item.tt, item.mom[0,0,:],label='Mom', color='blue', linestyle='-',marker='^')
    ax2.plot(item.tt, p_target, color='red', linestyle='--')
    ax2.set_xlabel('t')
    ax2.set_ylabel('p')
    
    if IFplot_best:
        ax1.plot(item.tt, item.pos_best[0,0,:], linestyle=':',color='green')
        ax2.plot(item.tt, item.mom_best[0,0,:], linestyle=':',color='green')
    
    x = item.pos[0,0,:]; y = item.mom[0,0,:]; z = item.tt
    Plot_phase3D(x, y, z, ['q', 'p', 't'], ax3, fig)
    Plot_phase3D(q_target, p_target, z, ['q', 'p', 't'], ax3, fig, color='red')
    
class PhySystem_Oscillatordis(PhySystem):
    # only one mass and one spring
    mass = 0.0
    k = 0.0
    def __init__(self, init_position, init_momentum, t_end, n_split ,mass, k, mu, IF_dissipative, device='cpu'):
        super().__init__(init_position, init_momentum, t_end, n_split, IF_dissipative, device)
        
        # oscillator specific
        self.mass = np.array(mass)
        self.k = np.array(k)
        self.omega = np.sqrt(self.k / self.mass)
        self.mu = mu
        
        # 检查质量参数维度
        if self.mass.size != self.num_of_obj:
            raise ValueError("Mass array size must match number of objects")

    def HamiltonFnc_np(self, q, p, t_now=0.0): # not used
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
        H = torch.sum(kinetic, dim=0)
        
        
        ## compute distance ## potential
        mask = torch.zeros(new_shape, device=self.device); 
        mask[:, 0, ...] = 1.0
        x_extended  = torch.sum(q_permuted * mask, dim=1)
            
        potential =  0.5  * torch.tensor(self.k, device=self.device) * (x_extended)**2
        H += torch.sum(potential,dim=0)

        return H #shape: [n_split, b]
    
    def dHdq_dissipative(self, q, p):
        '''
        Arg: q: [batch_size, num_obj, num_dim, n_split] or [num_obj, num_dim, n_split]
        '''
        if q.dim() == 3:
            mass_tensor = torch.as_tensor(self.mass, dtype=p.dtype, device=self.device).view([len(self.mass), 1, 1])
        elif q.dim() == 4:
            mass_tensor = torch.as_tensor(self.mass, dtype=p.dtype, device=self.device).view([1, len(self.mass), 1, 1])
              
        return + self.mu * p / mass_tensor
    
    def Ed_dissipative(self, q, p):
        '''
        Arg: q: [batch_size, num_obj, num_dim, n_split] or [num_obj, num_dim, n_split]
        '''            
        dq    = (q[...,1:] - q[...,:-1])      # shape: [batch_size, num_obj, num_dim, n_split]
        fd = - self.dHdq_dissipative(q, p).detach().clone().to(self.device) # shape: [batch_size, num_obj, num_dim, n_split]
        fd_mid = (fd[...,:-1] + fd[...,1:])/2.
        fd_mid = fd_mid  # no gradients
        dEd = torch.sum( - fd_mid * dq, dim=(-3,-2) )  # shape: [batch_size, n_split]
        Ed = torch.cumsum(  dEd, dim=(-1))  # shape: [batch_size, n_split]
        if q.dim() == 3:
            Ed = torch.cat( (torch.zeros(1, device=self.device, dtype=Ed.dtype), Ed), dim=-1)
        elif q.dim() == 4:
            Ed = torch.cat( (torch.zeros(Ed.shape[0], 1, device=self.device, dtype=Ed.dtype), Ed), dim=-1)  # shape: [batch_size, n_split+1]
        return Ed

def OscillatorDis_reparam(t_end,
                          n_split,
                          mass, k, mu,
                          known_pos, known_mom,
                          init_position, init_momentum,
                        pos_end_cond, mom_end_cond,
                        Hconfig):
    
    Oscillator_another = PhySystem_Oscillatordis(init_position=init_position,
                                init_momentum=init_momentum,
                                t_end = t_end,
                                n_split = n_split,
                                mass= mass,
                                k=  k,
                                mu = mu,
                                IF_dissipative=True,
                                device='cuda' if torch.cuda.is_available() else 'cpu')
    

    # transfer to new system
    Oscillator_another.pos = copy.deepcopy(known_pos)
    Oscillator_another.mom = copy.deepcopy(known_mom)   

    if known_pos[:,:,0] != init_position:
        Oscillator_another.pos[:,:,0] = init_position
    if known_mom[:,:,0] != init_momentum:
        Oscillator_another.mom[:,:,0] = init_momentum

    pos_end_cond = np.array(pos_end_cond)
    mom_end_cond = np.array(mom_end_cond)
    
    eps = 1e-12
    curves = Oscillator_another.Optimize_NewParams(
            pos_end_cond= pos_end_cond, 
            mom_end_cond= mom_end_cond,
            eps_q = eps,
            lambda_q = Hconfig['lambda_q'], # 1.0, #
            lambda_p = Hconfig['lambda_p'],
            eps_p = eps,                                      
            num_samples = (n_split+1) // 2 * 2 // 3 ,  # should  slightly larger than  n_split // 2 * 2 // 3  the first 2 is maximun frequency, the second 2 is Nyquist thoerm, the third 3 is samples times 
            IF_progressive=False,
            reduce_op_type='mean',  
            lr = Hconfig['lr'],  #case2:1e-0; case3:2e-1; case4-2:2e-2;
            lr_scheduler='Onplateau', #'Cos'
            lr_updatet= Hconfig['lr_updatet'], # case2:5000;case3:5000; case4-2:5000
            lr_decay=  Hconfig['lr_decay'], #case2:0.7; case3:0.7 case4-2:0.9
            num_iterations= Hconfig['num_iterations'],  #case2:25000;case3:30000; case4-2:250000
            dt_expect= t_end / n_split,
            IF_uniform_dt=True,
            t_min = 0.0,    # IF given t_min, then set t_min
            t_end_given = t_end,    # IF not given t_end, then set t_end None
            period_updatet = 9999999, # meaningless here
            decay_updatet = 0.9999999, # meaningless here
            lambda_smooth_q_1st = Hconfig['lambda_smooth_q_1st'], #case2:1e-2;case3:[1e-2, 5e-3]; case4-2:1e-2
            lambda_smooth_p_1st = Hconfig['lambda_smooth_p_1st'], #case2:1e-2;case3:[1e-2, 5e-3]; case4-2:1e-2 
            IF_weight = False,
            IF_autoscale_loss= Hconfig['IF_autoscale_loss'],
            perturbation_type = Hconfig['perturbation_type'], #case4-2:'sincos'  
            variation_method = 'direct_integral', #'indirect_substract', #
            optimizers=['SGDM'])
    
    return Oscillator_another, curves
#%%
if __name__ == "__main__":
    # test params
    mass = [1.0]
    k = 0.5
    mu_cr = 2 * np.sqrt(k / mass[0])   # dissipation coefficient critical damping: mu = 2*sqrt(m*k)
    mu =    1.5  # dissipation coefficient
    init_position = np.array([[1.0]])
    init_momentum = np.array([[0.0]])
    t_end =  50.0
    n_split = 1024

    Oscillator = PhySystem_Oscillatordis(init_position=init_position,
                                      init_momentum=init_momentum,
                                      t_end = t_end,
                                      n_split = n_split,
                                      mass=mass,
                                      k = k,
                                      mu = mu,
                                      IF_dissipative=True)    
    
    eps = 1e-12
    IFsolve = False
    if IFsolve:
        Oscillator.Solver(
                eps_q = eps, 
                eps_p = eps, 
                n_integral = 20, 
                num_samples = 5, 
                lr = 2e-3, 
                num_iterations=100,
                perturbation_type='sin')
    else:
        Oscillator.pos[0,0,:], Oscillator.mom[0,0,:]  = plot(Oscillator, None, None, None, None, IFplot=False)

    IFreparams = True
    if IFreparams:
        from Hconfigs.Q3_C2_config import get_config
        Hconfig = get_config()
        Oscillator_another, curves = OscillatorDis_reparam(t_end,
                                                          n_split,
                                                          [0.1], 0.1, 0.1,
                                                          Oscillator.pos, Oscillator.mom,
                                                          init_position, init_momentum,
                                                          np.array([[0.0]]), np.array([[0.0]]), 
                                                          Hconfig=Hconfig)
            
        

            
#%% 
    fig = plt.figure()
    ax1 = fig.add_subplot(3,3,1)
    ax2 = fig.add_subplot(3,3,2)
    ax3 = fig.add_subplot(3,3,3, projection='3d')
    ax4 = fig.add_subplot(3,3,4)
    ax5 = fig.add_subplot(3,3,5)
    ax6 = fig.add_subplot(3,3,6, projection='3d')
    ax7 = fig.add_subplot(3,3,7)
    ax8 = fig.add_subplot(3,3,8)
    ax9 = fig.add_subplot(3,3,9, projection='3d')
    
    
    plot(Oscillator, fig, ax1, ax2, ax3)
    
    if IFreparams:
        plot(Oscillator_another, fig, ax4, ax5, ax6, IFplot_best=True)
        

        fig2 = plt.figure()
        ax2_1 = fig2.add_subplot(2,3,1)
        loss = np.array(Oscillator_another.loss_reparams)
        labels = ['reg_smooth', 'unphysics' , 'delta_q' , 'delta_p', 'total' ]
        for i_loss in range(loss.shape[1]):
            ax2_1.semilogy(np.linspace(0,1, loss.shape[0]), loss[:,i_loss], label=labels[i_loss])
        plt.legend()
        
        ax2_2 = fig2.add_subplot(2,3,2)
        lr = np.array(Oscillator_another.lr_reparams)
        ax2_2.semilogy(np.linspace(0,1, lr.shape[0]), lr, label='lr')
        
    plt.tight_layout()
    plt.legend()
    plt.show()
    
    from tools.Animation import animate_curves
    animate_curves(curves, interval=0.2, title='demonstration')
    



# %%
