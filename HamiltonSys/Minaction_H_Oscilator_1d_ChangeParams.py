#%% 
import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


##
from . import Minaction_Hamilton_torch

##
PhySystem = Minaction_Hamilton_torch.PhySystem
Plot_phase3D =  Minaction_Hamilton_torch.Plot_phase3D


#%% user def function
def plot(Oscillator, fig, ax1, ax2, ax3):
    q_target = Oscillator.pos_best[0,0,0] * np.cos(Oscillator.omega * Oscillator.tt) + \
               Oscillator.mom_best[0,0,0] / Oscillator.omega / Oscillator.mass[0] * np.sin(Oscillator.omega * Oscillator.tt);
    p_target = - Oscillator.pos_best[0,0,0] * np.sin(Oscillator.omega * Oscillator.tt) * Oscillator.omega * Oscillator.mass[0] + \
               Oscillator.mom_best[0,0,0] * np.cos(Oscillator.omega * Oscillator.tt);
    ax1.plot(Oscillator.tt, Oscillator.pos_best[0,0,:],label='Pos', color='blue', linestyle='-',marker='s')
    ax1.plot(Oscillator.tt, q_target, color='red', linestyle='--')
    ax1.set_xlabel('t')
    ax1.set_ylabel('q')
    

    ax2.plot(Oscillator.tt, Oscillator.mom_best[0,0,:],label='Mom', color='blue', linestyle='-',marker='^')
    ax2.plot(Oscillator.tt, p_target, color='red', linestyle='--')
    ax2.set_xlabel('t')
    ax2.set_ylabel('p')
    
    x = Oscillator.pos[0,0,:]; y = Oscillator.mom[0,0,:]; z = Oscillator.tt
    Plot_phase3D(x, y, z, ['q', 'p', 't'], ax3, fig)
    Plot_phase3D(q_target, p_target, z, ['q', 'p', 't'], ax3, fig, color='red')
    
class PhySystem_Oscillator(PhySystem):
    # only one mass and one spring
    mass = 0.0
    k = 0.0
    def __init__(self, init_position, init_momentum, t_end, n_split ,mass, k, IF_dissipative=False, device='cpu'):
        super().__init__(init_position, init_momentum, t_end, n_split,IF_dissipative, device)
        
        # oscillator specific
        self.mass = np.array(mass)
        self.k = np.array(k)
        self.omega = np.sqrt(self.k / self.mass)
        
        # 检查质量参数维度
        if self.mass.size != self.num_of_obj:
            raise ValueError("Mass array size must match number of objects")

    # def HamiltonFnc_np(self, q, p, t_now=0.0):
    #     H = np.sum( np.sum(p**2, axis=1)/(2 * self.mass))
    #     for i_obj in range(self.num_of_obj):
    #         for j_obj in range(i_obj+1, self.num_of_obj):
    #             rr = 0.0
    #             for i_dim in range(self.dimensionality):
    #                 rr += np.square(q[i_obj, i_dim] - q[j_obj,i_dim])
    #             rr = np.sqrt(rr)
    #             H += self.G * self.mass[i_obj] * self.mass[j_obj] /rr 
    #     return H
    
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
    
def Oscillator_reparam(t_end,
                      n_split,
                      mass, k,
                      known_pos,
                      known_mom,
                      init_position,
                      init_momentum,
                      pos_end_cond,
                      mom_end_cond,
                      Hconfig):
    
    Oscillator_another = PhySystem_Oscillator(
        init_position=init_position,
        init_momentum=init_momentum,
        t_end = t_end,
        n_split=n_split,
        mass=[mass],
        k = k,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
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
            eps_p = eps,                                     
            lambda_q = 1.0,
            lambda_p = 1.0,
            num_samples = (n_split+1) // 2 * 2 // 3,  # should  slightly larger than  n_split // 2 * 2 // 3  the first 2 is maximun frequency, the second 2 is Nyquist thoerm, the third 3 is samples times 
            IF_progressive= False,
            reduce_op_type='mean',
            lr = Hconfig['lr'], #case2:2e-2; case3:similar to 2 case4:  to2
            lr_scheduler='Onplateau',
            lr_updatet= Hconfig['lr_updatet'], #case2:500; case3: similar to 2; case4: to2
            lr_decay=Hconfig['lr_decay'],
            num_iterations= Hconfig['num_iterations'], #case2:15000; case3:similar to 2; case4: 15000 
            dt_expect= t_end / n_split,
            IF_uniform_dt=True,
            t_min = t_end, # change minaction_Hamilton_torch line 247, Remove comments on 247-249
            t_end_given=None,
            period_updatet = Hconfig['period_updatet'], #case2:50; case3:similar to 2; casa4: to2
            decay_updatet = Hconfig['decay_updatet'], #case2:0.995; case3: similar to 2; case4: to2 
            lambda_smooth_q_1st=Hconfig['lambda_smooth_q_1st'],  #case2:[1e-1, 8e-3, 8e-4]; case3: similar to case2; case4: to2
            lambda_smooth_p_1st=Hconfig['lambda_smooth_p_1st'],  #case2:[1e-1, 8e-3, 8e-4]; case3: similar to case2; case4: to2
            perturbation_type= Hconfig['perturbation_type'], #'white_noise' 'sincos'
            variation_method='indirect_substract',
            optimizers=['Adam'])
    
    return Oscillator_another, curves
#%%
if __name__ == "__main__":
    # test params
    mass = [1.0]
    k = 1.0
    init_position = np.array([[0.0]])
    init_momentum = np.array([[1.0]])
    t_end = np.pi * 2
    n_split = 4 * 30

    Oscillator = PhySystem_Oscillator(init_position=init_position,
                                      init_momentum=init_momentum,
                                      t_end = t_end,
                                      n_split = n_split,
                                      mass=mass,
                                      k = k)    
    
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
        Oscillator.pos[0,0,:] =  np.sin(Oscillator.omega * Oscillator.tt) / Oscillator.omega / Oscillator.mass[0];
        Oscillator.mom[0,0,:]  = np.cos(Oscillator.omega * Oscillator.tt)

    IFreparams = True
    if IFreparams:
        Oscillator_another, curves = Oscillator_reparam(t_end,
                                                      n_split,
                                                      mass= mass,
                                                      k= k,
                                                      known_pos=Oscillator.pos,
                                                      known_mom=Oscillator.mom,
                                                      init_position=init_position,
                                                      init_momentum=init_momentum,
                                                      pos_end_cond=np.array([[0.5]]),
                                                      mom_end_cond=np.array([[np.inf]]))
       
        # Oscillator_another = PhySystem_Oscillator(init_position=init_position,
        #                                 init_momentum=init_momentum,
        #                                 t_end = t_end,
        #                                 n_split = n_split,
        #                                 mass= mass,
        #                                 k=  k,
        #                                 device='cuda' if torch.cuda.is_available() else 'cpu')
        

        # # transfer to new system
        # Oscillator_another.pos = copy.deepcopy(Oscillator.pos)
        # Oscillator_another.mom = copy.deepcopy(Oscillator.mom)   


        # pos_end_cond = np.array([[0.5]])
        # mom_end_cond = np.array([[np.inf]])

        # eps = 1e-12
        # curves = Oscillator_another.Optimize_NewParams(
        #         pos_end_cond= pos_end_cond, 
        #         mom_end_cond= mom_end_cond,
        #         eps_q = eps,
        #         eps_p = eps,                                      # n_split // 2 * 4  // 3  yzx thoerm
        #         lambda_q = 1.0,
        #         lambda_p = 1.0,
        #         num_samples = n_split // 2 * 2 // 3,  # should  slightly larger than  n_split // 2 * 2 // 3  the first 2 is maximun frequency, the second 2 is Nyquist thoerm, the third 3 is samples times 
        #         IF_progressive= False,
        #         reduce_op_type='mean',
        #         lr = [2e-2],
        #         lr_scheduler='Onplateau',
        #         lr_updatet=500, 
        #         lr_decay=0.7,
        #         num_iterations= 10000,
        #         dt_expect= t_end / n_split,
        #         IF_uniform_dt=True,
        #         t_min = t_end,
        #         period_updatet = 50,
        #         decay_updatet = 0.995,
        #         lambda_smooth_q_1st=8e-2,
        #         lambda_smooth_p_1st=8e-2,
        #         lambda_smooth_q_2nd=0e-1,
        #         lambda_smooth_p_2nd=0e-2,
        #         perturbation_type= 'white_noise',
        #         variation_method='indirect_substract',
        #         optimizers=['Adam'])
            

            
            

        

            
#%% 
    fig = plt.figure()
    ax1 = fig.add_subplot(2,3,1)
    ax2 = fig.add_subplot(2,3,2)
    ax3 = fig.add_subplot(2,3,3, projection='3d')
    ax4 = fig.add_subplot(2,3,4)
    ax5 = fig.add_subplot(2,3,5)
    ax6 = fig.add_subplot(2,3,6, projection='3d')
    
    plot(Oscillator, fig, ax1, ax2, ax3)
    
    if IFreparams:
        plot(Oscillator_another, fig, ax4, ax5, ax6)
        
        fig2 = plt.figure()
        ax2_1 = fig2.add_subplot(2,3,1)
        loss = np.array(Oscillator_another.loss_reparams)
        labels = ['reg_smooth', 'unphysics' ,'delta_q' , 'delta_p', 'total' ]
        for i_loss in range(loss.shape[1]):
            ax2_1.semilogy(np.linspace(0,1, loss.shape[0]), loss[:,i_loss], label=labels[i_loss])
    
    plt.tight_layout()
    plt.legend()
    plt.show()
    
    from tools.Animation import animate_curves
    animate_curves(curves, interval=0.02, title='demonstration')
    



# %%
