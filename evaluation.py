import torch
import tqdm
import numpy as np
from scipy import integrate

def eval_train_Unet_based_Time_Dependent_Score_Model():   
    num_steps = 500 
    signal_to_noise_ratio = 0.16
    def pc_sampler(score_model,
                   marginal_prob_std,
                   diffusion_coeff,
                   batch_size=64,
                   num_steps=num_steps,
                   snr=signal_to_noise_ratio,
                   device='cuda',
                   eps=1e-3
                   ):
        """Generate samples from score-based models with Predictor-Corrector method.
        
        Arg:
            Score_model: A pytorch model that represents the time-dependent score-based model.
            marginal_prob_std: A function that gives the standard deviation
                of the perturbation kernel.
            diffusion_coeff: A function that gives the diffusion coefficient
                of the SDE.
            num_steps: The number of sampling steps.
                Equivalent to the number of discretized time steps.
            device: 'cuda' or 'cpu'
            eps: The smallest time step for numerical stability.
            
        Returns:
            Samples.
        """
        
        # Step1: define inital sample
        t = torch.ones(batch_size, device=device)
        init_x = torch.randn(batch_size, 1, 28, 28, device=device) \
            * marginal_prob_std(t)[:, None, None, None]
            
        # Step2: define the timesteps of reverse process
        time_steps = torch.linspace(1., eps, num_steps, device=device)
        step_size = time_steps[0] - time_steps[1]
        
        # Step3: solve
        x = init_x 
        with torch.no_grad():
            for time_step in tqdm.tqdm(time_steps):   
                batch_time_step = torch.ones(batch_size, device=device) * time_step
                
                # Corrector step (Langevin MCMC)
                grad = score_model(x, batch_time_step)
                grad_norm = torch.norm(grad.reshape(grad.shape[0],-1),dim=-1).mean()
                noise_norm = np.sqrt(np.prod(x.shape[1:]))
                langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
                print(f'langevin_step_size = {langevin_step_size}')
                
                for _ in range(10):
                    x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)
                    grad = score_model(x, batch_time_step)
                    grad_norm = torch.norm(grad.reshape(grad.shape[0],-1), dim=-1).mean()
                    noise_norm = np.sqrt(np.prod(x.shape[1:]))
                    langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
                    print(f'langevin_step_size = {langevin_step_size}')
                    
                # Predictor step (Euler-Maruyama)
                g = diffusion_coeff(batch_time_step)
                mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
                x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)
                
        # Step4: the last step does not include any noise
        return mean_x

        
        
        
                     