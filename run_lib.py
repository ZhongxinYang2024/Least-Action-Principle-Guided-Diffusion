# coding=utf-8
"""Training and evaluation for score-based generative models. """
import os
import tensorflow as tf
from torch.utils import tensorboard
import datasets
import logging
import torch
import numpy as np
import time
import io
import matplotlib.pyplot as plt
from scipy import io as sio
##
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import losses
from utils import save_checkpoint, restore_checkpoint
import sde_lib
import sampling
import datasets
from tools.Animation import animate_curves
from data.CustomDataLoader import infinite_iterator
from data.CustomDataLoader import CustomDataset_SIN
from data.CustomDataLoader import CustomDataset_FreeFall
from data.CustomDataLoader import CustomDataset_BubbleNV
from data.CustomDataLoader import CustomDataset_BubbleV
from data.CustomDataLoader import CustomDataset_VortexSys
from data.CustomDataLoader import CustomDataset_Foil

from HamiltonSys.Minaction_H_freefall_1d_ChangeParams import FreeFall_reparam
from HamiltonSys.Minaction_H_Oscilator_1d_ChangeParams import Oscillator_reparam
from HamiltonSys.Minaction_H_OscilatorDis_1d_ChangeParams import OscillatorDis_reparam


#%% 
def train(config, workdir):
    """Runs the training pipeline.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """
    
    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    tf.io.gfile.makedirs(sample_dir)
    
    tb_dir = os.path.join(workdir, "tensorboard")
    tf.io.gfile.makedirs(tb_dir)
    writer = tensorboard.SummaryWriter(tb_dir)
    
    # Initialize model
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
    
    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", 'checkpoint.pth')
    tf.io.gfile.makedirs(checkpoint_dir)
    tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
    #Resume training when intermediate checkpoints are detected
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])
    
    match config.data.dataset.lower():
        case 'oscillator':
            dataset = CustomDataset_SIN(resolution=config.data.height).transfer2ds(config.training.batch_size)
        case 'freefall':
            dataset = CustomDataset_FreeFall(resolution=config.data.height).transfer2ds(config.training.batch_size)
        case 'bubblenv':
            dataset = CustomDataset_BubbleNV(resolution=config.data.height).transfer2ds(config.training.batch_size)
        case 'bubblev':
            dataset = CustomDataset_BubbleV(resolution=config.data.height).transfer2ds(config.training.batch_size)
        case 'vortexsys':
            dataset = CustomDataset_VortexSys(resolution=config.data.height).transfer2ds(config.training.batch_size)
        case 'foil':
            dataset = CustomDataset_Foil(height=config.data.height, width=config.data.width).transfer2ds(config.training.batch_size)
        case _:
            raise ValueError(f"Dataset {config.data.dataset} not recognized.")
            
    train_ds = dataset[0]
    eval_ds = dataset[1]
    train_iter = infinite_iterator(train_ds)
    eval_iter = infinite_iterator(eval_ds)
    
    
    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    
    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min = config.beta_min,
                            beta_max = config.beta_max,
                            N = config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, 
                               beta_max=config.model.beta_max, 
                               N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, 
                            sigma_max=config.model.sigma_max, 
                            N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")
      
    # Build one-step training and evaluation funcitons
    optimize_fn = losses.optimization_manager(config)
    reduce_mean = config.training.reduce_mean
    continuous = config.training.continuous
    likelihood_weighting = config.training.likelihood_weighting
    
    If_conditional_physics   = config.model.conditional_physics # yzx added 250904
    If_conditional_physics_distributed = config.model.conditional_physics_distributed # yzx added 250907
    
    train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                       reduce_mean=reduce_mean, continuous=continuous,
                                       likelihood_weighting=likelihood_weighting,
                                       If_conditional_physics=If_conditional_physics,
                                       If_conditional_physics_distributed=If_conditional_physics_distributed)
    eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                      reduce_mean=reduce_mean, continuous=continuous,
                                      likelihood_weighting=likelihood_weighting,
                                      If_conditional_physics=If_conditional_physics,
                                      If_conditional_physics_distributed=If_conditional_physics_distributed)
      
    # Build sampling functions
    if config.training.snapshot_sampling:
        sampling_shape = (config.training.batch_size, config.data.num_channels,
                      config.data.height, config.data.weight)
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)
    
    num_train_steps = config.training.n_iters
    
    # In case there are multiple hosts (e.g., TPU pods), only log to host 0
    logging.info("Starting training loop at step %d." % (initial_step,))

    for step in range(initial_step, num_train_steps + 1):
        # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
        Xy_train = next(train_iter)
        
        if If_conditional_physics_distributed:
            batch = [torch.from_numpy(Xy_train[0].numpy()).to(config.device).float(),
                    torch.from_numpy(Xy_train[1].numpy()).to(config.device).float(),
                    torch.from_numpy(Xy_train[2].numpy()).to(config.device).float()] 
        else:       
            batch = [torch.from_numpy(Xy_train[0].numpy()).to(config.device).float(),
                    torch.from_numpy(Xy_train[1].numpy()).to(config.device).float(),
                    None] 
        
        #batch = scaler(batch) no need to scaler
        
        # Execute one training step
        loss = train_step_fn(state, batch)
        if step % config.training.log_freq == 0:
            logging.info("step: %d, training_loss : %.5e" % (step, loss.item()))
            writer.add_scalar("training_loss", loss, step)
            
        # Save a temporary checkpoint to resume training after pre-emption periodically
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            save_checkpoint(checkpoint_meta_dir, state)

        # Report the loss on an evaluation dataset periodically
        if step % config.training.eval_freq == 0:
            Xy_train = next(eval_iter)
            
            if If_conditional_physics_distributed:
                eval_batch = [torch.from_numpy(Xy_train[0].numpy()).to(config.device).float(),
                        torch.from_numpy(Xy_train[1].numpy()).to(config.device).float(),
                        torch.from_numpy(Xy_train[2].numpy()).to(config.device).float()] 
            else:       
                eval_batch = [torch.from_numpy(Xy_train[0].numpy()).to(config.device).float(),
                        torch.from_numpy(Xy_train[1].numpy()).to(config.device).float(),
                        None] 
                
            #eval_batch = scaler(eval_batch)
            eval_loss = eval_step_fn(state, eval_batch)
            logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))
            writer.add_scalar("eval_loss", eval_loss.item(), step)
            
        # Save a checkpoint periodically and generate samples if needed
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
            # Save the checkpoint.
            save_step = step // config.training.snapshot_freq
            save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

            # Generate and save samples
            if config.training.snapshot_sampling:
                ema.store(score_model.parameters())
                ema.copy_to(score_model.parameters())
                sample, n = sampling_fn(score_model)
                ema.restore(score_model.parameters())
                this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                tf.io.gfile.makedirs(this_sample_dir)
                sample = sample.cpu().numpy()
                
                match config.data.dataset.lower():
                    case 'oscillator' | 'freefall' | 'bubblenv':      
                        with tf.io.gfile.GFile(
                            os.path.join(this_sample_dir, "samples.np"), "wb") as fout:
                            np.save(fout, sample)
                    case _:
                        raise ValueError(f"Dataset {config.data.dataset} not recognized.")  
                    
                        


def evaluate(config, workdir, eval_folder='eval',Hconfig=None):
    """Evaluate trained model.
    
    Args:
        config: Configuration to use.
        workdir: Working directory for checkpoints.
        eval_folder: The subfolder for storing evaluation results. Default to "eval"
    """
    # Create directory to eval_folder
    eval_dir = os.path.join(workdir, eval_folder)
    tf.io.gfile.makedirs(eval_dir)
    
    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    
    # Initialize model
    score_model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
    
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    
    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")
    
    # Build the sampling function when sampling is enabled
    if config.eval.enable_sampling:
        sampling_shape = (config.eval.batch_size, config.data.num_channels,
                            config.data.height, config.data.width)
        If_conditional_physics = config.model.conditional_physics
        If_conditional_physics_distributed = config.model.conditional_physics_distributed
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps,
                                               If_conditional_physics=If_conditional_physics,
                                               If_conditional_physics_distributed=If_conditional_physics_distributed)


    begin_ckpt = config.eval.begin_ckpt
    logging.info("begin checkpoint: %d" % (begin_ckpt,))
    for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
        # Wait if the target checkpoint doesn't exist yet
        waiting_message_printed = False
        ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
        while not tf.io.gfile.exists(ckpt_filename):
            if not waiting_message_printed:
                logging.warning("Waiting for the arrival of checkpoint_%d" % (ckpt,))
                waiting_message_printed = True
            time.sleep(60)

        # Wait for 2 additional mins in case the file exists but is not ready for reading
        ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
        try:
            state = restore_checkpoint(ckpt_path, state, device=config.device)
        except:
            time.sleep(60)
            try:
                state = restore_checkpoint(ckpt_path, state, device=config.device)
            except:
                time.sleep(120)
                state = restore_checkpoint(ckpt_path, state, device=config.device)
        ema.copy_to(score_model.parameters())


        # Generate samples 
        if config.eval.enable_sampling:
            num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
            for r in range(num_sampling_rounds):
                logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))

                # Directory to save samples. Different for each host to avoid writing conflicts
                this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")
                tf.io.gfile.makedirs(this_sample_dir)

                
                match config.data.dataset.lower():
                    case 'oscillator':
                        samples, n = sampling_fn(score_model)
                        samples = samples.cpu().numpy()
                        from TestCode3_eval_Con import plot_matrix_rows
                        samples_show = samples[0:16,0,:,0]
                        plot_matrix_rows(np.linspace(0,1,samples_show.shape[1]),samples_show)
                    case 'freefall':
                        save_path = f'./HamiltonSys/log/{config.data.dataset.lower()}'
                        from TestCode3_eval_Con import plot_matrix_rows
                        
                        scalar_g = Hconfig['scalar_g']
                        scalar_T = Hconfig['scalar_T']
                        scalar_m = Hconfig['scalar_m']
                        
                        preprocess_g = Hconfig['preprocess_g']
                        target_g  = Hconfig['target_g']
                        standard_T = Hconfig['standard_T']
                        preprocess_T = Hconfig['preprocess_T']
                        target_T = Hconfig['target_T']
                        
                        
                        ## ML itself
                        phy_cond_concentrate = torch.tensor([target_g / scalar_g] * config.eval.batch_size).to(config.device).unsqueeze(-1)
                        phy_cond = [phy_cond_concentrate, None]
                        samples, n = sampling_fn(score_model, phy_cond=phy_cond)
                        samples = samples.cpu().numpy().astype(np.float64)
                        q_show = samples[:,0,:,0] * (- 0.5 * scalar_g * scalar_T**2)
                        p_show = samples[:,1,:,0] * (- scalar_m * scalar_g * scalar_T)
                        plot_matrix_rows(np.linspace(0, standard_T,q_show.shape[-1]),q_show)
                        plot_matrix_rows(np.linspace(0, standard_T,p_show.shape[-1]),p_show)
                        
                        
                        ## HamiltonSys + ML
                        phy_cond_concentrate = torch.tensor([preprocess_g / scalar_g] * config.eval.batch_size).to(config.device).unsqueeze(-1)
                        phy_cond = [phy_cond_concentrate, None]
                        samples_pre, n = sampling_fn(score_model, phy_cond=phy_cond)
                        samples_pre = samples_pre.cpu().numpy().astype(np.float64)
                        

                        os.makedirs(save_path, exist_ok=True)
                        for i in range(config.eval.batch_size):
                            ## ML itself
                            np.savetxt(f'{save_path}/G{target_g:.1f}_T{target_T:.1f}_MlS{i}.dat', 
                                    np.concatenate([np.linspace(0, standard_T, config.data.height).reshape(-1,1), 
                                                    np.full_like(np.linspace(0, standard_T, config.data.height).reshape(-1,1), target_g),
                                                    (samples[i, 0,:,0] * (- 0.5 * scalar_g * scalar_T**2)).reshape(-1,1),
                                                    (samples[i, 1,:,0] * (- scalar_m * scalar_g * scalar_T)).reshape(-1,1)],axis=-1), 
                                        fmt='%.6e', header='t    g    q    p',  
                                        delimiter='\t')
                            
                            ## HamiltonSys + ML                          
                            np.savetxt(f'{save_path}/G{preprocess_g:.1f}_T{preprocess_T:.1f}_PreS{i}.dat', 
                                    np.concatenate([np.linspace(0, preprocess_T, config.data.height).reshape(-1,1), 
                                                    np.full_like(np.linspace(0, preprocess_T, config.data.height).reshape(-1,1), preprocess_g),
                                                    (samples_pre[i, 0,:,0] * (- 0.5 * scalar_g * scalar_T**2)).reshape(-1,1),
                                                    (samples_pre[i, 1,:,0] * (- scalar_m * scalar_g * scalar_T)).reshape(-1,1)],axis=-1), 
                                        fmt='%.6e', header='t    g    q    p',  
                                        delimiter='\t')
                            
                            
                            if Hconfig['if_use_hamiltonian']:
                                FreeFall_another, curves = FreeFall_reparam(preprocess_T, 
                                                                    config.data.height-1,
                                                                    target_g,
                                                                    samples_pre[i:i+1,0:1,:,0] * (- 0.5 * scalar_g * scalar_T**2), # obj, dim, time
                                                                    samples_pre[i:i+1,1:2,:,0] * (- scalar_m * scalar_g * scalar_T),
                                                                    np.array([[0.0]]),
                                                                    np.array([[0.0]]),
                                                                    np.array([[- 0.5 * target_g * target_T**2]]),
                                                                    np.array([[np.inf]]),
                                                                    Hconfig)
                            
                            
                                np.savetxt(f'{save_path}/G{target_g:.1f}_T{target_T:.1f}_OurS{i}.dat', 
                                        np.concatenate([FreeFall_another.tt.reshape(-1,1), 
                                                        np.full_like(FreeFall_another.tt.reshape(-1,1), target_g),
                                                        FreeFall_another.pos_best[0,0,:].reshape(-1,1),
                                                        FreeFall_another.mom_best[0,0,:].reshape(-1,1)],axis=-1), 
                                            fmt='%.6e', header='t    g    q    p',  
                                            delimiter='\t')

                                if Hconfig['if_obey_batchsize'] == False:
                                    break
                            
                        fig = plt.figure()
                        ax1 = fig.add_subplot(2,3,1)
                        ax2 = fig.add_subplot(2,3,2)
                        ax3 = fig.add_subplot(2,3,3, projection='3d')
                        from HamiltonSys.Minaction_H_freefall_1d_ChangeParams import plot
                        plot( FreeFall_another, fig, ax1, ax2, ax3)
                        animate_curves(curves, interval=0.3, title='demonstration')
                        
                    case 'bubblenv':
                        save_path = f'./HamiltonSys/log/{config.data.dataset.lower()}'
                        from TestCode3_eval_Con import plot_matrix_rows
                        
                        num_T = Hconfig['num_T'];  # num of T in training set
                        preprocess_q0 = Hconfig['preprocess_q0']; 
                        target_q0 = Hconfig['target_q0']; 
                        preprocess_p0 = Hconfig['preprocess_p0'];
                        target_p0 = Hconfig['target_p0'];
                        preprocess_kk = Hconfig['preprocess_kk']; 
                        target_kk = Hconfig['target_kk'];
                        preprocess_mm = Hconfig['preprocess_mm']; 
                        target_mm = Hconfig['target_mm'];
                        omega = np.sqrt(target_kk / target_mm)
                        preprocess_T = 2. * np.pi * 1.0 / np.sqrt(preprocess_kk / preprocess_mm);
                        target_T = 2. * np.pi * 1.0 / np.sqrt(target_kk / target_mm)

                        
                        ## ML itself
                        phy_cond_concentrate = torch.tensor([target_q0, target_p0, target_kk]).unsqueeze(0).repeat(config.eval.batch_size,1).to(config.device)
                        phy_cond = [phy_cond_concentrate, None]
                        samples, n = sampling_fn(score_model, phy_cond=phy_cond)
                        samples = samples.cpu().numpy().astype(np.float64)
                        q_show = samples[0:1,0,:,0] 
                        p_show = samples[0:1,1,:,0] 
                        plot_matrix_rows(np.linspace(0,target_T,q_show.shape[-1]),q_show)
                        plot_matrix_rows(np.linspace(0,target_T,p_show.shape[-1]),p_show)
                        
                        
                        phy_cond_concentrate = torch.tensor([preprocess_q0, preprocess_p0, preprocess_kk]).unsqueeze(0).repeat(config.eval.batch_size,1).to(config.device)
                        phy_cond = [phy_cond_concentrate, None]
                        samples_pre, n = sampling_fn(score_model, phy_cond=phy_cond)
                        samples_pre = samples_pre.cpu().numpy().astype(np.float64)

                        
                        
                        
                        os.makedirs(save_path, exist_ok=True)
                        for i in range(config.eval.batch_size):
                            ## ML itself
                            np.savetxt(f'{save_path}/q0{target_q0:.1f}_p0{target_p0:.1f}_kk{target_kk:.1f}_mm{target_mm:.1f}_T{target_T * num_T:.1f}_MlS{i}.dat', 
                                        np.concatenate([np.linspace(0, target_T, config.data.height).reshape(-1,1), 
                                                        np.full_like(np.linspace(0, 1, config.data.height).reshape(-1,1), target_q0),
                                                        np.full_like(np.linspace(0, 1, config.data.height).reshape(-1,1), target_p0),
                                                        np.full_like(np.linspace(0, 1, config.data.height).reshape(-1,1), target_kk),
                                                        np.full_like(np.linspace(0, 1, config.data.height).reshape(-1,1), target_mm),
                                                        samples[i:i+1,0,:,0].reshape(-1,1),
                                                        samples[i:i+1,1,:,0].reshape(-1,1)],axis=-1), 
                                        fmt='%.6e', header='t    q0    p0    kk    mm    q    p',  
                                        delimiter='\t')
                            
                            ## HamiltonSys + ML
                            np.savetxt(f'{save_path}/q0{preprocess_q0:.1f}_p0{preprocess_p0:.1f}_kk{preprocess_kk:.1f}_mm{preprocess_mm:.1f}_T{preprocess_T:.1f}_PreS{i}.dat', 
                                        np.concatenate([np.linspace(0, preprocess_T, config.data.height).reshape(-1,1), 
                                                        np.full_like(np.linspace(0, preprocess_T, config.data.height).reshape(-1,1), preprocess_q0),
                                                        np.full_like(np.linspace(0, preprocess_T, config.data.height).reshape(-1,1), preprocess_p0),
                                                        np.full_like(np.linspace(0, preprocess_T, config.data.height).reshape(-1,1), preprocess_kk),
                                                        np.full_like(np.linspace(0, preprocess_T, config.data.height).reshape(-1,1), preprocess_mm),
                                                        samples_pre[i:i+1,0,:,0].reshape(-1,1),
                                                        samples_pre[i:i+1,1,:,0].reshape(-1,1)],axis=-1), 
                                        fmt='%.6e', header='t    q0    p0    kk    mm    q    p',  
                                        delimiter='\t')
                            
                            if Hconfig['if_use_hamiltonian']:
                                for i_T in range(int(num_T + 0.5)):
                                    if i_T == 0:
                                        T_guess = preprocess_T
                                        q0_target = target_q0
                                        p0_target = target_p0
                                        qend_target =   target_q0 * np.cos(omega * target_T) + target_p0 / omega / target_mm * np.sin(omega * target_T) 
                                        q_init = samples[i:i+1,0:1,:,0]
                                        p_init = samples[i:i+1,1:2,:,0]
                                    else:
                                        T_guess = Oscillator_another.tt[-1]
                                        q0_target = Oscillator_another.pos_best[0,0,-1]
                                        p0_target = Oscillator_another.mom_best[0,0,-1]
                                        qend_target =   target_q0 * np.cos(omega * target_T) + target_p0 / omega / target_mm * np.sin(omega * target_T)
                                        q_init = Oscillator_another.pos_best
                                        p_init = Oscillator_another.mom_best
                                        
                                                                    
                                    Oscillator_another, curves = Oscillator_reparam(T_guess, 
                                                                        config.data.height-1,
                                                                        target_mm, target_kk,
                                                                        q_init, # obj, dim, time
                                                                        p_init,
                                                                        np.array([[q0_target]]),
                                                                        np.array([[p0_target]]),
                                                                        np.array([[qend_target]]),
                                                                        np.array([[np.inf]]),
                                                                        Hconfig)
                            
                            
                            
                                    np.savetxt(f'{save_path}/q0{target_q0:.1f}_p0{target_p0:.1f}_kk{target_kk:.1f}_mm{target_mm:.1f}_T{target_T * num_T:.1f}_OurS{i}_n{i_T}.dat', 
                                                np.concatenate([Oscillator_another.tt.reshape(-1,1), 
                                                                np.full_like(Oscillator_another.tt.reshape(-1,1), target_q0),
                                                                np.full_like(Oscillator_another.tt.reshape(-1,1), target_p0),
                                                                np.full_like(Oscillator_another.tt.reshape(-1,1), target_kk),
                                                                np.full_like(Oscillator_another.tt.reshape(-1,1), target_mm),
                                                                Oscillator_another.pos_best[0,0,:].reshape(-1,1),
                                                                Oscillator_another.mom_best[0,0,:].reshape(-1,1)],axis=-1), 
                                                fmt='%.6e', header='t    q0    p0    kk    mm    q    p',  
                                                delimiter='\t')
                                    
                                if Hconfig['if_obey_batchsize'] == False:
                                    break

                        fig = plt.figure()
                        ax1 = fig.add_subplot(2,3,1)
                        ax2 = fig.add_subplot(2,3,2)
                        ax3 = fig.add_subplot(2,3,3, projection='3d')
                        ax4 = fig.add_subplot(2,3,4)
                        from HamiltonSys.Minaction_H_Oscilator_1d_ChangeParams import plot
                        plot( Oscillator_another, fig, ax1, ax2, ax3)
                        ax4.loglog(np.linspace(1e-5,1,len(Oscillator_another.loss_reparams)), Oscillator_another.loss_reparams)
                        ax4.loglog(np.linspace(1e-5,1,len(Oscillator_another.lr_reparams)), Oscillator_another.lr_reparams)
                        animate_curves(curves, interval=0.3, title='demonstration')

                    case 'bubblev':
                        save_path = f'./HamiltonSys/log/{config.data.dataset.lower()}'
                        from TestCode3_eval_Con import plot_matrix_rows
                        
                        preprocess_q0 = Hconfig['preprocess_q0']; 
                        target_q0 = Hconfig['target_q0'];
                        preprocess_kk = Hconfig['preprocess_kk']; 
                        target_kk = Hconfig['target_kk'];
                        preprocess_mu = Hconfig['preprocess_mu']; 
                        target_mu = Hconfig['target_mu'];
                        preprocess_mm = Hconfig['preprocess_mm']; 
                        target_mm = Hconfig['target_mm'];
                        preprocess_T = Hconfig['preprocess_T']; 
                        target_T = Hconfig['target_T'];

                        ## ML itself
                        phy_cond_concentrate = torch.tensor([target_q0, target_kk, target_mu]).unsqueeze(0).repeat(config.eval.batch_size,1).to(config.device)
                        phy_cond = [phy_cond_concentrate, None] 
                        samples, n = sampling_fn(score_model, phy_cond=phy_cond)
                        samples = samples.cpu().numpy().astype(np.float64)
                        q_show = samples[0:1,0,:,0] 
                        p_show = samples[0:1,1,:,0] 
                        plot_matrix_rows(np.linspace(0,target_T,q_show.shape[-1]),q_show)
                        plot_matrix_rows(np.linspace(0,target_T,p_show.shape[-1]),p_show)
                        
                        ## HamiltonSys + ML
                        phy_cond_concentrate = torch.tensor([preprocess_q0, preprocess_kk, preprocess_mu]).unsqueeze(0).repeat(config.eval.batch_size,1).to(config.device)
                        phy_cond = [phy_cond_concentrate, None]
                        samples_pre, n = sampling_fn(score_model, phy_cond=phy_cond)
                        samples_pre = samples_pre.cpu().numpy().astype(np.float64)
                        

                        os.makedirs(save_path, exist_ok=True)
                        for i in range(config.eval.batch_size):
                            ## ML itself
                            np.savetxt(f'{save_path}/q0{target_q0:.1f}_kk{target_kk:.1f}_mu{target_mu:.1f}_mm{target_mm:.1f}_MlS{i}.dat', 
                                        np.concatenate([np.linspace(0, target_T, config.data.height).reshape(-1,1), 
                                                        np.full_like(np.linspace(0, 1, config.data.height).reshape(-1,1), target_q0),
                                                        np.full_like(np.linspace(0, 1, config.data.height).reshape(-1,1), target_kk),
                                                        np.full_like(np.linspace(0, 1, config.data.height).reshape(-1,1), target_mu),
                                                        np.full_like(np.linspace(0, 1, config.data.height).reshape(-1,1), target_mm),
                                                        samples[i,0,:,0].reshape(-1,1),
                                                        samples[i,1,:,0].reshape(-1,1)],axis=-1), 
                                        fmt='%.6e', header='t    q0    kk    mu    mm    q    p',  
                                        delimiter='\t')
                                                        
                            np.savetxt(f'{save_path}/q0{preprocess_q0:.1f}_kk{preprocess_kk:.1f}_mu{preprocess_mu:.1f}_mm{preprocess_mm:.1f}_PreS{i}.dat', 
                                        np.concatenate([np.linspace(0, preprocess_T, config.data.height).reshape(-1,1), 
                                                        np.full_like(np.linspace(0, preprocess_T, config.data.height).reshape(-1,1), preprocess_q0),
                                                        np.full_like(np.linspace(0, preprocess_T, config.data.height).reshape(-1,1), preprocess_kk),
                                                        np.full_like(np.linspace(0, preprocess_T, config.data.height).reshape(-1,1), preprocess_mu),
                                                        np.full_like(np.linspace(0, preprocess_T, config.data.height).reshape(-1,1), preprocess_mm),
                                                        samples_pre[i,0,:,0].reshape(-1,1),
                                                        samples_pre[i,1,:,0].reshape(-1,1)],axis=-1), 
                                        fmt='%.6e', header='t    q0    kk    mu    mm    q    p',  
                                        delimiter='\t')
                            
                            if Hconfig['if_use_hamiltonian']:            
                                Oscillator_another, curves = OscillatorDis_reparam(preprocess_T, 
                                                                    config.data.height-1,
                                                                    [target_mm], target_kk, target_mu,
                                                                    known_pos=samples_pre[i:i+1,0:1,:,0], # obj, dim, time
                                                                    known_mom=samples_pre[i:i+1,1:2,:,0],
                                                                    init_position=np.array([[target_q0]]),
                                                                    init_momentum=np.array([[0.0]]),
                                                                    pos_end_cond=np.array([[0.0]]),
                                                                    mom_end_cond=np.array([[0.0]]),
                                                                    Hconfig=Hconfig)
                                
                                
                                np.savetxt(f'{save_path}/q0{target_q0:.1f}_kk{target_kk:.1f}_mu{target_mu:.1f}_mm{target_mm:.1f}_OurS{i}.dat', 
                                            np.concatenate([Oscillator_another.tt.reshape(-1,1), 
                                                            np.full_like(Oscillator_another.tt.reshape(-1,1), target_q0),
                                                            np.full_like(Oscillator_another.tt.reshape(-1,1), target_kk),
                                                            np.full_like(Oscillator_another.tt.reshape(-1,1), target_mu),
                                                            np.full_like(Oscillator_another.tt.reshape(-1,1), target_mm),
                                                            Oscillator_another.pos_best[0,0,:].reshape(-1,1),
                                                            Oscillator_another.mom_best[0,0,:].reshape(-1,1)],axis=-1), 
                                            fmt='%.6e', header='t    q0    kk    mu    mm    q    p',  
                                            delimiter='\t')
                                
                                
                                if Hconfig['if_obey_batchsize'] == False:
                                    break
                        
                        fig = plt.figure()
                        ax1 = fig.add_subplot(2,3,1)
                        ax2 = fig.add_subplot(2,3,2)
                        ax3 = fig.add_subplot(2,3,3, projection='3d')
                        from HamiltonSys.Minaction_H_OscilatorDis_1d_ChangeParams import plot
                        plot(Oscillator_another, fig, ax1, ax2, ax3,IFplot=True,IFplot_best=True)
                        animate_curves(curves, interval=0.3, title='demonstration')
                        
                    case 'vortexsys':
                        from HamiltonSys.Minaction_H_Vortex_2d_ChangeParams import Vortex_reparam

                        save_path = f'./HamiltonSys/log/{config.data.dataset.lower()}'
                        
                        num_T = Hconfig['num_T']
                        preprocess_q0_2x = Hconfig['preprocess_q0_2x'] 
                        target_q0_2x = Hconfig['target_q0_2x']
                        preprocess_gamma2 = Hconfig['preprocess_gamma2']
                        target_gamma2 = Hconfig['target_gamma2']
                        
                        xc = (0.0 * 1.0 + target_gamma2 * target_q0_2x)/ (1.0 + target_gamma2 + 1e-20)
                        target_omega = np.abs(target_gamma2 + 1.0) / 2. / np.pi / target_q0_2x**2
                        
                        preprocess_T = 2.0 * np.pi * 1.0 / ( (preprocess_gamma2 + 1.0) / 2. / np.pi / preprocess_q0_2x**2 )
                        target_T = 2.0 * np.pi * 1.0 / ( (target_gamma2 + 1.0) / 2. / np.pi / target_q0_2x**2)
                        
                        
                        ## ML itself
                        phy_cond_concentrate = torch.tensor([target_q0_2x, target_gamma2]).unsqueeze(0).repeat(config.eval.batch_size,1).to(config.device)
                        phy_cond = [phy_cond_concentrate, None]
                        samples, n = sampling_fn(score_model, phy_cond=phy_cond)
                        samples = samples.cpu().numpy().astype(np.float64)
                        plt.plot(samples[0,0,:,0], samples[0,1,:,0],'r-'); plt.plot(samples[0,2,:,0], samples[0,3,:,0],'b-'); plt.show()
                        
                        ## H + ML
                        phy_cond_concentrate = torch.tensor([preprocess_q0_2x, preprocess_gamma2]).unsqueeze(0).repeat(config.eval.batch_size,1).to(config.device)
                        phy_cond = [phy_cond_concentrate, None]
                        samples_pre, n = sampling_fn(score_model, phy_cond=phy_cond)
                        samples_pre = samples_pre.cpu().numpy().astype(np.float64)
                        
                        
                        os.makedirs(save_path, exist_ok=True)
                        for i in range(config.eval.batch_size):
                            ## ML itself
                            x1 = samples[i,0,:,0]; q1 = x1 * np.sqrt(np.abs(1.0))
                            y1 = samples[i,1,:,0]; p1 = y1 * np.sqrt(np.abs(1.0)) * np.sign(1.0)
                            x2 = samples[i,2,:,0]; q2 = x2 * np.sqrt(np.abs(target_gamma2))
                            y2 = samples[i,3,:,0]; p2 = y2 * np.sqrt(np.abs(target_gamma2)) * np.sign(target_gamma2)
                                                       
                            np.savetxt(f'{save_path}/q20{target_q0_2x:.1f}_gamma2{target_gamma2:.1f}_T{target_T * num_T:.1f}_MlS{i}.dat', 
                                np.concatenate([np.linspace(0, target_T, config.data.height).reshape(-1,1), 
                                                np.full_like(np.linspace(0, 1, config.data.height).reshape(-1,1), target_q0_2x),
                                                np.full_like(np.linspace(0, 1, config.data.height).reshape(-1,1), target_gamma2),
                                                (x1).reshape(-1,1),
                                                (y1).reshape(-1,1),
                                                (x2).reshape(-1,1),
                                                (y2).reshape(-1,1)
                                                ],axis=-1), 
                                fmt='%.6e', header='t    q0_2x    gamma2    x1    y1    x2    y2',  
                                delimiter='\t')
                            
                            
                            x1 = samples_pre[i,0,:,0]; q1 = x1 * np.sqrt(np.abs(1.0))
                            y1 = samples_pre[i,1,:,0]; p1 = y1 * np.sqrt(np.abs(1.0)) * np.sign(1.0)
                            x2 = samples_pre[i,2,:,0]; q2 = x2 * np.sqrt(np.abs(target_gamma2))
                            y2 = samples_pre[i,3,:,0]; p2 = y2 * np.sqrt(np.abs(target_gamma2)) * np.sign(target_gamma2)
                            
                            
                            np.savetxt(f'{save_path}/q20{preprocess_q0_2x:.1f}_gamma2{preprocess_gamma2:.1f}_T{preprocess_T:.1f}_PreS{i}.dat', 
                                np.concatenate([np.linspace(0, preprocess_T,config.data.height).reshape(-1,1), 
                                                np.full_like(np.linspace(0, preprocess_T,config.data.height).reshape(-1,1), preprocess_q0_2x),
                                                np.full_like(np.linspace(0, preprocess_T,config.data.height).reshape(-1,1), preprocess_gamma2),
                                                (x1).reshape(-1,1),
                                                (y1).reshape(-1,1),
                                                (x2).reshape(-1,1),
                                                (y2).reshape(-1,1)
                                                ],axis=-1), 
                                fmt='%.6e', header='t    q0_2x    gamma2    x1    y1    x2    y2',  
                                delimiter='\t')
                                                        

                            if Hconfig['if_use_hamiltonian'] == True:
                                for i_T in range(int(num_T + 0.5)):
                                    if i_T == 0:
                                        known_pos = np.concatenate((q1.reshape(1,1,-1), q2.reshape(1,1,-1)), axis=(0))
                                        known_mom = np.concatenate((p1.reshape(1,1,-1), p2.reshape(1,1,-1)), axis=(0)) 
                                        init_pos = np.array([[0.0], [target_q0_2x * np.sqrt(np.abs(target_gamma2))   ]])
                                        init_mom = np.array([[0.0], [0.0]])
                                        guess_T = preprocess_T
                                    else:
                                        known_pos = Vortex_another.pos_best
                                        known_mom = Vortex_another.mom_best
                                        init_pos = Vortex_another.pos_best[:,:,-1]
                                        init_mom = Vortex_another.mom_best[:,:,-1]
                                        guess_T = Vortex_another.tt[-1]
                            
                                    Vortex_another, curves = Vortex_reparam(guess_T,   # just a guess
                                                                        config.data.height-1,
                                                                        [1.0,  target_gamma2],
                                                                        known_pos, # obj, dim, time
                                                                        known_mom,
                                                                        init_pos,
                                                                        init_mom,
                                                                        np.array([[0.0], [target_q0_2x * np.sqrt(np.abs(target_gamma2))]]),
                                                                        np.array([[np.inf], [np.inf]]),
                                                                        Hconfig)   
                                    # case4:      np.array([[np.inf], [np.inf]]),
                                    #                                     np.array([[-xc * np.sin(target_omega * target_T ) * np.sqrt(np.abs(1.0)) * np.sign(1.0) ], 
                                    #                                             [(target_q0_2x - xc) * np.sin(target_omega * target_T ) * np.sqrt(np.abs(target_gamma2)) * np.sign(target_gamma2) ]])
                                                            
                        
                                    np.savetxt(f'{save_path}/q20{target_q0_2x:.1f}_gamma2{target_gamma2:.1f}_T{target_T * num_T:.1f}_OurS{i}_n{i_T}.dat', 
                                                np.concatenate([Vortex_another.tt.reshape(-1,1), 
                                                                np.full_like(Vortex_another.tt.reshape(-1,1), target_q0_2x),
                                                                np.full_like(Vortex_another.tt.reshape(-1,1), target_gamma2),
                                                                (Vortex_another.pos_best[0,0,:]/1.0).reshape(-1,1),
                                                                (Vortex_another.mom_best[0,0,:]/1.0).reshape(-1,1),
                                                                (Vortex_another.pos_best[1,0,:]/np.sqrt(np.abs(target_gamma2))).reshape(-1,1),
                                                                (Vortex_another.mom_best[1,0,:]/np.sqrt(np.abs(target_gamma2))/np.sign(target_gamma2)).reshape(-1,1)
                                                                ],axis=-1), 
                                                fmt='%.6e', header='t    q0_2x    gamma2    x1    y1    x2    y2',  
                                                delimiter='\t')
                                        
                                        
                                if Hconfig['if_obey_batchsize'] == False:
                                    break
                        
                       

                                    
                        fig = plt.figure()
                        ax1 = fig.add_subplot(2,3,1)
                        ax2 = fig.add_subplot(2,3,2)
                        ax3 = fig.add_subplot(2,3,3, projection='3d')
                        ax4 = fig.add_subplot(2,3,4)
                        from HamiltonSys.Minaction_H_Vortex_2d_ChangeParams import plot
                        plot(Vortex_another, fig, ax1, ax2, ax3,IF_answer=True)
                        ax4.loglog(np.linspace(1e-5,1,len(Vortex_another.loss_reparams)), Vortex_another.loss_reparams)
                        ax4.loglog(np.linspace(1e-5,1,len(Vortex_another.lr_reparams)), Vortex_another.lr_reparams)
                        animate_curves(curves, interval=0.3, title='demonstration')
                        
                    case 'foil':
                        from HamiltonSys.Minaction_H_Foil_2d_ChangeParams import Foil_reparam
                        from HamiltonSys.ZhukovskyTransfomation import GetAFoil
                        save_path = f'./HamiltonSys/log/{config.data.dataset.lower()}'
                        
                        
                        preprocess_aoa = Hconfig['preprocess_aoa']
                        preprocess_creal = Hconfig['preprocess_creal']
                        preprocess_cimag = Hconfig['preprocess_cimag']
                        preprocess_radius = Hconfig['preprocess_radius']
                        
                        target_aoa = Hconfig['target_aoa']
                        target_creal = Hconfig['target_creal']
                        target_cimag = Hconfig['target_cimag']
                        target_radius = Hconfig['target_radius']
                        
                        ## ML itself
                        phy_cond_concentrate = torch.tensor([target_aoa, target_creal, target_cimag, target_radius]).unsqueeze(0).repeat(config.eval.batch_size,1).to(config.device)
                        _, Z_Mesh = GetAFoil(target_aoa, target_creal, target_cimag, target_radius)
                        XX, YY = Z_Mesh.real, Z_Mesh.imag
                        XY = np.vstack([XX[np.newaxis,...].astype(np.float32), YY[np.newaxis, ...].astype(np.float32)])
                        phy_cond_distirbuted = torch.from_numpy(XY).unsqueeze(0).repeat(config.eval.batch_size,1, 1, 1).to(config.device)
                        phy_cond = [phy_cond_concentrate, phy_cond_distirbuted]
                        samples, n = sampling_fn(score_model, phy_cond=phy_cond)
                        samples = samples.cpu().numpy().astype(np.float64)
                        
                        phy_cond_concentrate = torch.tensor([preprocess_aoa, preprocess_creal, preprocess_cimag, preprocess_radius]).unsqueeze(0).repeat(config.eval.batch_size,1).to(config.device)
                        _, Z_Mesh = GetAFoil(preprocess_aoa, preprocess_creal, preprocess_cimag, preprocess_radius)
                        XX_pre, YY_pre = Z_Mesh.real, Z_Mesh.imag
                        XY_pre = np.vstack([XX_pre[np.newaxis,...].astype(np.float32), YY_pre[np.newaxis, ...].astype(np.float32)])
                        phy_cond_distirbuted = torch.from_numpy(XY).unsqueeze(0).repeat(config.eval.batch_size,1, 1, 1).to(config.device)
                        phy_cond = [phy_cond_concentrate, phy_cond_distirbuted]
                        samples_pre, n = sampling_fn(score_model, phy_cond=phy_cond)
                        samples_pre = samples_pre.cpu().numpy().astype(np.float64)
                        
                        os.makedirs(save_path, exist_ok=True)
                        for i in range(config.eval.batch_size):
                            ## ML itself
                            UU, VV, PP = samples[i, 0, ...], samples[i, 1, ...], samples[i, 2, ...]
                            data_dict = {
                                'XX': XX, 'YY': YY,
                                'UU': UU, 'VV': VV, 'PP': PP, 
                                'aoa': target_aoa, 'creal': target_creal,
                                'cimag': target_cimag, 'R': target_radius}
                            filename  = f'{save_path}/Aoa{target_aoa:.2f}_cr{target_creal:.2f}_ci{target_cimag:.2f}_r{target_radius:.2f}_MlS{i}.mat'
                            sio.savemat(filename, data_dict)
                            
                            UU_pre, VV_pre, PP_pre = samples_pre[i, 0, ...], samples_pre[i, 1, ...], samples_pre[i, 2, ...]
                            data_dict = {
                                'XX': XX_pre, 'YY': YY_pre,
                                'UU': UU_pre, 'VV': VV_pre, 'PP': PP_pre, 
                                'aoa': preprocess_aoa, 'creal': preprocess_creal,
                                'cimag': preprocess_cimag, 'R': preprocess_radius}
                            filename  = f'{save_path}/Aoa{preprocess_aoa:.2f}_cr{preprocess_creal:.2f}_ci{preprocess_cimag:.2f}_r{preprocess_radius:.2f}_PreS{i}.mat'
                            sio.savemat(filename, data_dict)
                            
                            if Hconfig['if_use_hamiltonian']:
                                XX_new, YY_new, UU_new, VV_new = Foil_reparam(target_aoa, target_creal, target_cimag, target_radius,
                                                                              Hconfig, XX_pre, YY_pre, UU_pre, VV_pre, PP_pre)   
                                data_dict = {
                                    'XX': XX_new, 'YY': YY_new,
                                    'UU': UU_new, 'VV': VV_new,  
                                    'aoa': target_aoa, 'creal': target_creal,
                                    'cimag': target_cimag, 'R': target_radius}
                                filename  = f'{save_path}/Aoa{target_aoa:.2f}_cr{target_creal:.2f}_ci{target_cimag:.2f}_r{target_radius:.2f}_OurS{i}.mat'
                                sio.savemat(filename, data_dict)
                                
                                if Hconfig['if_obey_batchsize'] == False:
                                    break
                                
                            
                            
                        
                        fig = plt.figure(figsize=(15, 5))  
                        ax1 = fig.add_subplot(2,3,1)
                        ax2 = fig.add_subplot(2,3,2)
                        ax3 = fig.add_subplot(2,3,3)
                        ax4 = fig.add_subplot(2,3,4)
                        ax5 = fig.add_subplot(2,3,5)
                        ax6 = fig.add_subplot(2,3,6)

                        # 绘制等高线图并保存返回对象
                        ax1.contourf(XX, YY, UU)
                        ax2.contourf(XX, YY, VV)
                        ax3.contourf(XX, YY, PP)
                        ax4.contourf(XX_new, YY_new, UU_new)
                        ax5.contourf(XX_new, YY_new, VV_new)


                        plt.tight_layout()  # 自动调整子图间距
                        plt.show()

                                         
                    case _:
                        raise ValueError(f"Dataset {config.data.dataset} not recognized.")










# %%
