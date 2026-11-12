
def get_config():
    config = {}
    
    
    ## 
    # because ML often fails to extropolate, so use prepocess_cond to reach a physical solution
    # then optimize it to final solution we want.
    # we use q_end as an end point
    config['preprocess_q0'] = 1.0;
    config['target_q0'] = 1.0
    config['preprocess_kk'] = 0.5
    config['target_kk'] = 0.1
    config['preprocess_mu'] = 1.5
    config['target_mu'] = 0.1
    config['preprocess_mm'] = 1.0
    config['target_mm'] = 0.1
    config['preprocess_T'] = 50.0 
    config['target_T'] = 50.0
    config['IF_2split'] = False
    
    config['if_use_hamiltonian'] = True
    config['if_obey_batchsize'] = True
    
    
    ##
    config['lambda_p'] = 1.0
    config['lambda_q'] = 1.0
    
    config['IF_autoscale_loss'] = False
    config['lr'] = [4e-1]#[1e-2]
    config['lr_updatet'] = 5000
    config['lr_decay'] = 0.9
    config['num_iterations'] = 50000    
    config['lambda_smooth_q_1st'] = [0.0]
    config['lambda_smooth_p_1st'] = [8e-2, 8e-3, 8e-4]
    config['perturbation_type'] = 'sincos'
    return config