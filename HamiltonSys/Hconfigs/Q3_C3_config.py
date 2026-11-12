
def get_config():
    config = {}
    
    
    ## 
    # because ML often fails to extropolate, so use prepocess_cond to reach a physical solution
    # then optimize it to final solution we want.
    # we use q_end as an end point
    config['preprocess_q0'] = 2.0;
    config['target_q0'] = 10.0
    config['preprocess_kk'] = 1.0
    config['target_kk'] = 1.0
    config['preprocess_mu'] = 2.0
    config['target_mu'] = 2.0
    config['preprocess_mm'] = 1.0
    config['target_mm'] = 0.5
    config['preprocess_T'] = 50.0 
    config['target_T'] = 50.0
    
    config['if_use_hamiltonian'] = True
    config['if_obey_batchsize'] = False
    config['IF_2split'] = False
    
    ##
    config['lr'] = [2e-2]
    config['lr_updatet'] = 5000
    config['lr_decay'] = 0.7
    config['num_iterations'] = 250000    
    config['lambda_smooth_q_1st'] = [0.0]
    config['lambda_smooth_p_1st'] = [1e-2]
    config['perturbation_type'] = 'sincos'
    
    return config