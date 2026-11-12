def get_config():
    config  = {}
    
    ## 
    # because ML often fails to extropolate, so use prepocess_cond to reach a physical solution
    # then optimize it to final solution we want.
    config['num_T'] =  2.0
    config['preprocess_q0_2x']  = 1.0
    config['target_q0_2x'] = 1.0
    config['preprocess_gamma2'] = 1.0
    config['target_gamma2'] = 1.0
    
    
   
    config['if_use_hamiltonian'] = True
    config['if_obey_batchsize'] = True
    
    ##
    config['lr'] = [4e-3]
    config['lr_updatet'] = 500
    config['lr_decay'] = 0.7
    config['num_iterations'] = 2000
    config['period_updatet'] = 25
    config['decay_updatet'] = 0.995
    config['lambda_smooth_q_1st'] = [0.0]
    config['lambda_smooth_p_1st'] = [5e-4,5e-5]
    config['perturbation_type'] = 'sincos'
    return config
    
    