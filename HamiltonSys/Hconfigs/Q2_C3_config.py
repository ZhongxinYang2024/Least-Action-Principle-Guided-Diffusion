
def get_config():
    config = {}
    
    
    ## 
    # because ML often fails to extropolate, so use prepocess_cond to reach a physical solution
    # then optimize it to final solution we want.
    # we use q_end as an end point
    config['num_T'] = 2.0 # num of T in training set
    config['preprocess_q0'] =  -1.0
    config['target_q0'] = -2.0
    config['preprocess_p0'] = 1.0
    config['target_p0'] = 1.0
    config['preprocess_kk'] = 1.0 
    config['target_kk'] = 1.0
    
    
    
    
    config['preprocess_mm'] = 1.0 
    config['target_mm'] = 2.0
    
    config['if_use_hamiltonian'] = True
    config['if_obey_batchsize'] = True
    
    
    ##
    config['lr'] = [1e-2]
    config['lr_updatet'] = 500
    config['lr_decay'] = 0.7
    config['num_iterations'] = 20000    
    config['period_updatet'] = 50
    config['decay_updatet']  = 0.995
    config['lambda_smooth_q_1st'] = [5e-2, 8e-3, 2e-3]
    config['lambda_smooth_p_1st'] = [5e-2, 8e-3, 2e-3]
    config['perturbation_type'] = 'sinsin'
    
    return config