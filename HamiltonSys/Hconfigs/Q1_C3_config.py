
def get_config():
    config = {}
    
    ## the normailization g during training
    config['scalar_g'] = 15.0
    config['scalar_T'] = 2.0
    config['scalar_m'] = 1.0

    
    ## 
    # because ML often fails to extropolate, so use prepocess_cond to reach a physical solution
    # then optimize it to final solution we want.
    # we use q_end as an end point
    config['preprocess_g'] = 15.0
    config['target_g'] = 30.0
    
    config['standard_T'] = 2.0  # the standard T to normalize time during training.
    config['preprocess_T'] = 2.0
    config['target_T'] = 4.0
    
    
    config['if_use_hamiltonian'] = True
    config['if_obey_batchsize'] = True
    
    
    ##
    config['lr'] = [4e-1]
    config['lr_updatet'] = 2500
    config['lr_decay'] = 0.7
    config['num_iterations'] = 40000    
    config['period_updatet'] = 50
    config['decay_updatet']  = 0.995
    config['lambda_smooth_q_1st'] = [0.0]
    config['lambda_smooth_p_1st'] = [5e-2]
    config['perturbation_type'] = 'sincos'
    
    return config