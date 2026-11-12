def get_config():
    config = {}
    
    
    ## 
    # because ML often fails to extropolate, so use prepocess_cond to reach a physical solution
    # then optimize it to final solution we want.
    # we use q_end as an end point
    config['preprocess_aoa'] = 10.0
    config['preprocess_creal'] = -0.15
    config['preprocess_cimag'] = 0.0
    config['preprocess_radius'] = 1.2
    
    config['target_aoa'] = 30.0
    config['target_creal'] = -0.15
    config['target_cimag'] = 0.1
    config['target_radius'] = 1.2
    
    config['if_use_hamiltonian'] = True
    config['if_obey_batchsize'] = False
    
    
    ##
    config['num_iterations'] = 5000
    config['lr'] = 8e-6 #[1e-2]
    config['lr_updatet'] = 50
    config['lr_decay'] = 0.7
    
    return config
                        
                        