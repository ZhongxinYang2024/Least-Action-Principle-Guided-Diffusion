def get_config():
    config = {}
    
    
    ## 
    # because ML often fails to extropolate, so use prepocess_cond to reach a physical solution
    # then optimize it to final solution we want.
    # we use q_end as an end point
    config['preprocess_aoa'] = 0.0
    config['preprocess_creal'] = -0.1724137921
    config['preprocess_cimag'] = 0.0
    config['preprocess_radius'] = 1.2268965517
    config['target_aoa'] = 0.0
    config['target_creal'] = -0.1724137921
    config['target_cimag'] = 0.0
    config['target_radius'] = 1.2268965517
    
    config['if_use_hamiltonian'] = True
    config['if_obey_batchsize'] = False
    
    
    ##
    config['num_iterations'] = 3000
    config['lr'] = 4e-3#[1e-2]
    config['lr_updatet'] = 50
    config['lr_decay'] = 0.9
    return config
                        
                        