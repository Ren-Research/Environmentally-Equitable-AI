import numpy as np
from .solve import solve_action, solve_auxiliary
from tqdm import tqdm

def expert_DMD_algorithm(price_all_loc, carbon_all_loc, water_all_loc, workload_trace,
                         mu_c, mu_w, mask_array, lr_eta, z_max_array, 
                         N=10, max_cap=1, debug=False):
    
    assert price_all_loc.shape  == carbon_all_loc.shape
    assert carbon_all_loc.shape == water_all_loc.shape
    
    num_ins = price_all_loc.shape[1]
    assert workload_trace.shape  == (N, num_ins)
    
    matrix_G = np.zeros([N, N*N])
    for i in range(N):
        matrix_G[i, i*N:(i+1)*N] = 1

    vector_zt      = 0*np.zeros(2*N)
    vector_gamma_t = np.zeros(2*N)

    action_list = []
    if debug: gamma_list = []
    
    for t in tqdm(range(num_ins)):
        matrix_ct = np.diag(carbon_all_loc[:,t])
        matrix_wt = np.diag(water_all_loc[:,t])

        matrix_bt = np.zeros([2*N, N*N])
        matrix_bt[:N, :]    = mu_c * (matrix_G @ matrix_ct)
        matrix_bt[N:2*N, :] = mu_w * (matrix_G @ matrix_wt)

        vector_price = price_all_loc[:, t]
        calibrated_price = vector_price + vector_gamma_t @ matrix_bt

        action_mask, opt_cost = solve_action(calibrated_price.reshape(N,N), workload_trace[:, t], 
                         max_cap, mask_array, num_nodes = 10)

        action_mask = action_mask.reshape(-1)    

        vector_z, _ = solve_auxiliary(vector_gamma_t, z_max_array)
        g_t =  matrix_bt@ action_mask - vector_z
        
        vector_gamma_t = vector_gamma_t + lr_eta*g_t

        action_list   += [action_mask]
        if debug: gamma_list    += [vector_gamma_t]
    
    action_list = np.array(action_list).T
    
    if debug:
        gamma_list  = np.array(gamma_list).T
        return action_list, gamma_list
    else:
        return action_list
    