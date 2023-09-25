import numpy as np 
import cvxpy as cp


def offline_solver(price_all_loc, water_all_loc, carbon_all_loc,
                   workload_trace, mask_array, num_ins, 
                   l_0 = 1, l_1 = 100, l_2 = 100, max_cap = 1, 
                   verbose=True, f_type = "MAX"):
    '''
    Solve the offline problem
    Args:
        price_all_loc   : Energy price of all locations [10*10, num_ins]
        water_all_loc   : Water WUE of all locations [10*10, num_ins]
        carbon_all_loc  : Carbon consumption of all places [10*10, num_ins]
        workload_trace  : Workload trace
        mask_array      : Array with size of [10, 10, num_ins]
        num_ins         : Number of timesteps to solve
        l_1             : Water coeficiency
        l_2             : Carbon coeficiency
    Return:
        optimal_cost:
        action_mask:
    '''
    x            = cp.Variable([10*10, num_ins])
    x_masked     = cp.multiply(x, mask_array)
    
    energy_cost  = cp.sum(cp.multiply(x_masked, price_all_loc))

    water_cost   = cp.sum(cp.multiply(x_masked, water_all_loc), axis=1)
    water_cost   = cp.reshape(water_cost, [10,10], order="C")
    water_cost   = cp.sum(water_cost, axis=1)

    carbon_cost  = cp.sum(cp.multiply(x_masked, carbon_all_loc), axis=1)
    carbon_cost  = cp.reshape(carbon_cost, [10,10], order="C")
    carbon_cost  = cp.sum(carbon_cost, axis=1)

    
    if f_type == "MAX":
        # Max Price
        water_cost   = cp.norm(water_cost, p="inf")
        carbon_cost  = cp.norm(carbon_cost, p="inf")
    elif f_type == "AVG":
        # Average Price
        water_cost   = cp.sum(water_cost, axis=0)/10
        carbon_cost  = cp.sum(carbon_cost, axis=0)/10
    else:
        raise NotImplementedError
        
    total_cost   = l_0*energy_cost + l_1*water_cost + l_2*carbon_cost

    constraints = []
    for i in range(num_ins):
        for j in range(10):
            c_i = [
                cp.sum(x_masked[10*j:10*j+10, i]) <= max_cap,
                cp.sum(x_masked[j::10, i]) ==  workload_trace[j,i]
            ]
            constraints += c_i

    for i in range(num_ins):
        for j in range(100):
            constraints += [x[j,i] >= 0]


    objective = cp.Minimize(total_cost)
    prob      = cp.Problem(objective, constraints)
    if verbose: print("Start Solving...")
    prob.solve(verbose = verbose)
    
    optimal_cost   = prob.value
    action_optimal = x.value
    action_mask    = np.multiply(action_optimal, mask_array)
    
    return optimal_cost, action_mask


def solve_action(virtual_price, workload, max_cap, mask_array, num_nodes = 10):
    '''
    Solve the reward function problem arg min x∈Xt {⟨pt, xt⟩ + γt · bt · xt} 
    Args:
        virtual_price   : Energy price of all locations [num_ins, num_ins]
        workload        : Workload trace
        max_cap         : The maximum capability of datacenter
        mask_array      : Array with size of [num_ins, num_ins]
        num_nodes       : Number of datacenter
    Return:
        action_mask     : Action based on mask
        optimal_cost    : Optimal cost 
    '''
    assert virtual_price.shape == (num_nodes, num_nodes)
    
    x            = cp.Variable([num_nodes, num_nodes])
    x_masked     = cp.multiply(x, mask_array)
    
    constraints = []
    for i in range(10):
        c_i = [
            cp.sum(x_masked[i, :]) <= max_cap,
            cp.sum(x_masked[:, i]) == workload[i]
        ]
        
        constraints += c_i
    constraints += [x_masked >= 0]
    
    total_cost   = cp.sum(cp.multiply(x_masked, virtual_price))
    objective    = cp.Minimize(total_cost)
    prob         = cp.Problem(objective, constraints)
    prob.solve(verbose = False)
    
    optimal_cost   = prob.value
    action_optimal = x.value
    action_mask    = np.multiply(action_optimal, mask_array)
    
    return action_mask, optimal_cost

def solve_auxiliary(vector_gamma, z_max_array, num_nodes = 10):
    '''
    Solve the reward problem
    Args:
        vector_gamma    : Vector of gamma_t
        z_max_array     : the maximum value of z
        num_nodes       : Number of datacenter
    Return:
        z_optimal       : Optimal auxiliary action
        optimal_cost    : Optimal cost 
    '''
    
    assert vector_gamma.shape == (num_nodes*2,)
    
    z = cp.Variable([2*num_nodes])
    
    constraints  = []
    constraints += [z >= 0]
    constraints += [z <= z_max_array]
    
    total_cost    = cp.atoms.norm(z[:num_nodes], "inf") + cp.atoms.norm(z[num_nodes:], "inf") \
                     - vector_gamma @ z 
    objective     = cp.Minimize(total_cost)
    prob          = cp.Problem(objective, constraints)
    
    prob.solve(verbose = False)
    
    optimal_cost  = prob.value
    z_optimal     = z.value
    
    
    return z_optimal, optimal_cost



