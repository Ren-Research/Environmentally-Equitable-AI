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


def evaluate_single(action_mask, price_all_loc):
    '''
    Evaluate the price of single resource
    '''
    price_tensor  = price_all_loc.reshape([10,10,-1])
    price_res     = np.multiply(price_tensor, action_mask)
    price_res     = price_res.sum(axis=(1,2))
    
    return price_res

def evaluate_total(action_mask, price_all_loc, carbon_all_loc, 
                   water_all_loc, l_1, l_2, 
                   verbose = True, l_0 = 1):
    '''
    Evaluate the total cost of the policy
    '''    
    price_res   = evaluate_single(action_mask, price_all_loc)
    carbon_res  = evaluate_single(action_mask, carbon_all_loc)
    water_res   = evaluate_single(action_mask, water_all_loc)
    
    
    price_cost  = l_0*np.sum(price_res)
    water_cost  = l_1*np.linalg.norm(water_res, ord=np.inf)
    carbon_cost = l_2*np.linalg.norm(carbon_res, ord=np.inf)
    
    total_cost  = price_cost + carbon_cost + water_cost
    
    if verbose:
        print("Electric price  :  {:.3f}".format(price_cost))
        print("Total water     :  {:.3f}".format(water_cost))
        print("Total carbon    :  {:.3f}".format(carbon_cost))
        print("-----")
        print("Overall Cost    :  {:.3f}".format(total_cost))
    
    return total_cost


