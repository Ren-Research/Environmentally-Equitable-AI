import numpy as np

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