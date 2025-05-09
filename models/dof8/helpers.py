import torch
import numpy as np
import random
from models.dof8.dynamics import lf, slope, LOWER_BOUNDS, UPPER_BOUNDS
from core.utils import divide_samples_around_pivot
from configs.config_dof8 import EPSILON

def sampling_with_contact_condition(num_samples, device, sampling_func):
    '''
    Sample each contact condition with almost equal distribution

    Returns: 
        torch.Tensor: Final shuffled sample tensor of shape [num_samples, 8].
    '''

    # divide
    n_cond1, n_cond2, n_cond3 = [num_samples // 3 + (i < num_samples % 3) for i in range(3)]

    # condition1: x0 = 0, x1 = 0
    n_part1, n_part2 = divide_samples_around_pivot(num_samples=n_cond1, low=LOWER_BOUNDS[2], high=UPPER_BOUNDS[3], pivot=slope)
    
    # sample first half where x2 in [low, slope-EPSILON]
    part1_lower = LOWER_BOUNDS[2:].copy()
    part1_upper = UPPER_BOUNDS[2:].copy()
    part1_upper[0] = slope - EPSILON
    cond1_part1 = sampling_func(n_samples=n_part1, input_dim=14, device=device,
                 lower_bounds=part1_lower, upper_bounds=part1_upper)
    # add x0 = 0, x1 = 0 to all cond1 result
    zeros = torch.zeros(n_part1, 2, device=device, dtype=cond1_part1.dtype)
    cond1_part1 = torch.cat([zeros, cond1_part1], dim=1) # final shape [n, 8]

    # sample second half where x2 in [slope+EPSILON, high]
    part2_lower = LOWER_BOUNDS[2:].copy()
    part2_upper = UPPER_BOUNDS[2:].copy()
    part2_lower[0] = slope + EPSILON
    cond1_part2 = sampling_func(n_samples=n_part2, input_dim=14, device=device,
             lower_bounds=part2_lower, upper_bounds=part2_upper)
    # add x0 = 0, x1 = 0 to all cond1 result
    zeros = torch.zeros(n_part2, 2, device=device, dtype=cond1_part2.dtype)
    cond1_part2 = torch.cat([zeros, cond1_part2], dim=1) # final shape [n, 8]



    # condition2: x0 = 0, x1 = 0, x2 = slope
    cond2_lower = LOWER_BOUNDS[3:].copy()
    cond2_upper = UPPER_BOUNDS[3:].copy()
    cond2 = sampling_func(n_samples=n_cond2, input_dim=13, device=device,
             lower_bounds=cond2_lower, upper_bounds=cond2_upper)
    # add x0 = 0, x1 = 0, x2 = slope to all cond2 result
    head = torch.tensor([0.0, 0.0, slope], device=device, dtype=cond2.dtype).expand(n_cond2, 3)
    cond2 = torch.cat([head, cond2], dim=1) # final shape [n, 8]




    # condition3: x0 = lf*cos(x2), x1 = lf*sin(x2)
    cond3_lower = LOWER_BOUNDS[2:].copy()
    cond3_upper = UPPER_BOUNDS[2:].copy()
    cond3 = sampling_func(n_samples=n_cond3, input_dim=14, device=device,
             lower_bounds=cond3_lower, upper_bounds=cond3_upper)
    # add x0 = lf*cos(x2), x1 = lf*sin(x2) to all cond3 result
    x0 = lf*torch.cos(cond3[:,0]).unsqueeze(1)   # shape [n, 1]
    x1 = lf*torch.sin(cond3[:,0]).unsqueeze(1)   # shape [n, 1]
    cond3 = torch.cat([x0,x1,cond3], dim=1) # final shape [n, 8]

    # add all samples together to create num_samples length sample
    assert all(x.shape[1] == 8 for x in [cond1_part1, cond1_part2, cond2, cond3])
    final_result = torch.cat([cond1_part1, cond1_part2, cond2, cond3], dim=0)

    # shuffle, comment out if using dataloader
    perm = torch.randperm(final_result.shape[0])
    final_result = final_result[perm]

    return final_result




