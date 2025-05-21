import torch
import scipy.io
import math
import random
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np


########################################################################
# === Math utilities ===
##############################
# pseudo inverse with damping for stability
# allow batch
def damped_pseudo_inverse(A, lambda_reg=1e-4):
    if A.ndim == 2:
        # Unbatched case: A is [d, d]
        A_T = A.transpose(0, 1)
        A_reg = A_T @ A + lambda_reg * torch.eye(A.shape[1], device=A.device)
        return torch.linalg.solve(A_reg, A_T)

    elif A.ndim == 3:
        # Batched case: A is [n, d, d]
        A_T = A.transpose(-1, -2)  # [n, d, d]
        d = A.shape[-1]
        I = torch.eye(d, device=A.device).expand(A.shape[0], d, d)  # [n, d, d]
        A_reg = A_T @ A + lambda_reg * I
        return torch.linalg.solve(A_reg, A_T)  # [n, d, d]

    else:
        raise ValueError("Input tensor A must be 2D or 3D (batched)")

##############################
# loss function that punish outliers quadratically
# bound absolute value of out
# allow batch
def bounded_quad_loss(out, bound):
    # ensure bound is a scalar tensor on the same device
    bound = torch.as_tensor(bound, device=out.device, dtype=out.dtype)

    # Compute softplus of excess over bound
    excess = torch.nn.functional.softplus(torch.abs(out) - bound)

    if out.ndim == 1:
        # Unbatched: out is [d], return scalar
        return torch.mean(excess**2)
    elif out.ndim == 2:
        # Batched: out is [n, d], return [n]
        return torch.mean(excess**2, dim=1)
    else:
        raise ValueError("Input tensor 'out' must be 1D or 2D")
########################################################################
# fix random seed for reproductivity      
def divide_samples_around_pivot(num_samples, low, high, pivot):
    """
    Divides `num_samples` into two integers (num_region1, num_region2)
    such that:
    
      num_region1 + num_region2 == num_samples
      (num_region1 / num_region2) ≈ (pivot - low) / (high - pivot)
      
    This ensures the number of samples allocated to each side of the `pivot`
    is proportional to the size of the two sub-ranges:
      - Region 1: [low, pivot)
      - Region 2: (pivot, high]
      
    Parameters:
        num_samples (int): total number of samples to divide.
        low (float): lower bound of the total range.
        high (float): upper bound of the total range.
        pivot (float): dividing point between the two regions.
    
    Returns:
        num_region1 (int): number of samples in [low, pivot).
        num_region2 (int): number of samples in (pivot, high].
    """
    range1 = pivot - low
    range2 = high - pivot
    total_range = range1 + range2

    # Calculate proportional sample counts
    num_region1 = round(num_samples * (range1 / total_range))
    num_region2 = num_samples - num_region1  # ensure exact sum

    return num_region1, num_region2


########################################################################
# === File utilities ===
######################################################
# loading .mat data
def load_data(data_path):
    '''
    load data
    input: path to input data
    output: tensor of all datapoints:(number_of_data, input state dimension)
    '''
    mat = scipy.io.loadmat(data_path)
    X = torch.tensor(mat['xs'], requires_grad=False).float()
    return X


########################################################################
# === Machine Learning utilities ===
########################################################################
# fix random seed for reproductivity
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

########################################################################
# compute gradient norm
def compute_gradient_norm(model):
    total_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5



######################################################
# residual block
# for MLP models
class ResidualLinearNormBlock(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_repeats, output_dim, residual_interval=2):
        super().__init__()
        self.num_repeats = num_repeats
        self.residual_interval = residual_interval

        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Tanh()
        )

        self.hidden_layers = torch.nn.ModuleList()
        for _ in range(num_repeats - 1):
            self.hidden_layers.append(torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.Tanh()
            ))

        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, output_dim),
            torch.nn.Tanh()
        )

    def forward(self, x):
        x = self.input_layer(x)
        residual = x

        for i, layer in enumerate(self.hidden_layers):
            out = layer(x)
            is_last_layer = (i == len(self.hidden_layers) - 1)
            is_residual_point = ((i + 1) % self.residual_interval == 0)

            if is_residual_point or is_last_layer:
                x = out + residual
                residual = x
            else:
                x = out

        x = self.output_layer(x)
        return x
        
######################################################
# linear block
# for MLP models
def make_linear_norm_block(input_dim, hidden_dim, num_repeats, output_dim):
    """
    Creates a sequential neural network block consisting of repeated 
    Linear -> LayerNorm -> Tanh layers.

    Parameters:
        input_dim (int): Size of the input features.
        hidden_dim (int): Size of the hidden layer units.
        num_repeats (int): Number of repeated hidden layers.
        output_dim (int): Size of the final output layer.

    Returns:
        torch.nn.Sequential: A PyTorch Sequential module containing the full block.
    """
    layers = []
    layers.append(torch.nn.Linear(input_dim, hidden_dim))
    layers.append(torch.nn.LayerNorm(hidden_dim))
    layers.append(torch.nn.Tanh())

    for _ in range(num_repeats - 1):
        layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        layers.append(torch.nn.LayerNorm(hidden_dim))
        layers.append(torch.nn.Tanh())

    layers.append(torch.nn.Linear(hidden_dim, output_dim))
    layers.append(torch.nn.Tanh())

    return torch.nn.Sequential(*layers)
######################################################
'''
Cosine Warmup for training
# Adapted from:
# https://github.com/hiromis/CosineAnnealingWarmupRestarts
# Author: Hiromi Suenaga
# License: MIT
'''

class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr