import torch
import numpy as np
import random
from scipy.stats import qmc

########################################################################
# helpers

def _scale_samples(samples, lower_bounds, upper_bounds):
    """Scales unit hypercube samples to the given bounds."""
    return qmc.scale(samples, lower_bounds, upper_bounds)

def _validate_bounds(lower_bounds, upper_bounds, input_dim):
    """Ensures bounds are valid and sets defaults if needed."""
    if lower_bounds is None:
        lower_bounds = np.zeros(input_dim)
    if upper_bounds is None:
        upper_bounds = np.ones(input_dim)

    assert lower_bounds.shape == (input_dim,), "lower_bounds must match input_dim"
    assert upper_bounds.shape == (input_dim,), "upper_bounds must match input_dim"
    return lower_bounds, upper_bounds


########################################################################
# Latin Hypercube Sampling
def lhs_sampling(n_samples=5000, input_dim=4, device='cpu',
                 lower_bounds=None, upper_bounds=None):
    """
    Latin Hypercube Sampling.
    Returns: (n_samples, input_dim) torch.Tensor
    """
    lower_bounds, upper_bounds = _validate_bounds(lower_bounds, upper_bounds, input_dim)
    sampler = qmc.LatinHypercube(d=input_dim)
    samples = sampler.random(n=n_samples)
    scaled = _scale_samples(samples, lower_bounds, upper_bounds)
    return torch.tensor(scaled, dtype=torch.float32, device=device, requires_grad=False)

########################################################################
# Sobel Sequence
def sobol_sampling(n_samples=4096, input_dim=4, device='cpu',
                   lower_bounds=None, upper_bounds=None):
    """
    Sobol Sequence Sampling. n_samples should be power of 2 for best results.
    """
    lower_bounds, upper_bounds = _validate_bounds(lower_bounds, upper_bounds, input_dim)
    sampler = qmc.Sobol(d=input_dim, scramble=True)
    samples = sampler.random(n=n_samples)
    scaled = _scale_samples(samples, lower_bounds, upper_bounds)
    return torch.tensor(scaled, dtype=torch.float32, device=device, requires_grad=False)

########################################################################
# uniform random
def uniform_sampling(n_samples=100, input_dim=4, device='cpu',
                     lower_bounds=None, upper_bounds=None):
    """
    Uniform random sampling.
    """
    lower_bounds, upper_bounds = _validate_bounds(lower_bounds, upper_bounds, input_dim)
    samples = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(n_samples, input_dim))
    return torch.tensor(samples, dtype=torch.float32, device=device, requires_grad=False)
