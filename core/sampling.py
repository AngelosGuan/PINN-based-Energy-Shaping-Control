import torch
import numpy as np
import random
from scipy.stats import qmc

########################################################################
# helpers

def _validate_bounds(lower_bounds, upper_bounds, input_dim):
    """Ensures bounds are valid and sets defaults if needed."""
    if lower_bounds is None:
        lower_bounds = np.zeros(input_dim)
    if upper_bounds is None:
        upper_bounds = np.ones(input_dim)

    assert len(lower_bounds) == input_dim, "lower_bounds must match input_dim"
    assert len(upper_bounds) == input_dim, "upper_bounds must match input_dim"
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

    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)

    # Identify active indices (where lower != upper)
    active_indices = np.where(lower_bounds != upper_bounds)[0]
    fixed_indices = np.where(lower_bounds == upper_bounds)[0]

    active_dim = len(active_indices)
    if active_dim == 0:
        raise ValueError("All dimensions are fixed; no active dimensions to sample.")

    sampler = qmc.LatinHypercube(d=active_dim)
    active_samples = sampler.random(n=n_samples)

    # Scale active samples
    scaled_active = qmc.scale(active_samples,
                              lower_bounds[active_indices],
                              upper_bounds[active_indices])
    # Initialize full sample array and insert fixed values
    full_samples = np.zeros((n_samples, input_dim))
    full_samples[:, active_indices] = scaled_active

    for idx in fixed_indices:
        full_samples[:, idx] = lower_bounds[idx]  # or upper_bounds[idx], same value

    return torch.tensor(full_samples, dtype=torch.float32, device=device, requires_grad=False)

########################################################################
# Sobol Sequence with automatic inactive dimension handling
def sobol_sampling(n_samples=4096, input_dim=4, device='cpu',
                   lower_bounds=None, upper_bounds=None):
    """
    Sobol Sequence Sampling with automatic detection of inactive dimensions.
    n_samples should ideally be a power of 2 for best Sobol performance.
    """
    # Validate and convert bounds to numpy arrays
    lower_bounds, upper_bounds = _validate_bounds(lower_bounds, upper_bounds, input_dim)
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)

    # Identify active indices (where lower != upper)
    active_indices = np.where(lower_bounds != upper_bounds)[0]
    fixed_indices = np.where(lower_bounds == upper_bounds)[0]

    active_dim = len(active_indices)
    if active_dim == 0:
        raise ValueError("All dimensions are fixed; no active dimensions to sample.")

    # Sample only over active dimensions
    sampler = qmc.Sobol(d=active_dim, scramble=True)
    active_samples = sampler.random(n=n_samples)

    # Scale active samples
    scaled_active = qmc.scale(active_samples,
                              lower_bounds[active_indices],
                              upper_bounds[active_indices])

    # Initialize full sample array and insert fixed values
    full_samples = np.zeros((n_samples, input_dim))
    full_samples[:, active_indices] = scaled_active

    for idx in fixed_indices:
        full_samples[:, idx] = lower_bounds[idx]  # or upper_bounds[idx], same value

    return torch.tensor(full_samples, dtype=torch.float32, device=device, requires_grad=False)

########################################################################
# uniform random
def uniform_sampling(n_samples=100, input_dim=4, device='cpu',
                     lower_bounds=None, upper_bounds=None):
    """
    Uniform random sampling with handling of fixed (lower == upper) dimensions.
    Returns: (n_samples, input_dim) torch.Tensor
    """
    lower_bounds, upper_bounds = _validate_bounds(lower_bounds, upper_bounds, input_dim)

    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)

    # Identify active indices (where lower != upper)
    active_indices = np.where(lower_bounds != upper_bounds)[0]
    fixed_indices = np.where(lower_bounds == upper_bounds)[0]

    active_dim = len(active_indices)
    if active_dim == 0:
        raise ValueError("All dimensions are fixed; no active dimensions to sample.")

    # Sample only active dimensions
    active_samples = np.random.uniform(
        low=lower_bounds[active_indices],
        high=upper_bounds[active_indices],
        size=(n_samples, active_dim)
    )

    # Initialize full sample array and insert fixed values
    full_samples = np.zeros((n_samples, input_dim))
    full_samples[:, active_indices] = active_samples

    for idx in fixed_indices:
        full_samples[:, idx] = lower_bounds[idx]  # or upper_bounds[idx], same value

    return torch.tensor(full_samples, dtype=torch.float32, device=device, requires_grad=False)
