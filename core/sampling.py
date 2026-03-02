import torch
import numpy as np
import random
from scipy.stats import qmc
import configs.config_dof4 as config

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
                 lower_bounds=None, upper_bounds=None, seed=config.SEED):
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

    # needs explicit seeding to set deterministic
    sampler = qmc.LatinHypercube(d=active_dim, seed=seed)
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
# def sobol_sampling(n_samples=4096, input_dim=4, device='cpu',
#                    lower_bounds=None, upper_bounds=None, seed=config.SEED):
#     """
#     Sobol Sequence Sampling with automatic detection of inactive dimensions.
#     n_samples should ideally be a power of 2 for best Sobol performance.
#     """
#     # Validate and convert bounds to numpy arrays
#     lower_bounds, upper_bounds = _validate_bounds(lower_bounds, upper_bounds, input_dim)
#     lower_bounds = np.array(lower_bounds)
#     upper_bounds = np.array(upper_bounds)

#     # Identify active indices (where lower != upper)
#     active_indices = np.where(lower_bounds != upper_bounds)[0]
#     fixed_indices = np.where(lower_bounds == upper_bounds)[0]

#     active_dim = len(active_indices)
#     if active_dim == 0:
#         raise ValueError("All dimensions are fixed; no active dimensions to sample.")

#     # Sample only over active dimensions
#     sampler = qmc.Sobol(d=active_dim, scramble=True, seed=seed)
#     active_samples = sampler.random(n=n_samples)

#     # Scale active samples
#     scaled_active = qmc.scale(active_samples,
#                               lower_bounds[active_indices],
#                               upper_bounds[active_indices])

#     # Initialize full sample array and insert fixed values
#     full_samples = np.zeros((n_samples, input_dim))
#     full_samples[:, active_indices] = scaled_active

#     for idx in fixed_indices:
#         full_samples[:, idx] = lower_bounds[idx]  # or upper_bounds[idx], same value

#     return torch.tensor(full_samples, dtype=torch.float32, device=device, requires_grad=False)

def sobol_sampling(n_samples=4096, input_dim=4, device='cpu',
                   lower_bounds=None, upper_bounds=None, seed=config.SEED):
    """
    Sobol Sequence Sampling with automatic detection of inactive dimensions.
    n_samples should ideally be a power of 2 for best Sobol performance.
    """

    device = torch.device(device)

    # Validate bounds
    lower_bounds, upper_bounds = _validate_bounds(lower_bounds, upper_bounds, input_dim)

    lb = torch.as_tensor(lower_bounds, dtype=torch.float32, device=device)
    ub = torch.as_tensor(upper_bounds, dtype=torch.float32, device=device)

    # Active / fixed dimensions
    active_mask = lb != ub
    active_indices = torch.where(active_mask)[0]
    fixed_indices  = torch.where(~active_mask)[0]

    active_dim = active_indices.numel()
    if active_dim == 0:
        raise ValueError("All dimensions are fixed; no active dimensions to sample.")

    # ---- Sobol engine (Torch native) ----
    engine = torch.quasirandom.SobolEngine(
        dimension=active_dim,
        scramble=True,
        seed=seed
    )

    # Generate directly on target device
    u = engine.draw(n_samples).to(device)   # shape: (n_samples, active_dim)

    # Scale from [0,1] → [lb, ub]
    lb_active = lb[active_indices]
    ub_active = ub[active_indices]
    scaled_active = lb_active + u * (ub_active - lb_active)

    # Assemble full tensor
    full = lb.expand(n_samples, input_dim).clone()
    full[:, active_indices] = scaled_active

    return full  

########################################################################
# uniform random
def uniform_sampling(n_samples=100, input_dim=4, device='cpu',
                     lower_bounds=None, upper_bounds=None, seed=config.SEED):
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
    # needs to fix local generator with seed
    rng = np.random.default_rng(seed)
    active_samples = rng.uniform(
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


########################################################################
# adaptive sampling (RAD-D algorithm, partial replacement determined by replace_frac)
# following DeepXDE's implementation
# https://arxiv.org/pdf/2104.12325
# https://arxiv.org/pdf/2207.10289
class AdaptiveSamplerRAD:
    def __init__(
        self,
        residual_fn,                 # callable(model, X)->[N] pointwise residual
        proposal_sampler,            # callable(n)->[n, dim] valid domain samples
        replace_frac=0.25,           # fraction of interior to replace
        pool_mult=8,                 # pool size = replace_count * pool_mult
        k=1.0,                       # RAD exponent (paper's k ≥ 0)
        c=1.0,                       # RAD mixing coeff (paper's c ≥ 0)
        batch_size=32768,                 # score candidates in chunks
        eps=1e-12
    ):
        self.residual_fn = residual_fn
        self.proposal_sampler = proposal_sampler
        self.replace_frac = replace_frac
        self.pool_mult = pool_mult
        self.k = float(k)
        self.c = float(c)
        self.batch_size = int(batch_size)
        self.eps = eps

    @torch.no_grad()
    def compute_pmf(self, model, X_pool):

        # compute residuals in batches defined by self.batch_size 
        N = X_pool.shape[0]
        rks = []

        for i in range(0, N, self.batch_size):
            X_batch = X_pool[i:i+self.batch_size]
            residual_batch = self.residual_fn(model, X_batch).clamp_min_(0.0)
            if abs(self.k)<1e-12:
                r_k = torch.ones_like(residual_batch)
            else:
                r_k = (residual_batch + self.eps).pow(self.k)
            rks.append(r_k)

        rks = torch.cat(rks, dim=0)
        Ek = rks.mean()                             # MC approx of E[ε^k] over pool

        scores = rks + self.c * Ek                  # paper's ε^k + c E[ε^k]

        if torch.all(scores <= 0) or not torch.isfinite(scores).all():
            scores = torch.ones_like(scores)
        
        Z = scores.sum()
        if Z==0.0:
            raise ValueError("Dividing by zero in AdaptiveSamplerRAD.compute_pmf")
        return scores / Z 

    @torch.no_grad()
    def step(self, model, X):
        # assume X is already on device
        N, d = X.shape

        # how many to replace + pool size
        n_replace = max(1, int(self.replace_frac * N))    # self.replace_frac * N
        n_pool = max(n_replace + 1, n_replace * self.pool_mult)   # n_replace * self.pool_mult

        # draw candidates and pmf
        X_pool = self.proposal_sampler(n_pool)  # [n_pool, d]
        pmf = self.compute_pmf(model, X_pool)

        # sample without replacement and splice
        # set seed
        rng = torch.Generator(device=X.device).manual_seed(config.SEED)
        chosen = torch.multinomial(pmf, num_samples=n_replace, replacement=False, generator=rng)
        X_new = X_pool[chosen]
        victims = torch.randperm(N, device=X.device)[:n_replace]
        
        X_updated = X.clone()
        X_updated[victims] = X_new
        return X_updated

########################################################################
# helper function to translate sampling function to the proposal sampler function requred for RAD
def make_proposal(sampler_func, lower_bounds, upper_bounds, device, input_dim):
    def proposal_sampler(n):
        '''
        input: n - number of samples to generate
        '''
        return sampler_func(
            n_samples=n,
            input_dim=input_dim,
            device=device,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
    return proposal_sampler