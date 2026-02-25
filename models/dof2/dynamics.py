import numpy as numpy
import torch

# state variable ranges
LOWER_BOUNDS = [-1.57, -1.0, -6.0, -6.0]
UPPER_BOUNDS = [ 1.57, 7.0,  6.0,  6.0]

M = 0.6
m = 0.39
l = 0.36
g = 9.81
a = g/l
b = 1/l 
c = (M+m)/(m*l*l)
d = m*l*l 
k = 0.007


# def calculate_M(x):
#     return torch.stack([
#         torch.stack([torch.tensor(1.0, device = x.device, dtype=x.dtype), b*torch.cos(x[0])]),
#         torch.stack([b*torch.cos(x[0]), torch.tensor(c, device = x.device, dtype=x.dtype)])])

# def calculate_V(x):
#     return a*d*torch.cos(x[0])

# def calculate_dVdq(x):
#     return torch.stack(
#         [torch.stack([-a*d*torch.sin(x[0])]),
#         torch.tensor([0.0], device = x.device, dtype=x.dtype)
#         ])

# def calculate_J(x):
#     return torch.stack([
#         torch.stack([(k*k*b**3)/12*(torch.cos(x[0])**4)*torch.sin(x[0])]),
#         torch.stack([-(k*k*b*b)/12*(torch.cos(x[0])**3)*torch.sin(x[0])])])
def calculate_M(x):
    # x: shape (4,) or compatible
    one = x.new_tensor(1.0)
    c_  = x.new_tensor(c)
    cosq = torch.cos(x[0])
    off = b * cosq
    return torch.stack([
        torch.stack([one, off], dim=0),
        torch.stack([off, c_], dim=0)
    ], dim=0)

def calculate_V(x):
    # returns scalar tensor on x.device
    return x.new_tensor(a * d) * torch.cos(x[0])

def calculate_dVdq(x):
    # return shape (2,) (then caller can .view(2,1))
    sinq = torch.sin(x[0])
    return torch.stack([
        x.new_tensor(-a * d) * sinq,
        x.new_tensor(0.0)
    ], dim=0)

def calculate_J(x):
    # return shape (2,) (then caller can .view(2,1))
    cosq = torch.cos(x[0])
    sinq = torch.sin(x[0])
    coef1 = (k * k * b**3) / 12.0
    coef2 = (k * k * b*b) / 12.0
    j1 = x.new_tensor(coef1) * (cosq**4) * sinq
    j2 = x.new_tensor(-coef2) * (cosq**3) * sinq
    return torch.stack([j1, j2], dim=0)