import numpy as numpy
import torch

# state variable ranges
LOWER_BOUNDS = [-6.283, -1.0, -6.0, -6.0]
UPPER_BOUNDS = [ 6.283, 7.0,  6.0,  6.0]

M = 0.6
m = 0.39
l = 0.36
g = 9.81
a = g/l
b = 1/l 
c = (M+m)/(m*l*l)
d = m*l*l 
k = 0.007


def calculate_M(x):
    return torch.stack([
        torch.stack([torch.tensor(1.0, device = x.device), b*torch.cos(x[0])]),
        torch.stack([b*torch.cos(x[0]), torch.tensor(c, device = x.device)])])

def calculate_V(x):
    return torch.tensor(a*d*torch.cos(x[0]), device = x.device)

def calculate_dVdq(x):
    return torch.stack(
        [torch.stack([-a*d*torch.sin(x[0])]),
        torch.tensor([0.0], device = x.device)
        ])

def calculate_J(x):
    return torch.stack([
        torch.stack([(k*k*torch.pow(b, 3))/12*torch.pow(torch.cos(x[0]), 4)*torch.sin(x[0])]),
        torch.stack([-(k*k*b*b)/12*torch.pow(torch.cos(x[0]), 3)*torch.sin(x[0])])])
