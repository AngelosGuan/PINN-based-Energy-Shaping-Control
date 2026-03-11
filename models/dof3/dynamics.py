import torch

# dynamic parameters
CONTROL_BOUND = 40

# state variable ranges
# swing ankle, stance knee, hip, swing knee
LOWER_BOUNDS = [-0.4, -1.062, -1.062, -2.5, -5.0, -6.0]
UPPER_BOUNDS = [0.4, 1.149, 2.633, 2.5, 5.0, 8.0]

def calculate_Mmtx(X):
    single = (X.ndim == 1)
    if single:
        X = X.unsqueeze(0)

    device = X.device
    dtype = X.dtype

    theta1 = X[:, 0]
    theta2 = X[:, 1]
    theta3 = X[:, 2]

    m1 = torch.tensor(5, device=device, dtype=dtype)
    m2 = torch.tensor(3.5, device=device, dtype=dtype)
    m3 = torch.tensor(1.5, device=device, dtype=dtype)
    mh = torch.tensor(10, device=device, dtype=dtype)

    a1 = torch.tensor(0.53, device=device, dtype=dtype)
    b1 = torch.tensor(0.47, device=device, dtype=dtype)
    l1 = a1 + b1

    a2 = torch.tensor(0.15, device=device, dtype=dtype)
    b2 = torch.tensor(0.35, device=device, dtype=dtype)
    l2 = a2 + b2

    a3 = torch.tensor(0.25, device=device, dtype=dtype)
    b3 = torch.tensor(0.25, device=device, dtype=dtype)

    m11 = m1*a1**2 + (mh + m2 + m3)*l1**2
    m12 = -(m2*b2*l1 + m3*l1*l2)*torch.cos(theta1 - theta2)
    m13 = -m3*b3*l1*torch.cos(theta1 - theta3)

    m22 = m2*b2**2 + m3*l2**2
    m23 = m3*b3*l2*torch.cos(theta2 - theta3)

    m33 = m3*b3**2

    zeros = torch.zeros_like(theta1)

    row1 = torch.stack([m11.expand_as(theta1), m12, m13], dim=-1)
    row2 = torch.stack([m12, m22.expand_as(theta1), m23], dim=-1)
    row3 = torch.stack([m13, m23, m33.expand_as(theta1)], dim=-1)

    M = torch.stack([row1, row2, row3], dim=1)

    if single:
        M = M.squeeze(0)

    return M


def calculate_Nvect(X):
    single = (X.ndim == 1)
    if single:
        X = X.unsqueeze(0)

    device = X.device
    dtype = X.dtype

    theta1 = X[:, 0]
    theta2 = X[:, 1]
    theta3 = X[:, 2]

    m1 = torch.tensor(5, device=device, dtype=dtype)
    m2 = torch.tensor(3.5, device=device, dtype=dtype)
    m3 = torch.tensor(1.5, device=device, dtype=dtype)
    mh = torch.tensor(10, device=device, dtype=dtype)

    a1 = torch.tensor(0.53, device=device, dtype=dtype)
    b1 = torch.tensor(0.47, device=device, dtype=dtype)
    l1 = a1 + b1

    a2 = torch.tensor(0.15, device=device, dtype=dtype)
    b2 = torch.tensor(0.35, device=device, dtype=dtype)
    l2 = a2 + b2

    a3 = torch.tensor(0.25, device=device, dtype=dtype)
    b3 = torch.tensor(0.25, device=device, dtype=dtype)

    g = torch.tensor(9.8, device=device, dtype=dtype)

    g1 = -(m1*a1 + m2*l1 + m3*l1 + mh*l1)*g*torch.sin(theta1)
    g2 = (m2*b2 + m3*l2)*g*torch.sin(theta2)
    g3 = m3*b3*g*torch.sin(theta3)

    G = torch.stack([g1, g2, g3], dim=-1).unsqueeze(-1)

    if single:
        G = G.squeeze(0) #3,1

    return G

# def calculate_Cmtx(x, device, I1z, I2z, Ipz, Mp, Ms, Mt, l1, l2):
#     # TODO: change to stack to preserve gradient flow and multiply qdot if using this over calculate_Cqdot (e-8 error)
#     # not using currently
#     C = torch.tensor([[(1/2)*((-1)*l1*l2*(2*Mp+2*Ms+3*Mt)*torch.sin(x[1])+l2**2*(2*Ms+Mt)*torch.sin(x[2])+2*l1*l2*(2*Ms+Mt)*torch.sin(x[1]+x[2])+l1*Ms*((-1)*l2*torch.sin(x[3])+2*l2*torch.sin(x[2]+x[3])+3*l1*torch.sin(x[1]+x[2]+x[3]))), (1/4)*(2*l2**2*(2*Ms+Mt)*torch.sin(x[2])+(-1)*l1*l2*(2*Mp+2*Ms+3*Mt)*torch.sin(x[1])*(1+2*x[4]+x[5])+l1*l2*(2*Ms+Mt)*torch.sin(x[1]+x[2])*(2+2*x[4]+x[5]+x[6])+l1*Ms*((-2)*l2*torch.sin(x[3])+4*l2*torch.sin(x[2]+x[3])+l1*torch.sin(x[1]+x[2]+x[3])*(3+2*x[4]+x[5]+x[6]+x[7]))), (1/4)*(l1*l2*(2*Ms+Mt)*torch.sin(x[1]+x[2])*(2+2*x[4]+x[5]+x[6])+l2**2*(2*Ms+Mt)*torch.sin(x[2])*(1+2*x[4]+2*x[5]+x[6])+l1*Ms*((-2)*l2*torch.sin(x[3])+l1*torch.sin(x[1]+x[2]+x[3])*(3+2*x[4]+x[5]+x[6]+x[7])+l2*torch.sin(x[2]+x[3])*(2+2*x[4]+2*x[5]+x[6]+x[7]))), (1/4)*l1*Ms*(l1*torch.sin(x[1]+x[2]+x[3])*(3+2*x[4]+x[5]+x[6]+x[7])+l2*torch.sin(x[2]+x[3])*(2+2*x[4]+2*x[5]+x[6]+x[7])+(-1)*l2*torch.sin(x[3])*(1+2*x[4]+2*x[5]+2*x[6]+x[7]))], [(1/4)*(2*l2**2*(2*Ms+Mt)*torch.sin(x[2])+l1*l2*(2*Mp+2*Ms+3*Mt)*torch.sin(x[1])*((-1)+2*x[4]+x[5])+(-1)*l1*(l2*(2*Ms+Mt)*torch.sin(x[1]+x[2])*((-2)+2*x[4]+x[5]+x[6])+Ms*(2*l2*torch.sin(x[3])+(-4)*l2*torch.sin(x[2]+x[3])+l1*torch.sin(x[1]+x[2]+x[3])*((-3)+2*x[4]+x[5]+x[6]+x[7])))), (1/2)*l2*(l2*(2*Ms+Mt)*torch.sin(x[2])+l1*Ms*((-1)*torch.sin(x[3])+2*torch.sin(x[2]+x[3]))), (1/4)*l2*(l2*(2*Ms+Mt)*torch.sin(x[2])*(1+2*x[4]+2*x[5]+x[6])+l1*Ms*((-2)*torch.sin(x[3])+torch.sin(x[2]+x[3])*(2+2*x[4]+2*x[5]+x[6]+x[7]))), (1/4)*l1*l2*Ms*(torch.sin(x[2]+x[3])*(2+2*x[4]+2*x[5]+x[6]+x[7])+(-1)*torch.sin(x[3])*(1+2*x[4]+2*x[5]+2*x[6]+x[7]))], [(1/4)*((-1)*l2**2*(2*Ms+Mt)*torch.sin(x[2])*((-1)+2*x[4]+2*x[5]+x[6])+(-1)*l1*(l2*(2*Ms+Mt)*torch.sin(x[1]+x[2])*((-2)+2*x[4]+x[5]+x[6])+Ms*(2*l2*torch.sin(x[3])+l1*torch.sin(x[1]+x[2]+x[3])*((-3)+2*x[4]+x[5]+x[6]+x[7])+l2*torch.sin(x[2]+x[3])*((-2)+2*x[4]+2*x[5]+x[6]+x[7])))), (-1/4)*l2*(l2*(2*Ms+Mt)*torch.sin(x[2])*((-1)+2*x[4]+2*x[5]+x[6])+l1*Ms*(2*torch.sin(x[3])+torch.sin(x[2]+x[3])*((-2)+2*x[4]+2*x[5]+x[6]+x[7]))), (-1/2)*l1*l2*Ms*torch.sin(x[3]), (-1/4)*l1*l2*Ms*torch.sin(x[3])*(1+2*x[4]+2*x[5]+2*x[6]+x[7])], [(-1/4)*l1*Ms*(l1*torch.sin(x[1]+x[2]+x[3])*((-3)+2*x[4]+x[5]+x[6]+x[7])+l2*torch.sin(x[2]+x[3])*((-2)+2*x[4]+2*x[5]+x[6]+x[7])+(-1)*l2*torch.sin(x[3])*((-1)+2*x[4]+2*x[5]+2*x[6]+x[7])), (-1/4)*l1*l2*Ms*(torch.sin(x[2]+x[3])*((-2)+2*x[4]+2*x[5]+x[6]+x[7])+(-1)*torch.sin(x[3])*((-1)+2*x[4]+2*x[5]+2*x[6]+x[7])), (1/4)*l1*l2*Ms*torch.sin(x[3])*((-1)+2*x[4]+2*x[5]+2*x[6]+x[7]), 0]])
#     return C.to(x.device)

# batch safe version
def calculate_Cqdot(func_M, X):
    is_batched = X.ndim == 2  # (B, 6) or (6,)

    def Cqdot_vmap(x):
        def func_M1(x):
            M = func_M(x)
            q_dot = x[3:].view(-1,1)
            return M @ q_dot

        def func_M2(x):
            M = func_M(x)
            q_dot = x[3:].view(-1,1)
            return torch.transpose(q_dot,0,1) @ M @ q_dot

        q_dot = x[3:].view(-1,1)
        jacobianM1 = torch.func.jacrev(lambda x: func_M1(x))(x).reshape(3,6)[:,:3]
        jacobianM2 = torch.func.jacrev(lambda x: func_M2(x))(x).reshape(1,6)[:,:3]

        return jacobianM1 @ q_dot - 0.5 * torch.transpose(jacobianM2,0,1)

    if is_batched:
        return torch.vmap(Cqdot_vmap)(X)

    else:
        return Cqdot_vmap(X)

