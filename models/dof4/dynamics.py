import torch

# dynamic parameters
massRatio = 3.5/5
lenRatio = 0.5

CONTROL_BOUND = 40

# state variable ranges
# swing ankle, stance knee, hip, swing knee
LOWER_BOUNDS = [-0.262, 0.0, -0.8, 0.0, -2.5, -5, -5, -2.0]
UPPER_BOUNDS = [0.349, 0.70, 0.8, 1.484, 2.5, 3.5, 5, 7.0]

M = 13.51
Mp = 31.73 # hip mass
Mf = 1
g = 9.81
L = 0.856
l1 = L*(1-lenRatio)
l2 = L*(lenRatio)
la = 0.07                             
I1x = 0.0369
I2x = 0.1995                    # from "Neuromechanics of Human Movement" by Roger Enoka
M1 = M*(1-massRatio)
M2 = M*(massRatio)
I1y, I1z, I2y, I2z, Ipx, Ipy, Ipz = 0, 0, 0, 0, 0, 0, 0
lf = 0.2
Ms=M1
Mt=M2
Mfs=Mf
Mss=M1
Mts=M2

def calculate_Mmtx(x, device, I1z, I2z, Ipz, Mp, Ms, Mt, l1, l2): 
    M = torch.tensor([[2*I1x+2*I2x+Ipx+l1**2*(Mp+(3/2)*Ms+2*Mt)+(1/2)*l2**2*(2*Mp+4*Ms+3*Mt)+l1*l2*(2*(Mp+Ms)+3*Mt)*torch.cos(x[1])+(-1)*l2**2*(2*Ms+Mt)*torch.cos(x[2])+(-1)*l1*(l2*(2*Ms+Mt)*torch.cos(x[1]+x[2])+l2*Ms*((-1)*torch.cos(x[3])+torch.cos(x[2]+x[3]))+l1*Ms*torch.cos(x[1]+x[2]+x[3])), (1/4)*(4*I1x+8*I2x+4*Ipx+l1**2*Ms+l2**2*(4*Mp+8*Ms+6*Mt)+2*l1*l2*(2*(Mp+Ms)+3*Mt)*torch.cos(x[1])+(-4)*l2**2*(2*Ms+Mt)*torch.cos(x[2])+(-2)*l1*(l2*(2*Ms+Mt)*torch.cos(x[1]+x[2])+2*l2*Ms*((-1)*torch.cos(x[3])+torch.cos(x[2]+x[3]))+l1*Ms*torch.cos(x[1]+x[2]+x[3]))), (1/4)*(4*I1x+4*I2x+l1**2*Ms+l2**2*(4*Ms+Mt)+(-2)*l2**2*(2*Ms+Mt)*torch.cos(x[2])+(-2)*l1*(l2*(2*Ms+Mt)*torch.cos(x[1]+x[2])+l2*Ms*((-2)*torch.cos(x[3])+torch.cos(x[2]+x[3]))+l1*Ms*torch.cos(x[1]+x[2]+x[3]))), I1x+(1/4)*l1*Ms*(l1+2*l2*torch.cos(x[3])+(-2)*l2*torch.cos(x[2]+x[3])+(-2)*l1*torch.cos(x[1]+x[2]+x[3]))], [(1/4)*(4*I1x+8*I2x+4*Ipx+l1**2*Ms+l2**2*(4*Mp+8*Ms+6*Mt)+2*l1*l2*(2*(Mp+Ms)+3*Mt)*torch.cos(x[1])+(-4)*l2**2*(2*Ms+Mt)*torch.cos(x[2])+(-2)*l1*(l2*(2*Ms+Mt)*torch.cos(x[1]+x[2])+2*l2*Ms*((-1)*torch.cos(x[3])+torch.cos(x[2]+x[3]))+l1*Ms*torch.cos(x[1]+x[2]+x[3]))), I1x+2*I2x+Ipx+(1/4)*(l1**2*Ms+l2**2*(4*Mp+8*Ms+6*Mt))+(-1)*l2**2*(2*Ms+Mt)*torch.cos(x[2])+l1*l2*Ms*(torch.cos(x[3])+(-1)*torch.cos(x[2]+x[3])), (1/4)*(4*I1x+4*I2x+l1**2*Ms+l2**2*(4*Ms+Mt)+(-2)*l2*(l2*(2*Ms+Mt)*torch.cos(x[2])+l1*Ms*((-2)*torch.cos(x[3])+torch.cos(x[2]+x[3])))), I1x+(1/4)*l1*Ms*(l1+2*l2*torch.cos(x[3])+(-2)*l2*torch.cos(x[2]+x[3]))], [(1/4)*(4*I1x+4*I2x+l1**2*Ms+l2**2*(4*Ms+Mt)+(-2)*l2**2*(2*Ms+Mt)*torch.cos(x[2])+(-2)*l1*(l2*(2*Ms+Mt)*torch.cos(x[1]+x[2])+l2*Ms*((-2)*torch.cos(x[3])+torch.cos(x[2]+x[3]))+l1*Ms*torch.cos(x[1]+x[2]+x[3]))), (1/4)*(4*I1x+4*I2x+l1**2*Ms+l2**2*(4*Ms+Mt)+(-2)*l2*(l2*(2*Ms+Mt)*torch.cos(x[2])+l1*Ms*((-2)*torch.cos(x[3])+torch.cos(x[2]+x[3])))), I1x+I2x+(1/4)*(l1**2*Ms+l2**2*(4*Ms+Mt))+l1*l2*Ms*torch.cos(x[3]), I1x+(1/4)*l1*Ms*(l1+2*l2*torch.cos(x[3]))], [I1x+(1/4)*l1*Ms*(l1+2*l2*torch.cos(x[3])+(-2)*l2*torch.cos(x[2]+x[3])+(-2)*l1*torch.cos(x[1]+x[2]+x[3])), I1x+(1/4)*l1*Ms*(l1+2*l2*torch.cos(x[3])+(-2)*l2*torch.cos(x[2]+x[3])), I1x+(1/4)*l1*Ms*(l1+2*l2*torch.cos(x[3])), I1x+(1/4)*l1**2*Ms]])
    return M.to(x.device)

def calculate_Nvect(x, device, Mp, Ms, Mt, l1, l2, g):
    N = torch.tensor([[(1/2)*g*((-1)*l1*(2*Mp+3*Ms+4*Mt)*torch.sin(x[0])+(-1)*l2*(2*(Mp+Ms)+3*Mt)*torch.sin(x[0]+x[1])+l2*(2*Ms+Mt)*torch.sin(x[0]+x[1]+x[2])+l1*Ms*torch.sin(x[0]+x[1]+x[2]+x[3]))], [(1/2)*g*((-1)*l2*(2*(Mp+Ms)+3*Mt)*torch.sin(x[0]+x[1])+l2*(2*Ms+Mt)*torch.sin(x[0]+x[1]+x[2])+l1*Ms*torch.sin(x[0]+x[1]+x[2]+x[3]))], [(1/2)*g*(l2*(2*Ms+Mt)*torch.sin(x[0]+x[1]+x[2])+l1*Ms*torch.sin(x[0]+x[1]+x[2]+x[3]))], [(1/2)*g*l1*Ms*torch.sin(x[0]+x[1]+x[2]+x[3])]])
    return N.to(x.device)

def calculate_Cmtx(x, device, I1z, I2z, Ipz, Mp, Ms, Mt, l1, l2):
    # TODO: change to stack to preserve gradient flow and multiply qdot if using this over calculate_Cqdot (e-8 error)
    C = torch.tensor([[(1/2)*((-1)*l1*l2*(2*Mp+2*Ms+3*Mt)*torch.sin(x[1])+l2**2*(2*Ms+Mt)*torch.sin(x[2])+2*l1*l2*(2*Ms+Mt)*torch.sin(x[1]+x[2])+l1*Ms*((-1)*l2*torch.sin(x[3])+2*l2*torch.sin(x[2]+x[3])+3*l1*torch.sin(x[1]+x[2]+x[3]))), (1/4)*(2*l2**2*(2*Ms+Mt)*torch.sin(x[2])+(-1)*l1*l2*(2*Mp+2*Ms+3*Mt)*torch.sin(x[1])*(1+2*x[4]+x[5])+l1*l2*(2*Ms+Mt)*torch.sin(x[1]+x[2])*(2+2*x[4]+x[5]+x[6])+l1*Ms*((-2)*l2*torch.sin(x[3])+4*l2*torch.sin(x[2]+x[3])+l1*torch.sin(x[1]+x[2]+x[3])*(3+2*x[4]+x[5]+x[6]+x[7]))), (1/4)*(l1*l2*(2*Ms+Mt)*torch.sin(x[1]+x[2])*(2+2*x[4]+x[5]+x[6])+l2**2*(2*Ms+Mt)*torch.sin(x[2])*(1+2*x[4]+2*x[5]+x[6])+l1*Ms*((-2)*l2*torch.sin(x[3])+l1*torch.sin(x[1]+x[2]+x[3])*(3+2*x[4]+x[5]+x[6]+x[7])+l2*torch.sin(x[2]+x[3])*(2+2*x[4]+2*x[5]+x[6]+x[7]))), (1/4)*l1*Ms*(l1*torch.sin(x[1]+x[2]+x[3])*(3+2*x[4]+x[5]+x[6]+x[7])+l2*torch.sin(x[2]+x[3])*(2+2*x[4]+2*x[5]+x[6]+x[7])+(-1)*l2*torch.sin(x[3])*(1+2*x[4]+2*x[5]+2*x[6]+x[7]))], [(1/4)*(2*l2**2*(2*Ms+Mt)*torch.sin(x[2])+l1*l2*(2*Mp+2*Ms+3*Mt)*torch.sin(x[1])*((-1)+2*x[4]+x[5])+(-1)*l1*(l2*(2*Ms+Mt)*torch.sin(x[1]+x[2])*((-2)+2*x[4]+x[5]+x[6])+Ms*(2*l2*torch.sin(x[3])+(-4)*l2*torch.sin(x[2]+x[3])+l1*torch.sin(x[1]+x[2]+x[3])*((-3)+2*x[4]+x[5]+x[6]+x[7])))), (1/2)*l2*(l2*(2*Ms+Mt)*torch.sin(x[2])+l1*Ms*((-1)*torch.sin(x[3])+2*torch.sin(x[2]+x[3]))), (1/4)*l2*(l2*(2*Ms+Mt)*torch.sin(x[2])*(1+2*x[4]+2*x[5]+x[6])+l1*Ms*((-2)*torch.sin(x[3])+torch.sin(x[2]+x[3])*(2+2*x[4]+2*x[5]+x[6]+x[7]))), (1/4)*l1*l2*Ms*(torch.sin(x[2]+x[3])*(2+2*x[4]+2*x[5]+x[6]+x[7])+(-1)*torch.sin(x[3])*(1+2*x[4]+2*x[5]+2*x[6]+x[7]))], [(1/4)*((-1)*l2**2*(2*Ms+Mt)*torch.sin(x[2])*((-1)+2*x[4]+2*x[5]+x[6])+(-1)*l1*(l2*(2*Ms+Mt)*torch.sin(x[1]+x[2])*((-2)+2*x[4]+x[5]+x[6])+Ms*(2*l2*torch.sin(x[3])+l1*torch.sin(x[1]+x[2]+x[3])*((-3)+2*x[4]+x[5]+x[6]+x[7])+l2*torch.sin(x[2]+x[3])*((-2)+2*x[4]+2*x[5]+x[6]+x[7])))), (-1/4)*l2*(l2*(2*Ms+Mt)*torch.sin(x[2])*((-1)+2*x[4]+2*x[5]+x[6])+l1*Ms*(2*torch.sin(x[3])+torch.sin(x[2]+x[3])*((-2)+2*x[4]+2*x[5]+x[6]+x[7]))), (-1/2)*l1*l2*Ms*torch.sin(x[3]), (-1/4)*l1*l2*Ms*torch.sin(x[3])*(1+2*x[4]+2*x[5]+2*x[6]+x[7])], [(-1/4)*l1*Ms*(l1*torch.sin(x[1]+x[2]+x[3])*((-3)+2*x[4]+x[5]+x[6]+x[7])+l2*torch.sin(x[2]+x[3])*((-2)+2*x[4]+2*x[5]+x[6]+x[7])+(-1)*l2*torch.sin(x[3])*((-1)+2*x[4]+2*x[5]+2*x[6]+x[7])), (-1/4)*l1*l2*Ms*(torch.sin(x[2]+x[3])*((-2)+2*x[4]+2*x[5]+x[6]+x[7])+(-1)*torch.sin(x[3])*((-1)+2*x[4]+2*x[5]+2*x[6]+x[7])), (1/4)*l1*l2*Ms*torch.sin(x[3])*((-1)+2*x[4]+2*x[5]+2*x[6]+x[7]), 0]])
    return C.to(x.device)

# batch safe version
def calculate_Cqdot(func_M, X):
    is_batched = X.ndim == 2  # (B, 8) or (8,)

    def Cqdot_vmap(x):
        def func_M1(x):
            M = func_M(x)
            q_dot = x[4:].view(-1,1)
            return M @ q_dot

        def func_M2(x):
            M = func_M(x)
            q_dot = x[4:].view(-1,1)
            return torch.transpose(q_dot,0,1) @ M @ q_dot

        q_dot = x[4:].view(-1,1)
        jacobianM1 = torch.func.jacrev(lambda x: func_M1(x))(x).reshape(4,8)[:,:4]
        jacobianM2 = torch.func.jacrev(lambda x: func_M2(x))(x).reshape(1,8)[:,:4]

        return jacobianM1 @ q_dot - 0.5 * torch.transpose(jacobianM2,0,1)

    if is_batched:
        return torch.vmap(Cqdot_vmap)(X)

    else:
        return Cqdot_vmap(X)



# # Coriolis Matrix
# def calculate_Cqdot(func_M, x):
#     def func_M1(x):
#         M = func_M(x)
#         q_dot = x[4:].view(-1,1)
#         return M @ q_dot

#     def func_M2(x):
#         M = func_M(x)
#         q_dot = x[4:].view(-1,1)
#         return torch.transpose(q_dot,0,1) @ M @ q_dot

#     q_dot = x[4:].view(-1,1)
#     jacobianM1 = torch.func.jacrev(lambda x: func_M1(x))(x).reshape(4,8)[:,:4]
#     jacobianM2 = torch.func.jacrev(lambda x: func_M2(x))(x).reshape(1,8)[:,:4]

#     return jacobianM1 @ q_dot - 0.5 * torch.transpose(jacobianM2,0,1)