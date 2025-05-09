import  torch

# used in M or condition
Mp = 31.73    # same as Mh in paper
Mf = 1.0
g = 9.81
l1 = 0.428  # same as ls
l2 = 0.428  # same as lt
la = 0.07
lf = 0.2
I1x = 0.0369      # same as Is
I2x = 0.1995      # same as It
Ipx = 0
slope = 0.095
M1 = (13.51)*(1-(3.5/5))
M2 = (13.51)*(3.5/5)

# unused but may change for different 8DOF model
I1y, I1z, I2y, I2z, Ipy, Ipz = 0, 0, 0, 0, 0, 0

# state variable ranges
LOWER_BOUNDS = [0.0, 0.0, -0.3, 0.0, -0.2, -0.5, -0.1, -0.26, 0.0, 0.0, -20.0, -5.0, -2.5, -5.0, -0.5, -15.0]
UPPER_BOUNDS = [0.2, 0.02, 0.1, 0.3, 0.05, 0.5, 0.3, 0.2, 0.0, 0.0, 5.0, 25.0, 2.5, 1.0, 10.0, 5.0]

# control torque
CONTROL_BOUND = 40.0
##################################
# contact condition (accept both batch and unbatch)
def determine_phase_masks(x, slope, lf, EPSILON, device):
    """
    Determines phase masks for different contact conditions 
    (heel contact, flat foot, toe contact) for both single input or batched input.

    Parameters:
        x (torch.Tensor): Input state tensor of shape [dim] or [batch_size, dim].
        slope (float): Slope value for comparison.
        lf (float): Foot length or reference length.
        EPSILON (float): Tolerance threshold for equality checks.
        device (torch.device): Device to place the output masks on.

    Returns:
        tuple of torch.Tensor:
            (phase1_mask, phase2_mask, phase3_mask), each as float32 tensors.
            Shape [batch_size] if batched, or scalar if unbatched.
    """
    # if single data, turn into batch size 1
    unbatched = False
    if x.ndim == 1:
        x = x.unsqueeze(0)  # [dim] -> [1, dim]
        unbatched = True

    # flat foot
    phase2_cond = (
        (torch.abs(x[:, 0]) < EPSILON) &
        (torch.abs(x[:, 1]) < EPSILON) &
        (torch.abs(x[:, 2] - slope) < EPSILON)
    )

    # heel contact
    phase1_cond = (
        (torch.abs(x[:, 0]) < EPSILON) &
        (torch.abs(x[:, 1]) < EPSILON)
    )

    # toe contact
    phase3_cond = (
        (torch.abs(x[:, 0] - lf * torch.cos(x[:, 2])) < EPSILON) &
        (torch.abs(x[:, 1] - lf * torch.sin(x[:, 2])) < EPSILON)
    )

    phase2_mask = phase2_cond & ~phase3_cond
    phase1_mask = phase1_cond & (~phase2_cond)
    phase3_mask = phase3_cond & ~phase1_cond

    # convert to float
    phase1_mask = phase1_mask.float().to(device)
    phase2_mask = phase2_mask.float().to(device)
    phase3_mask = phase3_mask.float().to(device)

    # if unbatched, return scalar masks
    if unbatched:
        phase1_mask = phase1_mask.squeeze(0)
        phase2_mask = phase2_mask.squeeze(0)
        phase3_mask = phase3_mask.squeeze(0)

    return phase1_mask, phase2_mask, phase3_mask

##################################
# Coriolis Matrix
def calculate_Cqdot(func_M, x):
    def func_M1(x):
        M = func_M(x)
        q_dot = x[8:].view(-1,1)
        return M @ q_dot

    def func_M2(x):
        M = func_M(x)
        q_dot = x[8:].view(-1,1)
        return torch.transpose(q_dot,0,1) @ M @ q_dot

    q_dot = x[8:].view(-1,1)
    jacobianM1 = torch.func.jacrev(lambda x: func_M1(x))(x).reshape(8,16)[:,:8]
    jacobianM2 = torch.func.jacrev(lambda x: func_M2(x))(x).reshape(1,16)[:,:8]

    return jacobianM1 @ q_dot - 0.5 * torch.transpose(jacobianM2,0,1)

##################################
# Mass Matrix
def calculate_Mmtx(x, device, Mp, Mf, g, l1, l2, la, lf, I1x, I2x, Ipx, slope, M1, M2):
    """
    Translated from the given MATLAB code into Python / PyTorch.
    x is assumed to be at least 16 elements long, so x[i-1] is valid up to i=16.
    """

    Mmtx = torch.stack([

        # -----------------------
        # Row 1 (8 columns)
        # -----------------------
        torch.stack([
            torch.tensor(2*(M1+M2+Mf) + Mp,device=device),  # column 1
            torch.tensor(0.0,device=device),                  # column 2
            0.5*(
                (-1)*la*(4*M1+4*M2+3*Mf+2*Mp)*torch.cos(x[2]) +
                (-1)*l1*(3*M1+2*(2*M2+Mf+Mp))*torch.cos(x[2] + x[3]) +
                (-2)*l2*M1*torch.cos(x[2] + x[3] + x[4]) +
                (-3)*l2*M2*torch.cos(x[2] + x[3] + x[4]) +
                (-2)*l2*Mf*torch.cos(x[2] + x[3] + x[4]) +
                (-2)*l2*Mp*torch.cos(x[2] + x[3] + x[4]) +
                2*l2*M1*torch.cos(x[2] + x[3] + x[4] + x[5]) +
                l2*M2*torch.cos(x[2] + x[3] + x[4] + x[5]) +
                2*l2*Mf*torch.cos(x[2] + x[3] + x[4] + x[5]) +
                l1*M1*torch.cos(x[2] + x[3] + x[4] + x[5] + x[6]) +
                2*l1*Mf*torch.cos(x[2] + x[3] + x[4] + x[5] + x[6]) +
                la*Mf*torch.cos(x[2] + x[3] + x[4] + x[5] + x[6] + x[7])
            ),  # column 3
            0.5*(
                (-1)*l1*(3*M1 + 2*(2*M2+Mf+Mp))*torch.cos(x[2] + x[3]) +
                (-1)*l2*(2*M1+3*M2+2*(Mf+Mp))*torch.cos(x[2] + x[3] + x[4]) +
                2*l2*M1*torch.cos(x[2] + x[3] + x[4] + x[5]) +
                l2*M2*torch.cos(x[2] + x[3] + x[4] + x[5]) +
                2*l2*Mf*torch.cos(x[2] + x[3] + x[4] + x[5]) +
                l1*M1*torch.cos(x[2] + x[3] + x[4] + x[5] + x[6]) +
                2*l1*Mf*torch.cos(x[2] + x[3] + x[4] + x[5] + x[6]) +
                la*Mf*torch.cos(x[2] + x[3] + x[4] + x[5] + x[6] + x[7])
            ),  # column 4
            0.5*(
                (-1)*l2*(2*M1+3*M2+2*(Mf+Mp))*torch.cos(x[2] + x[3] + x[4]) +
                l2*(2*M1+M2+2*Mf)*torch.cos(x[2] + x[3] + x[4] + x[5]) +
                l1*(M1+2*Mf)*torch.cos(x[2] + x[3] + x[4] + x[5] + x[6]) +
                la*Mf*torch.cos(x[2] + x[3] + x[4] + x[5] + x[6] + x[7])
            ),  # column 5
            0.5*(
                l2*(2*M1+M2+2*Mf)*torch.cos(x[2] + x[3] + x[4] + x[5]) +
                l1*(M1+2*Mf)*torch.cos(x[2] + x[3] + x[4] + x[5] + x[6]) +
                la*Mf*torch.cos(x[2] + x[3] + x[4] + x[5] + x[6] + x[7])
            ),  # column 6
            0.5*(
                l1*(M1+2*Mf)*torch.cos(x[2] + x[3] + x[4] + x[5] + x[6]) +
                la*Mf*torch.cos(x[2] + x[3] + x[4] + x[5] + x[6] + x[7])
            ),  # column 7
            0.5*la*Mf*torch.cos(x[2] + x[3] + x[4] + x[5] + x[6] + x[7])  # column 8
        ]),

        # -----------------------
        # Row 2 (8 columns)
        # -----------------------
        torch.stack([
            torch.tensor(0,device=device),                        # column 1
            torch.tensor(2*(M1+M2+Mf) + Mp, device=device),        # column 2
            0.5*(
                (-1)*la*(4*M1+4*M2+3*Mf+2*Mp)*torch.sin(x[2]) +
                (-1)*l1*(3*M1+2*(2*M2+Mf+Mp))*torch.sin(x[2] + x[3]) +
                (-2)*l2*M1*torch.sin(x[2] + x[3] + x[4]) +
                (-3)*l2*M2*torch.sin(x[2] + x[3] + x[4]) +
                (-2)*l2*Mf*torch.sin(x[2] + x[3] + x[4]) +
                (-2)*l2*Mp*torch.sin(x[2] + x[3] + x[4]) +
                2*l2*M1*torch.sin(x[2] + x[3] + x[4] + x[5]) +
                l2*M2*torch.sin(x[2] + x[3] + x[4] + x[5]) +
                2*l2*Mf*torch.sin(x[2] + x[3] + x[4] + x[5]) +
                l1*M1*torch.sin(x[2] + x[3] + x[4] + x[5] + x[6]) +
                2*l1*Mf*torch.sin(x[2] + x[3] + x[4] + x[5] + x[6]) +
                la*Mf*torch.sin(x[2] + x[3] + x[4] + x[5] + x[6] + x[7])
            ), # column 3
            0.5*(
                (-1)*l1*(3*M1 + 2*(2*M2+Mf+Mp))*torch.sin(x[2] + x[3]) +
                (-1)*l2*(2*M1+3*M2+2*(Mf+Mp))*torch.sin(x[2] + x[3] + x[4]) +
                2*l2*M1*torch.sin(x[2] + x[3] + x[4] + x[5]) +
                l2*M2*torch.sin(x[2] + x[3] + x[4] + x[5]) +
                2*l2*Mf*torch.sin(x[2] + x[3] + x[4] + x[5]) +
                l1*M1*torch.sin(x[2] + x[3] + x[4] + x[5] + x[6]) +
                2*l1*Mf*torch.sin(x[2] + x[3] + x[4] + x[5] + x[6]) +
                la*Mf*torch.sin(x[2] + x[3] + x[4] + x[5] + x[6] + x[7])
            ), # column 4
            0.5*(
                (-1)*l2*(2*M1+3*M2+2*(Mf+Mp))*torch.sin(x[2] + x[3] + x[4]) +
                l2*(2*M1+M2+2*Mf)*torch.sin(x[2] + x[3] + x[4] + x[5]) +
                l1*(M1+2*Mf)*torch.sin(x[2] + x[3] + x[4] + x[5] + x[6]) +
                la*Mf*torch.sin(x[2] + x[3] + x[4] + x[5] + x[6] + x[7])
            ), # column 5
            0.5*(
                l2*(2*M1+M2+2*Mf)*torch.sin(x[2] + x[3] + x[4] + x[5]) +
                l1*(M1+2*Mf)*torch.sin(x[2] + x[3] + x[4] + x[5] + x[6]) +
                la*Mf*torch.sin(x[2] + x[3] + x[4] + x[5] + x[6] + x[7])
            ), # column 6
            0.5*(
                l1*(M1+2*Mf)*torch.sin(x[2] + x[3] + x[4] + x[5] + x[6]) +
                la*Mf*torch.sin(x[2] + x[3] + x[4] + x[5] + x[6] + x[7])
            ), # column 7
            0.5*la*Mf*torch.sin(x[2] + x[3] + x[4] + x[5] + x[6] + x[7])  # column 8
        ]),

        # -----------------------
        # Row 3 (8 columns)
        # -----------------------
        torch.stack([
            0.5*(
                (-1)*la*(4*M1+4*M2+3*Mf+2*Mp)*torch.cos(x[2]) +
                (-1)*l1*(3*M1+2*(2*M2+Mf+Mp))*torch.cos(x[2] + x[3]) +
                (-2)*l2*M1*torch.cos(x[2] + x[3] + x[4]) +
                (-3)*l2*M2*torch.cos(x[2] + x[3] + x[4]) +
                (-2)*l2*Mf*torch.cos(x[2] + x[3] + x[4]) +
                (-2)*l2*Mp*torch.cos(x[2] + x[3] + x[4]) +
                2*l2*M1*torch.cos(x[2] + x[3] + x[4] + x[5]) +
                l2*M2*torch.cos(x[2] + x[3] + x[4] + x[5]) +
                2*l2*Mf*torch.cos(x[2] + x[3] + x[4] + x[5]) +
                l1*M1*torch.cos(x[2] + x[3] + x[4] + x[5] + x[6]) +
                2*l1*Mf*torch.cos(x[2] + x[3] + x[4] + x[5] + x[6]) +
                la*Mf*torch.cos(x[2] + x[3] + x[4] + x[5] + x[6] + x[7])
            ), # column 1
            0.5*(
                (-1)*la*(4*M1+4*M2+3*Mf+2*Mp)*torch.sin(x[2]) +
                (-1)*l1*(3*M1+2*(2*M2+Mf+Mp))*torch.sin(x[2] + x[3]) +
                (-2)*l2*M1*torch.sin(x[2] + x[3] + x[4]) +
                (-3)*l2*M2*torch.sin(x[2] + x[3] + x[4]) +
                (-2)*l2*Mf*torch.sin(x[2] + x[3] + x[4]) +
                (-2)*l2*Mp*torch.sin(x[2] + x[3] + x[4]) +
                2*l2*M1*torch.sin(x[2] + x[3] + x[4] + x[5]) +
                l2*M2*torch.sin(x[2] + x[3] + x[4] + x[5]) +
                2*l2*Mf*torch.sin(x[2] + x[3] + x[4] + x[5]) +
                l1*M1*torch.sin(x[2] + x[3] + x[4] + x[5] + x[6]) +
                2*l1*Mf*torch.sin(x[2] + x[3] + x[4] + x[5] + x[6]) +
                la*Mf*torch.sin(x[2] + x[3] + x[4] + x[5] + x[6] + x[7])
            ), # column 2
            2*I1x + 2*I2x + Ipx +
            0.25*(la**2*Mf) +
            0.25*M1*(l1 + 2*la*torch.cos(x[3]))**2 +
            Mp*(l2 + l1*torch.cos(x[4]) + la*torch.cos(x[3] + x[4]))**2 +
            0.25*M2*(l2 + 2*l1*torch.cos(x[4]) + 2*la*torch.cos(x[3]+x[4]))**2 +
            0.25*M2*(l2 + (-2)*l2*torch.cos(x[5]) + (-2)*l1*torch.cos(x[4]+x[5]) +
                     (-2)*la*torch.cos(x[3]+x[4]+x[5]))**2 +
            M1*((-0.5)*l1 + (-1)*l2*torch.cos(x[6]) + l2*torch.cos(x[5]+x[6]) +
                l1*torch.cos(x[4]+x[5]+x[6]) +
                la*torch.cos(x[3]+x[4]+x[5]+x[6]))**2 +
            Mf*((-0.5)*la + (-1)*l1*torch.cos(x[7]) + (-1)*l2*torch.cos(x[6]+x[7]) +
                l2*torch.cos(x[5]+x[6]+x[7]) +
                l1*torch.cos(x[4]+x[5]+x[6]+x[7]) +
                la*torch.cos(x[3]+x[4]+x[5]+x[6]+x[7]))**2 +
            la**2*M1*(torch.sin(x[3]))**2 +
            M2*(l1*torch.sin(x[4]) + la*torch.sin(x[3]+x[4]))**2 +
            Mp*(l1*torch.sin(x[4]) + la*torch.sin(x[3]+x[4]))**2 +
            M2*(l2*torch.sin(x[5]) + l1*torch.sin(x[4]+x[5]) +
                la*torch.sin(x[3]+x[4]+x[5]))**2 +
            M1*((-1)*l2*torch.sin(x[6]) + l2*torch.sin(x[5]+x[6]) +
                l1*torch.sin(x[4]+x[5]+x[6]) +
                la*torch.sin(x[3]+x[4]+x[5]+x[6]))**2 +
            Mf*((-1)*l1*torch.sin(x[7]) + (-1)*l2*torch.sin(x[6]+x[7]) +
                l2*torch.sin(x[5]+x[6]+x[7]) +
                l1*torch.sin(x[4]+x[5]+x[6]+x[7]) +
                la*torch.sin(x[3]+x[4]+x[5]+x[6]+x[7]))**2,  # column 3
            0.25*(
                8*I1x + 8*I2x + 4*Ipx +
                6*l1**2*M1 + 8*l2**2*M1 +
                8*l1**2*M2 + 6*l2**2*M2 +
                8*l1**2*Mf + 8*l2**2*Mf + la**2*Mf +
                4*(l1**2 + l2**2)*Mp +
                2*l1*la*(3*M1 + 2*(2*M2+Mf+Mp))*torch.cos(x[3]) +
                4*l1*l2*(2*M1 + 3*M2 + 2*(Mf+Mp))*torch.cos(x[4]) +
                (-2)*(
                    (-1)*l2*la*(2*M1+3*M2+2*(Mf+Mp))*torch.cos(x[3] + x[4]) +
                    2*l2**2*(2*M1+M2+2*Mf)*torch.cos(x[5]) +
                    4*l1*l2*M1*torch.cos(x[4]+x[5]) +
                    2*l1*l2*M2*torch.cos(x[4]+x[5]) +
                    4*l1*l2*Mf*torch.cos(x[4]+x[5]) +
                    2*l2*la*M1*torch.cos(x[3]+x[4]+x[5]) +
                    l2*la*M2*torch.cos(x[3]+x[4]+x[5]) +
                    2*l2*la*Mf*torch.cos(x[3]+x[4]+x[5]) +
                    (-2)*l1*l2*M1*torch.cos(x[6]) +
                    (-4)*l1*l2*Mf*torch.cos(x[6]) +
                    2*l1*l2*M1*torch.cos(x[5] + x[6]) +
                    4*l1*l2*Mf*torch.cos(x[5] + x[6]) +
                    2*l1**2*M1*torch.cos(x[4]+x[5]+x[6]) +
                    4*l1**2*Mf*torch.cos(x[4]+x[5]+x[6]) +
                    l1*la*M1*torch.cos(x[3]+x[4]+x[5]+x[6]) +
                    2*l1*la*Mf*torch.cos(x[3]+x[4]+x[5]+x[6]) +
                    (-2)*l1*la*Mf*torch.cos(x[7]) +
                    (-2)*l2*la*Mf*torch.cos(x[6]+x[7]) +
                    2*l2*la*Mf*torch.cos(x[5]+x[6]+x[7]) +
                    2*l1*la*Mf*torch.cos(x[4]+x[5]+x[6]+x[7]) +
                    la**2*Mf*torch.cos(x[3]+x[4]+x[5]+x[6]+x[7])
                )
            ),  # column 4
            (0.25)*(
            4*I1x+8*I2x+4*Ipx
            +l1**2*M1+8*l2**2*M1+6*l2**2*M2
            +4*l1**2*Mf+8*l2**2*Mf+la**2*Mf
            +4*l2**2*Mp+2*l1*l2*(2*M1+3*M2+2*(Mf+Mp))*torch.cos(x[4])
            +2*l2*la*(2*M1+3*M2+2*(Mf+Mp))*torch.cos(x[3]+x[4])
            +(-2)*(2*l2**2*(2*M1+M2+2*Mf)*torch.cos(x[5])
            +l1*l2*(2*M1+M2+2*Mf)*torch.cos(x[4]+x[5])
            +2*l2*la*M1*torch.cos(x[3]+x[4]+x[5])
            +l2*la*M2*torch.cos(x[3]+x[4]+x[5])
            +2*l2*la*Mf*torch.cos(x[3]+x[4]+x[5])
            +(-2)*l1*l2*M1*torch.cos(x[6])
            +(-4)*l1*l2*Mf*torch.cos(x[6])
            +2*l1*l2*M1*torch.cos(x[5]+x[6])
            +4*l1*l2*Mf*torch.cos(x[5]+x[6])
            +l1**2*M1*torch.cos(x[4]+ x[5]+x[6])
            +2*l1**2*Mf*torch.cos(x[4]+x[5]+x[6])
            +l1*la*M1*torch.cos(x[3]+x[4]+x[5]+x[6])
            +2*l1*la*Mf*torch.cos(x[3]+x[4]+x[5]+x[6])
            +(-2)*l1*la*Mf*torch.cos(x[7])
            +(-2)*l2*la*Mf*torch.cos(x[6]+x[7])
            +2*l2*la*Mf*torch.cos(x[5]+x[6]+x[7])
            +l1*la*Mf*torch.cos(x[4]+x[5]+x[6]+x[7])
            +la**2*Mf*torch.cos(x[3]+x[4]+x[5]+x[6]+x[7]))
            ), # column 5
            (0.25)*(
            4*I1x+4*I2x+l1**2*M1
            +4*l2**2*M1+l2**2*M2
            +(4*(l1**2+l2**2)+la**2)*Mf
            +(-2)*l2**2*(2*M1+M2+2*Mf)*torch.cos(x[5])
            +(-2)*l1*l2*(2*M1+M2+2*Mf)*torch.cos(x[4]+x[5])
            +(-2)*(l2*la*(2*M1+M2+2*Mf)*torch.cos(x[3]+x[4]+x[5])
            +l1*(M1+2*Mf)*((-2)*l2*torch.cos(x[6])
            +l2*torch.cos(x[5]+x[6])
            +l1*torch.cos(x[4]+x[5]+x[6])
            +la*torch.cos(x[3]+x[4]+x[5]+x[6]))
            +la*Mf*(l2*((-2)*torch.cos(x[6])+torch.cos(x[5]+x[6]))+l1*((-2)+torch.cos(x[4]+x[5]+x[6]))+ la*torch.cos(x[3]+x[4]+x[5]+x[6]))*torch.cos(x[7])
            +(-1)*la*Mf*((-2)*l2*torch.sin(x[6])+l2*torch.sin(x[5]+x[6])+l1*torch.sin(x[4]+x[5]+x[6])+ la*torch.sin(x[3]+x[4]+x[5]+x[6]))*torch.sin(x[7]))
            ), # column 6
            (0.25)*(
            4*I1x+la**2*Mf+l1**2*(M1+4*Mf)
            +2*l1*l2*(M1+2*Mf)*torch.cos(x[6])
            +(-2)*l1*l2*(M1+2*Mf)*torch.cos(x[5]+x[6])
            +(-2)*(
                l1**2*(M1+  2*Mf)*torch.cos(x[4]+x[5]+x[6])
                +la*(l1*(M1+2*Mf)*torch.cos(x[3]+x[4]+x[5]+x[6])+Mf*((-2)*l1*torch.cos(x[7])+(-1)*l2*torch.cos(x[6]+x[7])+l2*torch.cos(x[5]+x[6]+x[7])+l1*torch.cos(x[4]+x[5]+x[6]+x[7])+ la*torch.cos(x[3]+x[4]+x[5]+x[6]+x[7])))
                )
            ), # column 7
            (0.25)*la*Mf*(
                la+2*l1*torch.cos(x[7])+2*l2*torch.cos(x[6]+x[7])+(-2)*l2*torch.cos(x[5]+x[6]+ x[7])+(-2)*l1*torch.cos(x[4]+x[5]+x[6]+x[7])+(-2)*la*torch.cos(x[3]+ x[4]+x[5]+x[6]+x[7])) #column 8

        ]),

        # -----------------------
        # Row 4 (8 columns)
        # -----------------------
        torch.stack([
            0.5*(
                (-1)*l1*(3*M1+2*(2*M2+Mf+Mp))*torch.cos(x[2]+x[3]) +
                (-1)*l2*(2*M1+3*M2+2*(Mf+Mp))*torch.cos(x[2]+x[3]+x[4]) +
                2*l2*M1*torch.cos(x[2]+x[3]+x[4]+x[5]) +
                l2*M2*torch.cos(x[2]+x[3]+x[4]+x[5]) +
                2*l2*Mf*torch.cos(x[2]+x[3]+x[4]+x[5]) +
                l1*M1*torch.cos(x[2]+x[3]+x[4]+x[5]+x[6]) +
                2*l1*Mf*torch.cos(x[2]+x[3]+x[4]+x[5]+x[6]) +
                la*Mf*torch.cos(x[2]+x[3]+x[4]+x[5]+x[6]+x[7])
            ),  # column 1
            0.5*(
                (-1)*l1*(3*M1+2*(2*M2+Mf+Mp))*torch.sin(x[2]+x[3]) +
                (-1)*l2*(2*M1+3*M2+2*(Mf+Mp))*torch.sin(x[2]+x[3]+x[4]) +
                2*l2*M1*torch.sin(x[2]+x[3]+x[4]+x[5]) +
                l2*M2*torch.sin(x[2]+x[3]+x[4]+x[5]) +
                2*l2*Mf*torch.sin(x[2]+x[3]+x[4]+x[5]) +
                l1*M1*torch.sin(x[2]+x[3]+x[4]+x[5]+x[6]) +
                2*l1*Mf*torch.sin(x[2]+x[3]+x[4]+x[5]+x[6]) +
                la*Mf*torch.sin(x[2]+x[3]+x[4]+x[5]+x[6]+x[7])
            ),  # column 2
            0.25*(
                8*I1x + 8*I2x + 4*Ipx +
                6*l1**2*M1 + 8*l2**2*M1 +
                8*l1**2*M2 + 6*l2**2*M2 +
                8*l1**2*Mf + 8*l2**2*Mf + la**2*Mf +
                4*(l1**2 + l2**2)*Mp +
                2*l1*la*(3*M1 + 2*(2*M2+Mf+Mp))*torch.cos(x[3]) +
                4*l1*l2*(2*M1 + 3*M2 + 2*(Mf+Mp))*torch.cos(x[4]) +
                (-2)*(
                    (-1)*l2*la*(2*M1+3*M2+2*(Mf+Mp))*torch.cos(x[3] + x[4]) +
                    2*l2**2*(2*M1+M2+2*Mf)*torch.cos(x[5]) +
                    4*l1*l2*M1*torch.cos(x[4]+x[5]) +
                    2*l1*l2*M2*torch.cos(x[4]+x[5]) +
                    4*l1*l2*Mf*torch.cos(x[4]+x[5]) +
                    2*l2*la*M1*torch.cos(x[3]+x[4]+x[5]) +
                    l2*la*M2*torch.cos(x[3]+x[4]+x[5]) +
                    2*l2*la*Mf*torch.cos(x[3]+x[4]+x[5]) +
                    (-2)*l1*l2*M1*torch.cos(x[6]) +
                    (-4)*l1*l2*Mf*torch.cos(x[6]) +
                    2*l1*l2*M1*torch.cos(x[5]+x[6]) +
                    4*l1*l2*Mf*torch.cos(x[5]+x[6]) +
                    2*l1**2*M1*torch.cos(x[4]+x[5]+x[6]) +
                    4*l1**2*Mf*torch.cos(x[4]+x[5]+x[6]) +
                    l1*la*M1*torch.cos(x[3]+x[4]+x[5]+x[6]) +
                    2*l1*la*Mf*torch.cos(x[3]+x[4]+x[5]+x[6]) +
                    (-2)*l1*la*Mf*torch.cos(x[7]) +
                    (-2)*l2*la*Mf*torch.cos(x[6]+x[7]) +
                    2*l2*la*Mf*torch.cos(x[5]+x[6]+x[7]) +
                    2*l1*la*Mf*torch.cos(x[4]+x[5]+x[6]+x[7]) +
                    la**2*Mf*torch.cos(x[3]+x[4]+x[5]+x[6]+x[7])
                )
            ),  # column 3
            2*I1x + 2*I2x + Ipx + 
            l1**2*((1.5)*M1 + 2*(M2+Mf) + Mp) +
            0.25*(
                la**2*Mf + l2**2*(8*M1 + 6*M2 + 8*Mf + 4*Mp)
            ) +
            l1*l2*(2*M1 + 3*M2 + 2*(Mf+Mp))*torch.cos(x[4]) +
            (-1)*l2**2*(2*M1+M2+2*Mf)*torch.cos(x[5]) +
            (-2)*l1*l2*M1*torch.cos(x[4]+x[5]) +
            (-1)*l1*l2*M2*torch.cos(x[4]+x[5]) +
            (-2)*l1*l2*Mf*torch.cos(x[4]+x[5]) +
            l1*l2*M1*torch.cos(x[6]) +
            2*l1*l2*Mf*torch.cos(x[6]) +
            (-1)*l1*l2*M1*torch.cos(x[5]+x[6]) +
            (-2)*l1*l2*Mf*torch.cos(x[5]+x[6]) +
            (-1)*l1**2*M1*torch.cos(x[4]+x[5]+x[6]) +
            (-2)*l1**2*Mf*torch.cos(x[4]+x[5]+x[6]) +
            l1*la*Mf*torch.cos(x[7]) +
            l2*la*Mf*torch.cos(x[6]+x[7]) +
            (-1)*l2*la*Mf*torch.cos(x[5]+x[6]+x[7]) +
            (-1)*l1*la*Mf*torch.cos(x[4]+x[5]+x[6]+x[7]),  # column 4
            0.25*(
                4*I1x + 8*I2x + 4*Ipx +
                l1**2*M1 + 8*l2**2*M1 + 6*l2**2*M2 +
                4*l1**2*Mf + 8*l2**2*Mf + la**2*Mf +
                4*l2**2*Mp +
                2*l1*l2*(2*M1+3*M2+2*(Mf+Mp))*torch.cos(x[4]) +
                (-4)*l2**2*(2*M1+M2+2*Mf)*torch.cos(x[5]) +
                (-2)*(
                    l1*l2*(2*M1+M2+2*Mf)*torch.cos(x[4]+x[5]) +
                    l1*(M1+2*Mf)*(
                        (-2)*l2*torch.cos(x[6]) +
                        2*l2*torch.cos(x[5]+x[6]) +
                        l1*torch.cos(x[4]+x[5]+x[6])
                    ) +
                    la*Mf*(
                        2*l2*((-1)*torch.cos(x[6]) + torch.cos(x[5]+x[6])) +
                        l1*((-2) + torch.cos(x[4]+x[5]+x[6]))
                    )*torch.cos(x[7]) +
                    (-1)*la*Mf*(
                        (-2)*l2*torch.sin(x[6]) +
                        2*l2*torch.sin(x[5]+x[6]) +
                        l1*torch.sin(x[4]+x[5]+x[6])
                    )*torch.sin(x[7])
                )
            ),  # column 5
            (0.25)*(
            4*I1x+4*I2x+la**2*Mf+l1**2*(M1+4*Mf)+l2**2*(4*M1+M2+4*Mf)
            +(-2)*(
            l2**2*(2*M1+M2+ 2*Mf)*torch.cos(x[5])
            +l1*l2*(2*M1+M2+2*Mf)*torch.cos(x[4]+x[5])
            +l1*(M1+2*Mf)*((-2)*l2*torch.cos(x[6])+l2*torch.cos(x[5]+x[6])+l1*torch.cos(x[4]+x[5]+x[6]))
            +la*Mf*(l2*((-2)*torch.cos(x[6])+torch.cos(x[5]+ x[6]))+l1*((-2)+torch.cos(x[4]+x[5]+x[6])))*torch.cos(x[7])
            +(-1)*la*Mf*((-2)*l2*torch.sin(x[6])+l2*torch.sin(x[5]+x[6])+l1*torch.sin(x[4]+x[5]+x[6]))*torch.sin(x[7])
            )
            ), # column 6
            (0.25)*(
            4*I1x+la**2*Mf+l1**2*(M1+ 4*Mf)
            +2*l1*l2*(M1+2*Mf)*torch.cos(x[6])
            +(-2)*l1*l2*(M1+2*Mf)*torch.cos(x[5]+x[6])
            +(-2)*(l1**2*(M1+2*Mf)*torch.cos(x[4]+x[5]+x[6])
            +la*Mf*((-2)*l1*torch.cos(x[7])
            +(-1)*l2*torch.cos(x[6]+x[7])
            +l2*torch.cos(x[5]+x[6]+x[7])
            +l1*torch.cos(x[4]+x[5]+x[6]+x[7])))
            ), # column 7
            (0.25)*la*Mf*(la+2*l1*torch.cos(x[7])+2*l2*torch.cos(x[6]+x[7])+(-2)*l2*torch.cos(x[5]+x[6]+x[7])+(-2)*l1*torch.cos(x[4]+x[5]+x[6]+x[7])) # column 8

        ]),

        # -----------------------
        # Row 5 (8 columns)
        # -----------------------
        torch.stack([
            0.5*(
                (-1)*l2*(2*M1+3*M2+2*(Mf+Mp))*torch.cos(x[2] + x[3] + x[4]) +
                l2*(2*M1+M2+2*Mf)*torch.cos(x[2] + x[3] + x[4] + x[5]) +
                l1*(M1+2*Mf)*torch.cos(x[2] + x[3] + x[4] + x[5] + x[6]) +
                la*Mf*torch.cos(x[2] + x[3] + x[4] + x[5] + x[6] + x[7])
            ),  # column 1
            0.5*(
                (-1)*l2*(2*M1+3*M2+2*(Mf+Mp))*torch.sin(x[2] + x[3] + x[4]) +
                l2*(2*M1+M2+2*Mf)*torch.sin(x[2] + x[3] + x[4] + x[5]) +
                l1*(M1+2*Mf)*torch.sin(x[2] + x[3] + x[4] + x[5] + x[6]) +
                la*Mf*torch.sin(x[2] + x[3] + x[4] + x[5] + x[6] + x[7])
            ),  # column 2
            0.25*(
                4*I1x + 8*I2x + 4*Ipx +
                l1**2*M1 + 8*l2**2*M1 + 6*l2**2*M2 +
                4*l1**2*Mf + 8*l2**2*Mf + la**2*Mf +
                4*l2**2*Mp +
                2*l1*l2*(2*M1+3*M2+2*(Mf+Mp))*torch.cos(x[4]) +
                2*l2*la*(2*M1+3*M2+2*(Mf+Mp))*torch.cos(x[3] + x[4]) +
                (-2)*(
                    2*l2**2*(2*M1+M2+2*Mf)*torch.cos(x[5]) +
                    l1*l2*(2*M1+M2+2*Mf)*torch.cos(x[4]+x[5]) +
                    2*l2*la*M1*torch.cos(x[3]+x[4]+x[5]) +
                    l2*la*M2*torch.cos(x[3]+x[4]+x[5]) +
                    2*l2*la*Mf*torch.cos(x[3]+x[4]+x[5]) +
                    (-2)*l1*l2*M1*torch.cos(x[6]) +
                    (-4)*l1*l2*Mf*torch.cos(x[6]) +
                    2*l1*l2*M1*torch.cos(x[5]+x[6]) +
                    4*l1*l2*Mf*torch.cos(x[5]+x[6]) +
                    l1**2*M1*torch.cos(x[4]+x[5]+x[6]) +
                    2*l1**2*Mf*torch.cos(x[4]+x[5]+x[6]) +
                    l1*la*M1*torch.cos(x[3]+x[4]+x[5]+x[6]) +
                    2*l1*la*Mf*torch.cos(x[3]+x[4]+x[5]+x[6]) +
                    (-2)*l1*la*Mf*torch.cos(x[7]) +
                    (-2)*l2*la*Mf*torch.cos(x[6]+x[7]) +
                    2*l2*la*Mf*torch.cos(x[5]+x[6]+x[7]) +
                    l1*la*Mf*torch.cos(x[4]+x[5]+x[6]+x[7]) +
                    la**2*Mf*torch.cos(x[3]+x[4]+x[5]+x[6]+x[7])
                )
            ),  # column 3
            0.25*(
                4*I1x + 8*I2x + 4*Ipx +
                l1**2*M1 + 8*l2**2*M1 + 6*l2**2*M2 +
                4*l1**2*Mf + 8*l2**2*Mf + la**2*Mf +
                4*l2**2*Mp +
                2*l1*l2*(2*M1+3*M2+2*(Mf+Mp))*torch.cos(x[4]) +
                (-4)*l2**2*(2*M1+M2+2*Mf)*torch.cos(x[5]) +
                (-2)*(
                    l1*l2*(2*M1+M2+2*Mf)*torch.cos(x[4]+x[5]) +
                    l1*(M1+2*Mf)*(
                        (-2)*l2*torch.cos(x[6]) +
                        2*l2*torch.cos(x[5]+x[6]) +
                        l1*torch.cos(x[4]+x[5]+x[6])
                    ) +
                    la*Mf*(
                        2*l2*((-1)*torch.cos(x[6]) + torch.cos(x[5]+x[6])) +
                        l1*((-2) + torch.cos(x[4]+x[5]+x[6]))
                    )*torch.cos(x[7]) +
                    (-1)*la*Mf*(
                        (-2)*l2*torch.sin(x[6]) +
                        2*l2*torch.sin(x[5]+x[6]) +
                        l1*torch.sin(x[4]+x[5]+x[6])
                    )*torch.sin(x[7])
                )
            ),  # column 4
            (0.25)*(
            4*I1x+8*I2x+4*Ipx
            +la**2*Mf+l1**2*(M1+4*Mf)
            +2*l2**2*(4*M1+3*M2+4*Mf+2*Mp)
            +(-4)*l2**2*(2*M1+M2+2*Mf)*torch.cos(x[5])
            +4*(
                l1*l2*(M1+2*Mf)*torch.cos(x[6])
                +(-1)*l1*l2*(M1+2*Mf)*torch.cos(x[5]+x[6])
                +la*Mf*(l1*torch.cos(x[7])
                    +l2*torch.cos(x[6]+x[7])
                    +(-1)*l2*torch.cos(x[5]+x[6]+x[7]))
                )
            ), # column 5
            (0.25)*(
            4*I1x+4*I2x
            +l1**2*M1+4*l2**2*M1+l2**2*M2
            +(4*(l1**2+l2**2)+la**2)*Mf
            +(-2)*l2**2*(2*M1+M2+2*Mf)*torch.cos(x[5])
            +4*l1*l2*(M1+2*Mf)*torch.cos(x[6])
            +(-2)*(l1*l2*(M1+2*Mf)*torch.cos(x[5]+x[6])
            +la*Mf*(
                (-2)*l1*torch.cos(x[7])
                +(-2)*l2*torch.cos(x[6]+x[7])
                +l2*torch.cos(x[5]+x[6]+x[7]))
                )
            ), # column 6
            (0.25)*(
            4*I1x+la**2*Mf+l1**2*(M1+4*Mf)
            +2*l1*l2*(M1+2*Mf)*torch.cos(x[6])
            +(-2)*l1*l2*(M1+2*Mf)*torch.cos(x[5]+x[6])
            +2*la*Mf*(2*l1*torch.cos(x[7])+l2*torch.cos(x[6]+x[7])+(-1)*l2*torch.cos(x[5]+x[6]+x[7]))
            ), # column 7
            (0.25)*la*Mf*(la+2*l1*torch.cos(x[7])+2*l2*torch.cos(x[6]+x[7])+(-2)*l2*torch.cos(x[5]+x[6]+x[7])) # column 8
        ]),

        # -----------------------
        # Row 6 (8 columns)
        # -----------------------
        torch.stack([
            0.5*(
                l2*(2*M1+M2+2*Mf)*torch.cos(x[2] + x[3] + x[4] + x[5]) +
                l1*(M1+2*Mf)*torch.cos(x[2] + x[3] + x[4] + x[5] + x[6]) +
                la*Mf*torch.cos(x[2] + x[3] + x[4] + x[5] + x[6] + x[7])
            ),  # column 1
            0.5*(
                l2*(2*M1+M2+2*Mf)*torch.sin(x[2] + x[3] + x[4] + x[5]) +
                l1*(M1+2*Mf)*torch.sin(x[2] + x[3] + x[4] + x[5] + x[6]) +
                la*Mf*torch.sin(x[2] + x[3] + x[4] + x[5] + x[6] + x[7])
            ),  # column 2
            0.25*(
                4*I1x + 4*I2x +
                l1**2*M1 + 4*l2**2*M1 + l2**2*M2 +
                (4*(l1**2 + l2**2) + la**2)*Mf +
                (-2)*l2**2*(2*M1+M2+2*Mf)*torch.cos(x[5]) +
                (-2)*l1*l2*(2*M1+M2+2*Mf)*torch.cos(x[4]+x[5]) +
                (-2)*(
                    l2*la*(2*M1+M2+2*Mf)*torch.cos(x[3]+x[4]+x[5]) +
                    l1*(M1+2*Mf)*(
                        (-2)*l2*torch.cos(x[6]) +
                        l2*torch.cos(x[5]+x[6]) +
                        l1*torch.cos(x[4]+x[5]+x[6]) +
                        la*torch.cos(x[3]+x[4]+x[5]+x[6])
                    ) +
                    la*Mf*(
                        l2*((-2)*torch.cos(x[6]) + torch.cos(x[5]+x[6])) +
                        l1*((-2) + torch.cos(x[4]+x[5]+x[6])) +
                        la*torch.cos(x[3]+x[4]+x[5]+x[6])
                    )*torch.cos(x[7]) +
                    (-1)*la*Mf*(
                        (-2)*l2*torch.sin(x[6]) +
                        l2*torch.sin(x[5]+x[6]) +
                        l1*torch.sin(x[4]+x[5]+x[6]) +
                        la*torch.sin(x[3]+x[4]+x[5]+x[6])
                    )*torch.sin(x[7])
                )
            ),  # column 3
            (0.25)*(
            4*I1x+4*I2x+la**2*Mf+l1**2*(M1+4*Mf)+l2**2*(4*M1+M2+4*Mf)
            +(-2)*(
                l2**2*(2*M1+M2+2*Mf)*torch.cos(x[5])
                +l1*l2*(2*M1+M2+2*Mf)*torch.cos(x[4]+x[5])
                +l1*(M1+2*Mf)*((-2)*l2*torch.cos(x[6])+l2*torch.cos(x[5]+x[6])+l1*torch.cos(x[4]+x[5]+x[6]))
                +la*Mf*(l2*((-2)*torch.cos(x[6])+torch.cos(x[5]+x[6]))+l1*((-2)+torch.cos(x[4]+x[5]+x[6])))*torch.cos(x[7])
                +(-1)*la*Mf*((-2)*l2*torch.sin(x[6])+l2*torch.sin(x[5]+x[6])+l1*torch.sin(x[4]+x[5]+x[6]))*torch.sin(x[7])
                )
            ),   # column 4
            (0.25)*(
                4*I1x+4*I2x+l1**2*M1+4*l2**2*M1+l2**2*M2
                +(4*(l1**2+l2**2)+la**2)*Mf
                +(-2)*l2**2*(2*M1+M2+2*Mf)*torch.cos(x[5])
                +4*l1*l2*(M1+2*Mf)*torch.cos(x[6])
                +(-2)*(
                    l1*l2*(M1+2*Mf)*torch.cos(x[5]+x[6])
                    +la*Mf*((-2)*l1*torch.cos(x[7])+(-2)*l2*torch.cos(x[6]+x[7])+l2*torch.cos(x[5]+x[6]+x[7]))
                )
            ), # column 5
            I1x+(0.25)*(4*I2x+la**2*Mf+l1**2*(M1+4*Mf)+l2**2*(4*M1+M2+4*Mf))+l1*l2*(M1+2*Mf)*torch.cos(x[6])+l1*la*Mf*torch.cos(x[7])+l2*la*Mf*torch.cos(x[6]+x[7]), # column 6
            (0.25)*(4*I1x+la**2*Mf+l1**2*(M1+4*Mf)+2*l1*l2*(M1+2*Mf)*torch.cos(x[6])+4*l1*la*Mf*torch.cos(x[7])+2*l2*la*Mf*torch.cos(x[6]+x[7])), # column 7
            (0.25)*la*Mf*(la+2*l1*torch.cos(x[7])+2*l2*torch.cos(x[6]+x[7])) # column 8
        ]),

        # -----------------------
        # Row 7 (8 columns)
        # -----------------------
        torch.stack([
            (0.5)*(l1*(M1+2*Mf)*torch.cos(x[2]+x[3]+x[4]+x[5]+x[6])+la*Mf*torch.cos(x[2]+x[3]+x[4]+x[5]+x[6]+x[7])), # col 1
            (0.5)*(l1*(M1+2*Mf)*torch.sin(x[2]+x[3]+x[4]+x[5]+x[6])+la*Mf*torch.sin(x[2]+x[3]+x[4]+x[5]+x[6]+x[7])),  # col 2
            (0.25)*(
            4*I1x+la**2*Mf+l1**2*(M1+4*Mf)
            +2*l1*l2*(M1+2*Mf)*torch.cos(x[6])
            +(-2)*l1*l2*(M1+2*Mf)*torch.cos(x[5]+x[6])
            +(-2)*(
                l1**2*(M1+2*Mf)*torch.cos(x[4]+x[5]+x[6])
                +la*(
                    l1*(M1+2*Mf)*torch.cos(x[3]+x[4]+x[5]+x[6])
                    +Mf*(
                        (-2)*l1*torch.cos(x[7])
                        +(-1)*l2*torch.cos(x[6]+x[7])
                        +l2*torch.cos(x[5]+x[6]+x[7])
                        +l1*torch.cos(x[4]+x[5]+x[6]+x[7])
                        +la*torch.cos(x[3]+x[4]+x[5]+x[6]+x[7])
                        )
                    )
                )
            ), # col 3
            (0.25)*(
            4*I1x+la**2*Mf+l1**2*(M1+4*Mf)
            +2*l1*l2*(M1+2*Mf)*torch.cos(x[6])
            +(-2)*l1*l2*(M1+2*Mf)*torch.cos(x[5]+x[6])
            +(-2)*(
                l1**2*(M1+2*Mf)*torch.cos(x[4]+x[5]+x[6])
                +la*Mf*((-2)*l1*torch.cos(x[7])+(-1)*l2*torch.cos(x[6]+x[7])+l2*torch.cos(x[5]+x[6]+x[7])+l1*torch.cos(x[4]+x[5]+x[6]+x[7]))
                )
            ),   # col 4
            (0.25)*(
            4*I1x+la**2*Mf+l1**2*(M1+4*Mf)
            +2*l1*l2*(M1+2*Mf)*torch.cos(x[6])
            +(-2)*l1*l2*(M1+2*Mf)*torch.cos(x[5]+x[6])
            +2*la*Mf*(2*l1*torch.cos(x[7])+l2*torch.cos(x[6]+x[7])+(-1)*l2*torch.cos(x[5]+x[6]+x[7]))
            ),  # col 5
            (0.25)*(4*I1x+la**2*Mf+l1**2*(M1+4*Mf)+2*l1*l2*(M1+2*Mf)*torch.cos(x[6])+4*l1*la*Mf*torch.cos(x[7])+2*l2*la*Mf*torch.cos(x[6]+x[7])),  # col 6
            I1x+(0.25)*(la**2*Mf+l1**2*(M1+4*Mf))+l1*la*Mf*torch.cos(x[7]),  # col 7
            (0.25)*la*Mf*(la+2*l1*torch.cos(x[7])) # col 8
        ]),

        # -----------------------
        # Row 8 (8 columns)
        # -----------------------
        torch.stack([
            (0.5)*la*Mf*torch.cos(x[2]+x[3]+x[4]+x[5]+x[6]+x[7]),  # col 1

            (0.5)*la*Mf*torch.sin(x[2]+x[3]+x[4]+x[5]+x[6]+x[7]), # col 2

            (0.25)*la*Mf*(la+2*l1*torch.cos(x[7])+2*l2*torch.cos(x[6]+x[7])+(-2)*l2*torch.cos(x[5]+x[6]+x[7])+(-2)*l1*torch.cos(x[4]+x[5]+x[6]+x[7])+(-2)*la*torch.cos(x[3]+x[4]+x[5]+x[6]+x[7])), # col 3

            (0.25)*la*Mf*(la+2*l1*torch.cos(x[7])+2*l2*torch.cos(x[6]+x[7])+(-2)*l2*torch.cos(x[5]+x[6]+x[7])+(-2)*l1*torch.cos(x[4]+x[5]+x[6]+x[7])), # col 4

            (0.25)*la*Mf*(la+2*l1*torch.cos(x[7])+2*l2*torch.cos(x[6]+x[7])+(-2)*l2*torch.cos(x[5]+x[6]+x[7])),  # col 5

            (0.25)*la*Mf*(la+2*l1*torch.cos(x[7])+2*l2*torch.cos(x[6]+x[7])),  # col 6

            (0.25)*la*Mf*(la+2*l1*torch.cos(x[7])), # col 7

            torch.tensor((0.25)*la**2*Mf, device=device)  # col 8
        ])
    ])

    return Mmtx

##################################
# Potential Vector
def calculate_Nvect(x, device, Mp, Mf, g, l1, l2, la, lf, I1x, I2x, Ipx, slope, M1, M2):
    Nvect = torch.stack([
        # 1) 0
        torch.tensor(0.0, device=device),

        # 2) g*(2*(M1+M2+Mf) + Mp)
        torch.tensor(g * (2*(M1 + M2 + Mf) + Mp),device=device),

        # 3) 0.5*g*( ... )
        0.5 * g * (
            (-1)*la*(4*M1 + 4*M2 + 3*Mf + 2*Mp)*torch.sin(x[2]) +
            (-1)*l1*(3*M1 + 2*(2*M2 + Mf + Mp))*torch.sin(x[2] + x[3]) +
            (-2)*l2*M1*torch.sin(x[2] + x[3] + x[4]) +
            (-3)*l2*M2*torch.sin(x[2] + x[3] + x[4]) +
            (-2)*l2*Mf*torch.sin(x[2] + x[3] + x[4]) +
            (-2)*l2*Mp*torch.sin(x[2] + x[3] + x[4]) +
             2*l2*M1*torch.sin(x[2] + x[3] + x[4] + x[5]) +
             l2*M2*torch.sin(x[2] + x[3] + x[4] + x[5]) +
             2*l2*Mf*torch.sin(x[2] + x[3] + x[4] + x[5]) +
             l1*M1*torch.sin(x[2] + x[3] + x[4] + x[5] + x[6]) +
             2*l1*Mf*torch.sin(x[2] + x[3] + x[4] + x[5] + x[6]) +
             la*Mf*torch.sin(x[2] + x[3] + x[4] + x[5] + x[6] + x[7])
        ),

        # 4) 0.5*g*( ... )
        0.5 * g * (
            (-1)*l1*(3*M1 + 2*(2*M2 + Mf + Mp))*torch.sin(x[2] + x[3]) +
            (-1)*l2*(2*M1 + 3*M2 + 2*(Mf + Mp))*torch.sin(x[2] + x[3] + x[4]) +
             2*l2*M1*torch.sin(x[2] + x[3] + x[4] + x[5]) +
             l2*M2*torch.sin(x[2] + x[3] + x[4] + x[5]) +
             2*l2*Mf*torch.sin(x[2] + x[3] + x[4] + x[5]) +
             l1*M1*torch.sin(x[2] + x[3] + x[4] + x[5] + x[6]) +
             2*l1*Mf*torch.sin(x[2] + x[3] + x[4] + x[5] + x[6]) +
             la*Mf*torch.sin(x[2] + x[3] + x[4] + x[5] + x[6] + x[7])
        ),

        # 5) 0.5*g*( ... )
        0.5 * g * (
            (-1)*l2*(2*M1 + 3*M2 + 2*(Mf + Mp))*torch.sin(x[2] + x[3] + x[4]) +
             l2*(2*M1 + M2 + 2*Mf)*torch.sin(x[2] + x[3] + x[4] + x[5]) +
             l1*(M1 + 2*Mf)*torch.sin(x[2] + x[3] + x[4] + x[5] + x[6]) +
             la*Mf*torch.sin(x[2] + x[3] + x[4] + x[5] + x[6] + x[7])
        ),

        # 6) 0.5*g*( ... )
        0.5 * g * (
             l2*(2*M1 + M2 + 2*Mf)*torch.sin(x[2] + x[3] + x[4] + x[5]) +
             l1*(M1 + 2*Mf)*torch.sin(x[2] + x[3] + x[4] + x[5] + x[6]) +
             la*Mf*torch.sin(x[2] + x[3] + x[4] + x[5] + x[6] + x[7])
        ),

        # 7) 0.5*g*( ... )
        0.5 * g * (
             l1*(M1 + 2*Mf)*torch.sin(x[2] + x[3] + x[4] + x[5] + x[6]) +
             la*Mf*torch.sin(x[2] + x[3] + x[4] + x[5] + x[6] + x[7])
        ),

        # 8) 0.5*g*la*Mf*torch.sin(...)
        0.5 * g * la * Mf * torch.sin(x[2] + x[3] + x[4] + x[5] + x[6] + x[7])
        
    ]).unsqueeze(-1)   

    # [8,1]
    return Nvect