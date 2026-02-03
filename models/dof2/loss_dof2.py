import torch
from core.utils import bounded_quad_loss, damped_pseudo_inverse
from models.dof2 import dynamics
from core.sampling import uniform_sampling
import numpy as np
from core.utils import assert_finite

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def custom_inverse(M):
#     cond_threshold = 1e4
#     epsilon = 1e-2

#     is_batched = M.dim() == 3

#     if not is_batched:
#         M = M.unsqueeze(0)  # → (1, d, d)

#     B, d, _ = M.shape

#     conds = torch.linalg.cond(M)  # (B,)
#     conds = conds.unsqueeze(-1).unsqueeze(-1)  # shape: (B, 1, 1) for broadcasting

#     I = torch.eye(d, device=M.device).expand(B, d, d)
    
#     # mask: shape (B, 1, 1), value 1.0 if ill-conditioned, 0.0 if not
#     mask = (conds > cond_threshold).float()

#     M_reg = M + mask * (epsilon * I)

#     result = torch.linalg.pinv(M_reg)

#     if not is_batched:
#         result = result.squeeze(0)  # back to (d, d)

#     return result

def custom_inverse(A):
    det_eps = 1e-6
    a,b = A[...,0,0], A[...,0,1]
    c,d = A[...,1,0], A[...,1,1]
    det = a*d - b*c
    det = torch.sign(det) * torch.clamp(det.abs(), min=A.new_tensor(det_eps))
    return torch.stack([
        torch.stack([ d/det, -b/det], dim=-1),
        torch.stack([-c/det,  a/det], dim=-1),
    ], dim=-2)


#vmap functions for partial derivatives
# batch version
def calculate_d1(model, X):

    def d1_vmap(x):
        def func_d1(x):
            qdot = x[2:].view(-1,1)  #2x1
            M = model.calculate_M(x)
            p = M @ qdot
            Md_tilde = model.calculate_Md_hat(x)
            Md = M @ Md_tilde @ M 
            Md_inv = custom_inverse(Md)

            out = torch.transpose(p, -2, -1) @ Md_inv @ p
            return out.squeeze()

        jacobian_d1 = torch.func.jacrev(func_d1)(x)
        return jacobian_d1[:2].view(2,1)

    return torch.vmap(d1_vmap)(X)

def calculate_S(model, X):

    def s_vmap(x):
        def func_s(x):
            qdot = x[2:].view(-1,1)  #2x1
            M = model.calculate_M(x)
            out = M @ qdot
            return out.squeeze()

        jacobian_s = torch.func.jacrev(func_s)(x).reshape(2,4)[:,:2]
        return jacobian_s

    return torch.vmap(s_vmap)(X)

def calculate_d3(model, X):

    def d3_vmap(x):
        def func_d3(x):
            qdot = x[2:].view(-1,1)  #2x1
            M = model.calculate_M(x)
            p = M @ qdot
            M_inv = custom_inverse(M)

            out = torch.transpose(p, -2, -1) @ M_inv @ p
            return out.squeeze()

        jacobian_d3 = torch.func.jacrev(func_d3)(x)
        return jacobian_d3[:2].view(2,1)

    return torch.vmap(d3_vmap)(X)

def calculate_dV(X):
    def dV_vmap(x):
        return dynamics.calculate_dVdq(x).view(2,1)

    return torch.vmap(dV_vmap)(X)

def calculate_dVd(model, X):

    def dVd_vmap(x):
        def func_Vd(x):
            out = model.calculate_Vd(x)
            return out.squeeze()

        jacobian_dVd = torch.func.jacrev(func_Vd)(x)
        return jacobian_dVd[:2].view(2,1)

    return torch.vmap(dVd_vmap)(X)

def calculate_dV(X):
    def dV_vmap(x):
        return dynamics.calculate_dVdq(x).view(2,1)

    return torch.vmap(dV_vmap)(X)


def calculate_J(X):
    def dJ_vmap(x):
        return dynamics.calculate_J(x).view(2,1)
    return torch.vmap(dJ_vmap)(X)

class customLoss:
    def __init__(self):
        self.G_left_annihilator = torch.tensor([[1.0, 0.0]], device=device)
        self.G = torch.tensor([[0],[1.0]], device = device)
        self.W = torch.tensor([[0,1.0],[-1.0, 0]], device = device)

    ##################################
    # no vmap
    def get_PDE_Loss_trajectory(self, model, X):

        qdot = X[:, 2:].unsqueeze(-1)     # [n,2,1]

        M = model.calculate_M(X)  #[n,2,2]  
        assert_finite('M', M)
        Md_tilde = model.calculate_Md_hat(X)  #[n,2,2]
        assert_finite('Md_tilde', Md_tilde)
        Md = M @ Md_tilde @ M    #[n,2,2]
        assert_finite('Md', Md)

        M_inv = custom_inverse(M)  #[n,2,2]
        assert_finite('M_inv', M_inv)
        Md_inv = custom_inverse(Md) #[n,2,2]
        assert_finite('Md_inv', Md_inv)
        Md_tilde_inv = M @ Md_inv @ M
        assert_finite('Md_tilde_inv', Md_tilde_inv)

        d1 = calculate_d1(model, X) #[n,2,1]
        assert_finite('d1', d1)
        S = calculate_S(model, X) # [n,2,2]
        assert_finite('S', S)
        d3 = calculate_d3(model, X) #[n,2,1]
        assert_finite('d3', d3)
        dV = calculate_dV(X)        #[n,2,1]
        assert_finite('dV', dV)
        dVd = calculate_dVd(model, X) #[n,2,1]
        assert_finite('dVd', dVd)

        p = M @ qdot



        J = calculate_J(X) #[n,2,1]
        assert_finite('J', J)
        alpha = (torch.transpose(qdot, -2, -1) @ Md_tilde_inv @ J) #[n,1,1]
        assert_finite('alpha', alpha)
        J2_tilde = alpha * self.W  #[n,2,2]
        J2 = M @ J2_tilde @ M + S @ Md_tilde @ M - M @ Md_tilde @ torch.transpose(S, -2, -1)
        assert_finite('J2', J2)


        p1 = Md @ M_inv @ d1 
        p2 = 2.0 * J2 @ Md_inv @ p

        p5 = Md @ M_inv @ dVd

        eqn1 = self.G_left_annihilator @ (p1 - p2 - d3)   #[n,1,1]
        eqn2 = self.G_left_annihilator @ (dV - p5) #[n,1,1]
        eqn1 = eqn1.squeeze(-1).squeeze(-1) #[n]
        eqn2 = eqn2.squeeze(-1).squeeze(-1) #[n]

        return eqn1 + eqn2 

    ##################################
    # pointwise residual for adaptive sampling (RAD)
    @torch.no_grad()
    def residual_pointwise(self, model, X):
        return self.get_PDE_Loss_trajectory(model, X)

    ##################################
    # modify this to compute everything in one go
    def total_loss(self, model, X):
        
        residual_loss = self.get_PDE_Loss_trajectory(model, X).mean()

        assert_finite('residual_loss', residual_loss)
        
        # Vd has min at equilibrium
        w_grad = 1.0
        w_hess = 1.0
        x_eq = torch.zeros(4, device=X.device, dtype=X.dtype)
        q_eq = x_eq[:2].detach().clone().requires_grad_(True)
        # build the full x passed to Vd: [q_eq, qdot_eq] with qdot fixed (detached)
        x_for_Vd = torch.cat([q_eq, x_eq[2:].detach()])  # shape [4]

        Vd_eq = model.calculate_Vd(x_for_Vd).squeeze()
        # gradient of Vd w.r.t q only
        grad_q = torch.autograd.grad(Vd_eq, q_eq, create_graph=True)[0]  # shape [2]

        # zero grad at x_eq
        loss_grad = torch.sum(grad_q**2)

        H = q_eq.new_zeros(2, 2)
        for i in range(2):
            H[i, :] = torch.autograd.grad(grad_q[i], q_eq, retain_graph=True, create_graph=True)[0]

        # symmetrize (numerical stability)
        H = 0.5 * (H + H.T)

        assert_finite('H', H)
        
        # PSD penalty without eigendecomposition (2x2 principal minors)
        # PSD iff a>=0, c>=0, det>=0 for symmetric [[a,b],[b,c]]
        a = H[0, 0]
        c = H[1, 1]
        b = H[0, 1]
        det = a * c - b * b
        eps = q_eq.new_tensor(1e-6)
        # pos def hessian 
        loss_hessian = loss_hessian = (
            torch.relu(eps - a) +
            torch.relu(eps - c) + 
            torch.relu(eps - det))

        assert_finite('loss_hessian', loss_hessian)
        assert_finite('loss_grad', loss_grad)

        Vdmin_loss = w_grad * loss_grad + w_hess * loss_hessian



        sparse_X = uniform_sampling(n_samples=100, input_dim=model.INPUT_DIM, device=X.device,
                     lower_bounds=dynamics.LOWER_BOUNDS, upper_bounds=dynamics.UPPER_BOUNDS)
        sparse_loss = self.get_PDE_Loss_trajectory(model, sparse_X).mean()


        W1, W2, W3 = 1.0, 100.0, 0.1

        #total =  W1*residual_loss + W2* control_loss + W4*deviation_loss + W5*eig_loss + W6*sparse_loss 
        total =  W1*residual_loss + W2* Vdmin_loss + W3 * sparse_loss
        
        losses = [
            residual_loss.detach().cpu(),
            Vdmin_loss.detach().cpu(),
            sparse_loss.detach().cpu()
        ]
        return total, losses
    #####################################################
    # new total loss with max error loss
    def total_loss_checkpoint(self, model, X, max_error):

        residual_loss = self.get_PDE_Loss_trajectory(model, X).mean()
        
        # Vd has min at equilibrium
        w_grad = 1.0
        w_hess = 1.0
        x_eq = torch.tensor([0.0, 0.0, 0.0, 0.0], device = X.device, requires_grad=True)
        Vd_eq = model.calculate_Vd(x_eq).squeeze()
        grad_Vd = torch.autograd.grad(Vd_eq, x_eq, create_graph=True)[0]
        grad_q = grad_Vd[:2]

        # zero grad at x_eq
        loss_grad = torch.sum(grad_q**2)

        H = torch.zeros(2, 2, device=X.device)
        for i in range(2):
            H[i] = torch.autograd.grad(grad_q[i], x_eq, retain_graph=True)[0][:2]
        eigvals = torch.linalg.eigvalsh(H)
        # pos def hessian 
        loss_hessian = torch.sum(torch.relu(-eigvals))

        Vdmin_loss = loss_Vd_min = w_grad * loss_grad + w_hess * loss_hessian



        sparse_X = uniform_sampling(n_samples=100, input_dim=model.INPUT_DIM, device=X.device,
                     lower_bounds=dynamics.LOWER_BOUNDS, upper_bounds=dynamics.UPPER_BOUNDS)
        sparse_loss = self.get_PDE_Loss_trajectory(model, sparse_X).mean()

        # max_error
        max_error_loss = self.get_PDE_Loss_trajectory(model, max_error).mean()


        W1, W2, W3, W4 = 1.0, 100.0, 0.1, 0.1

        #total =  W1*residual_loss + W2* control_loss + W4*deviation_loss + W5*eig_loss + W6*sparse_loss 
        total =  W1*residual_loss + W2* Vdmin_loss + W3 * sparse_loss + W4*max_error_loss
        
        losses = [
            residual_loss.detach().cpu(),
            Vdmin_loss.detach().cpu(),
            sparse_loss.detach().cpu(),
            max_error_loss.detach().cpu()
        ]
        return total, losses
        


