import torch
from core.utils import bounded_quad_loss, damped_pseudo_inverse
from models.dof4 import dynamics
from core.sampling import uniform_sampling
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def custom_inverse(M):
    cond_threshold = 1e4
    epsilon = 1e-2

    is_batched = M.dim() == 3

    if not is_batched:
        M = M.unsqueeze(0)  # → (1, d, d)

    B, d, _ = M.shape

    conds = torch.linalg.cond(M)  # (B,)
    conds = conds.unsqueeze(-1).unsqueeze(-1)  # shape: (B, 1, 1) for broadcasting

    I = torch.eye(d, device=M.device).expand(B, d, d)
    
    # mask: shape (B, 1, 1), value 1.0 if ill-conditioned, 0.0 if not
    mask = (conds > cond_threshold).float()

    M_reg = M + mask * (epsilon * I)

    result = torch.linalg.pinv(M_reg)

    if not is_batched:
        result = result.squeeze(0)  # back to (d, d)

    return result
    #return torch.linalg.pinv(M)
    #return damped_pseudo_inverse(M)

# def custom_inverse(M, epsilon=1e-3, cond_threshold=1e4):
#     """
#     Batch-safe pseudo-inverse with soft regularization.
#     Applies ε * I scaled by normalized condition number when κ(M) is high.
#     """
#     is_batched = M.dim() == 3
#     if not is_batched:
#         M = M.unsqueeze(0)  # (1, d, d)

#     B, d, _ = M.shape
#     conds = torch.linalg.cond(M)  # (B,)
#     conds = conds.clamp(min=1.0)  # avoid division by zero

#     # Scaling factor: (conds / threshold) capped at 1.0
#     scale = (conds / cond_threshold).clamp(max=1.0)  # (B,)
#     scale = scale.view(-1, 1, 1)  # broadcast to (B, 1, 1)

#     I = torch.eye(d, device=M.device).expand(B, d, d)
#     M_reg = M + scale * (epsilon * I)

#     M_inv = torch.linalg.pinv(M_reg)

#     if not is_batched:
#         M_inv = M_inv.squeeze(0)

#     return M_inv


class customLoss:
    def __init__(self):
        self.B_left_annihilator = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0]], device=device)
        self.B = torch.tensor([[0], [0], [1.0], [0]], device=device)
        # self.B = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1.0, 1.0, 0],[0, 0, 0, 0, 0]], device=device)
        B_T = torch.transpose(self.B,0,1)
        self.B_pinv = torch.inverse(B_T @ self.B) @ B_T

    ##################################
    # no vmap
    def get_PDE_Loss_trajectory(self, model, X):
        # X shape: (B, 8)
        q_dot = X[:, 4:]  # (B, 4)

        M = model.calculate_M(X)            # (B, 4, 4)
        M_hat = model.calculate_M_hat(X)    # (B, 4, 4)
        M_inv = custom_inverse(M)        # (B, 4, 4)


        if self.pos_def:
            # only for KMK, change later
            K = model.forward(X)
            ks = torch.torch.diagonal(K, dim1=-2, dim2=-1)
            ks_inv = 1.0 / ks
            K_inv = torch.diag_embed(ks_inv)
            M_hat_inv = torch.matmul(torch.matmul(K_inv, M_inv), K_inv)
        else:
            M_hat_inv = custom_inverse(M_hat)  # (B, 4, 4)



        Cqdot = dynamics.calculate_Cqdot(model.calculate_M, X)       # (B, 4, 1)
        Chatqdot = dynamics.calculate_Cqdot(model.calculate_M_hat, X)  # (B, 4, 1)

        N = model.calculate_N(X)  # (B, 4, 1)
        Nhat = N  # Change this if N̂ differs from N

        hat = Chatqdot + Nhat                     # (B, 4, 1)
        Mhat_inv_hat = torch.matmul(M_hat_inv, hat)  # (B, 4, 1)
        diff = Cqdot + N - torch.matmul(M, Mhat_inv_hat)  # (B, 4, 1)

        # Apply annihilator: B_left_annihilator shape (k, 4), transpose to (4, k)
        matching_tensor = torch.matmul(diff.squeeze(-1), self.B_left_annihilator.T)  # (B, k)

        # Compute vector norm over k for each batch → (B,)
        return 0.5 * torch.norm(matching_tensor, dim=-1)


    #################################################################
    # PDE loss (mean across all data points: scalar)
    def get_PDE_Loss(self, model, X):
        L1s = self.get_PDE_Loss_trajectory(model, X)
        L1_mean = L1s.mean()
        # return avg loss, normalized avg loss
        return L1_mean, L1_mean/(torch.max(L1s) + 1e-8)

    ##################################
    # pointwise residual for adaptive sampling (RAD)
    @torch.no_grad()
    def residual_pointwise(self, model, X):
        return self.get_PDE_Loss_trajectory(model, X)

    ##################################
    # modify this to compute everything in one go
    def total_loss(self, model, X):

        # X shape: (B, 8)
        q_dot = X[:, 4:]  # (B, 4)

        M = model.calculate_M(X)            # (B, 4, 4)
        M_hat = model.calculate_M_hat(X)    # (B, 4, 4)
        M_inv = custom_inverse(M)        # (B, 4, 4)

        if self.pos_def:
            # only for KMK, change later
            K = model.forward(X)
            ks = torch.torch.diagonal(K, dim1=-2, dim2=-1)
            ks_inv = 1.0 / ks
            K_inv = torch.diag_embed(ks_inv)
            M_hat_inv = torch.matmul(torch.matmul(K_inv, M_inv), K_inv)
        else:
            M_hat_inv = custom_inverse(M_hat)  # (B, 4, 4)

        # TODO: make this batch safe
        Cqdot = dynamics.calculate_Cqdot(model.calculate_M, X)       # (B, 4, 1)
        Chatqdot = dynamics.calculate_Cqdot(model.calculate_M_hat, X)  # (B, 4, 1)

        N = model.calculate_N(X)  # (B, 4, 1)
        Nhat = N  # Change this if N̂ differs from N

        hat = Chatqdot + Nhat                     # (B, 4, 1)
        Mhat_inv_hat = torch.matmul(M_hat_inv, hat)  # (B, 4, 1)
        diff = Cqdot + N - torch.matmul(M, Mhat_inv_hat)  # (B, 4, 1)

        # Apply annihilator: B_left_annihilator shape (k, 4), transpose to (4, k)
        matching_tensor = torch.matmul(diff.squeeze(-1), self.B_left_annihilator.T)  # (B, k)

        # Compute vector norm over k for each batch → (B,)
        L1s = 0.5 * torch.norm(matching_tensor, dim=-1)

        # compute control 
        invB_exp = self.B_pinv.unsqueeze(0).expand(X.shape[0],-1,-1)

        us = torch.bmm(invB_exp, diff).squeeze(-1)     # → (B, 5)

        # residual_loss
        residual_loss = L1s.mean()

        # control_loss
        control_loss = bounded_quad_loss(us, dynamics.CONTROL_BOUND).mean()

        # deviation_loss
        bound = 0.5
        # Frobenius norm of difference
        diff_norm = torch.linalg.norm(M_hat - M, ord='fro', dim=(1, 2))  # [n]
        base_norm = torch.linalg.norm(M, ord='fro', dim=(1, 2)) + 1e-12  # [n]
        diffs = diff_norm / base_norm 
        deviation_loss = bounded_quad_loss(diffs, bound)


        # eig_loss
        alpha = 0.5
        M_eigvals = torch.linalg.eigvalsh(M)      # [n, d]
        a = M_eigvals.min(dim=-1).values * (1.0 - alpha)  # [n]
        b = M_eigvals.max(dim=-1).values * (1.0 + alpha)  # [n]
        eigvals_hat = torch.linalg.eigvalsh(M_hat)  # [n, d]
        a = a.unsqueeze(-1)
        b = b.unsqueeze(-1)
        lower_violation = torch.nn.functional.softplus(a - eigvals_hat)  # [n, d]
        upper_violation = torch.nn.functional.softplus(eigvals_hat - b)  # [n, d]
        penalties = (lower_violation + upper_violation).sum(dim=-1)  # [n]
        eig_loss = penalties.mean()

        # sparse_loss
        sparse_X = uniform_sampling(n_samples=100, input_dim=model.INPUT_DIM, device=X.device,
                     lower_bounds=dynamics.LOWER_BOUNDS, upper_bounds=dynamics.UPPER_BOUNDS)
        sparse_loss, _ = self.get_PDE_Loss(model, sparse_X)


        #### not used

        # pos_def_loss
        min_eig_value = 1e-2
        pos_penalties = torch.nn.functional.softplus(min_eig_value-eigvals_hat).sum(dim=-1)
        pos_def_loss = pos_penalties.mean()


        # assert len(weights) == 5
        # W1, W2, W4, W5, W6 = weights[0], weights[1], weights[2], weights[3], weights[4]

        #total =  W1*residual_loss + W2* control_loss + W4*deviation_loss + W5*eig_loss + W6*sparse_loss 
        total =  1.0*residual_loss + 0.001* control_loss + 0.1*sparse_loss
        
        losses = [
            residual_loss.detach().cpu(),
            control_loss.detach().cpu(),
            deviation_loss.detach().cpu(),
            eig_loss.detach().cpu(),
            sparse_loss.detach().cpu(),
            pos_def_loss.detach().cpu()
        ]
        return total, losses

    #####################################################
    # new total loss with max error loss
    def total_loss_checkpoint(self, model, X, max_error):

        # X shape: (B, 8)
        q_dot = X[:, 4:]  # (B, 4)

        M = model.calculate_M(X)            # (B, 4, 4)
        M_hat = model.calculate_M_hat(X)    # (B, 4, 4)
        M_inv = custom_inverse(M)        # (B, 4, 4)
        if self.pos_def:
            # only for KMK, change later
            K = model.forward(X)
            ks = torch.torch.diagonal(K, dim1=-2, dim2=-1)
            ks_inv = 1.0 / ks
            K_inv = torch.diag_embed(ks_inv)
            M_hat_inv = torch.matmul(torch.matmul(K_inv, M_inv), K_inv)
        else:
            M_hat_inv = custom_inverse(M_hat)  # (B, 4, 4)

        # TODO: make this batch safe
        Cqdot = dynamics.calculate_Cqdot(model.calculate_M, X)       # (B, 4, 1)
        Chatqdot = dynamics.calculate_Cqdot(model.calculate_M_hat, X)  # (B, 4, 1)

        N = model.calculate_N(X)  # (B, 4, 1)
        Nhat = N  # Change this if N̂ differs from N

        hat = Chatqdot + Nhat                     # (B, 4, 1)
        Mhat_inv_hat = torch.matmul(M_hat_inv, hat)  # (B, 4, 1)
        diff = Cqdot + N - torch.matmul(M, Mhat_inv_hat)  # (B, 4, 1)

        # Apply annihilator: B_left_annihilator shape (k, 4), transpose to (4, k)
        matching_tensor = torch.matmul(diff.squeeze(-1), self.B_left_annihilator.T)  # (B, k)

        # Compute vector norm over k for each batch → (B,)
        L1s = 0.5 * torch.norm(matching_tensor, dim=-1)

        # compute control 
        invB_exp = self.B_pinv.unsqueeze(0).expand(X.shape[0],-1,-1)

        us = torch.bmm(invB_exp, diff).squeeze(-1)     # → (B, 5)

        # residual_loss
        residual_loss = L1s.mean()

        # control_loss
        control_loss = bounded_quad_loss(us, dynamics.CONTROL_BOUND).mean()

        # deviation_loss
        bound = 0.5
        # Frobenius norm of difference
        diff_norm = torch.linalg.norm(M_hat - M, ord='fro', dim=(1, 2))  # [n]
        base_norm = torch.linalg.norm(M, ord='fro', dim=(1, 2)) + 1e-12  # [n]
        diffs = diff_norm / base_norm 
        deviation_loss = bounded_quad_loss(diffs, bound)


        # eig_loss
        alpha = 0.5
        M_eigvals = torch.linalg.eigvalsh(M)      # [n, d]
        a = M_eigvals.min(dim=-1).values * (1.0 - alpha)  # [n]
        b = M_eigvals.max(dim=-1).values * (1.0 + alpha)  # [n]
        eigvals_hat = torch.linalg.eigvalsh(M_hat)  # [n, d]
        a = a.unsqueeze(-1)
        b = b.unsqueeze(-1)
        lower_violation = torch.nn.functional.softplus(a - eigvals_hat)  # [n, d]
        upper_violation = torch.nn.functional.softplus(eigvals_hat - b)  # [n, d]
        penalties = (lower_violation + upper_violation).sum(dim=-1)  # [n]
        eig_loss = penalties.mean()

        # sparse_loss
        sparse_X = uniform_sampling(n_samples=100, input_dim=model.INPUT_DIM, device=X.device,
                     lower_bounds=dynamics.LOWER_BOUNDS, upper_bounds=dynamics.UPPER_BOUNDS)
        sparse_loss, _ = self.get_PDE_Loss(model, sparse_X)

        # max error loss
        max_error_loss, _ = self.get_PDE_Loss(model, max_error)


        #### not used

        # pos_def_loss
        min_eig_value = 1e-2
        pos_penalties = torch.nn.functional.softplus(min_eig_value-eigvals_hat).sum(dim=-1)
        pos_def_loss = pos_penalties.mean()



        #total =  W1*residual_loss + W2* control_loss + W4*deviation_loss + W5*eig_loss + W6*sparse_loss 
        total =  1.0*residual_loss + 0.1*sparse_loss
        
        losses = [
            residual_loss.detach().cpu(),
            control_loss.detach().cpu(),
            deviation_loss.detach().cpu(),
            eig_loss.detach().cpu(),
            pos_def_loss.detach().cpu(),
            sparse_loss.detach().cpu(),
            max_error_loss.detach().cpu()
        ]
        return total, losses

########################################################################
# compute weights
def calculate_weights(loss_funcs, model, X, print_path=None):
    # constants
    return [1.0/20, 1.0/763412, 1.0, 1.0, 1.0/20]

