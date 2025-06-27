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
    # vmap (not necessary here!)
    def get_PDE_Loss_trajectory(self, model, X):
        # helpers

        # x:(4,1) y(3,1)
        def matching_condition_vmap(x):
            q_dot = x[4:].view(-1,1)

            M = model.calculate_M(x)
            M_hat = model.calculate_M_hat(x)
            M_inv = custom_inverse(M)
            M_hat_inv =  custom_inverse(M_hat)

            Cqdot = dynamics.calculate_Cqdot(model.calculate_M, x)
            Chatqdot = dynamics.calculate_Cqdot(model.calculate_M_hat, x)

            N = model.calculate_N(x)
            Nhat = N


            hat = Chatqdot + Nhat 
            diff = Cqdot + N - M @ M_hat_inv @ hat 
            matching_tensor = self.B_left_annihilator @ diff 
            return 0.5 * torch.linalg.vector_norm(matching_tensor, ord=2)

        # use torch.vmap to vectorize the transform function
        vmap_pde_loss_func = torch.vmap(matching_condition_vmap)

        # apply the vectorized transform function on data
        L1s = vmap_pde_loss_func(X)
        return L1s

    ##################################
    # no vmap
    def get_PDE_Loss_trajectory_batch(self, model, X):
        # X shape: (B, 8)
        q_dot = X[:, 4:]  # (B, 4)

        M = model.calculate_M(X)            # (B, 4, 4)
        M_hat = model.calculate_M_hat(X)    # (B, 4, 4)
        M_inv = custom_inverse(M)        # (B, 4, 4)
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
        return 0.5 * torch.norm(matching_tensor, dim=1)


    #################################################################
    # PDE loss (mean across all data points: scalar)
    def get_PDE_Loss(self, model, X):
        L1s = self.get_PDE_Loss_trajectory(model, X)
        L1_mean = L1s.mean()
        # return avg loss, normalized avg loss
        return L1_mean, L1_mean/(torch.max(L1s) + 1e-8)

    ##################################
    # TODO: add adaptive sampling

    ##################################
    # modify this to compute everything in one go
    def total_loss(self, model, X, weights):

        # X shape: (B, 8)
        q_dot = X[:, 4:]  # (B, 4)

        M = model.calculate_M(X)            # (B, 4, 4)
        M_hat = model.calculate_M_hat(X)    # (B, 4, 4)
        M_inv = custom_inverse(M)        # (B, 4, 4)
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
        L1s = 0.5 * torch.norm(matching_tensor, dim=1)

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
        pos_penalties = torch.nn.functional.softplus(-eigvals_hat).sum(dim=-1)
        pos_def_loss = pos_penalties.mean()


        assert len(weights) == 5
        W1, W2, W4, W5, W6 = weights[0], weights[1], weights[2], weights[3], weights[4]

        total =  W1*residual_loss + W2* control_loss + W4*deviation_loss + W5*eig_loss + W6*sparse_loss 
        losses = [
            residual_loss.detach().cpu(),
            control_loss.detach().cpu(),
            deviation_loss.detach().cpu(),
            eig_loss.detach().cpu(),
            sparse_loss.detach().cpu(),
            pos_def_loss.detach().cpu()
        ]
        return total, losses


########################################################################
# compute weights
def calculate_weights(loss_funcs, model, X, print_path=None):
    with torch.no_grad():
        weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        _, [residual_loss, control_loss, deviation_loss, eig_loss, sparse_loss, pos_def_loss] = loss_funcs.total_loss(model, X, weights)

        # eps = 1e-10
        # weights = np.array([
        #     1.0 / (1.0 + np.log(1.0 + residual_loss + eps)),
        #     1.0 / (1.0 + np.log(1.0 + control_loss + eps)),
        #     1.0 / (1.0 + np.log(1.0 + deviation_loss + eps)),
        #     1.0 / (1.0 + np.log(1.0 + eig_loss + eps)),
        #     1.0 / (1.0 + np.log(1.0 + sparse_loss + eps))
        # ])
        # clamped_weights = np.clip(weights, a_min=0.1, a_max=10.0)

        # Optional debug printing
        if print_path is not None:
            with open(print_path, "a") as f:
                #print(f"W1: {clamped_weights[0]:.6f}, W2: {clamped_weights[1]:.6f}, W4: {clamped_weights[2]:.6f}, "
                #      f"W5: {clamped_weights[3]:.6f}, W6: {clamped_weights[4]:.6f}", file=f)
                print(f"L1: {residual_loss:.6f}, L2: {control_loss:.6f}, "
                      f"L4: {deviation_loss:.6f}, L5: {eig_loss:.6f}, L6: {sparse_loss:.6f}, "
                      f"L7: {pos_def_loss:.6f}", file=f)

        #return clamped_weights
        return [1.0, 0.0, 0.0, 0.0, 0.0]