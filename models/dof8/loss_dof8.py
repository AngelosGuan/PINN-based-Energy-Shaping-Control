import torch
import numpy as np
from core.utils import damped_pseudo_inverse, bounded_quad_loss
from models.dof8.dynamics import slope, lf, determine_phase_masks, calculate_Cqdot, CONTROL_BOUND
from models.dof8.helpers import sampling_with_contact_condition
from configs.config_dof8 import EPSILON
from core.sampling import uniform_sampling

# might need dataloader here for OOM issue in sparse sampling

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def custom_inverse(A):
    return damped_pseudo_inverse(A, lambda_reg=1e-1)

class customLoss:
    def __init__(self):
        self.B = torch.tensor([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]], dtype=torch.float32, device=device)

    ##################################
    def get_PDE_Loss_trajectory(self, model, X):
        # helpers

        def calculate_B_annihilator(x):
            M = model.calculate_M(x)
            I = torch.eye(3,device=device)
            Y = - M[0:3, 3:8] @ custom_inverse(M[3:8, 3:8])
            B = torch.cat((I, Y), dim=1)
            return B

        def calculate_AWA(x, invM):
            phase1_mask, phase2_mask, phase3_mask = determine_phase_masks(x=x, slope=slope, lf=lf, EPSILON=EPSILON, device=device)
            A1 = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0],[0, 1, 0, 0, 0, 0, 0, 0]], dtype=x.dtype, device=device)
            A1_T = torch.transpose(A1, 0, 1)
            A2 = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0],[0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0]], dtype=x.dtype, device=device)
            A2_T = torch.transpose(A2, 0, 1)
            A3 = torch.cat([
                torch.tensor([[1, 0], [0, 1]], dtype=x.dtype, device=device),
                torch.stack([lf * torch.sin(x[2]), -lf * torch.cos(x[2])]).view(2, 1),
                torch.zeros((2, 5), device=device)
            ], dim=1)
            A3_T = torch.transpose(A3, 0, 1)

            temp = (
                phase1_mask * A1_T @ custom_inverse(A1 @ invM @ A1_T) @ A1 + 
                phase2_mask * A2_T @ custom_inverse(A2 @ invM @ A2_T) @ A2 + 
                phase3_mask * A3_T @ custom_inverse(A3 @ invM @ A3_T) @ A3
                )
            return temp

        def common_factor(x, invM):
            # I - A^T W A M^-1
            temp = calculate_AWA(x, invM)
            return torch.eye(8,device=device) - temp @ invM 

        def calculate_lambda(x, invM, mat):
            # except for Mlamb = M
            com = common_factor(x, invM)
            return com @ mat

        def matching_condition_vmap(x):
            q_dot = x[8:].view(-1,1)
            
            Mmtx = model.calculate_M(x)
            Mmtx_hat = model.calculate_M_hat(x)
            Mmtx_inv = custom_inverse(Mmtx)
            Mmtx_hat_inv = custom_inverse(Mmtx_hat)

            B_annihilator = calculate_B_annihilator(x)

            Cqdot = calculate_Cqdot(model.calculate_M, x)
            Chatqdot = calculate_Cqdot(model.calculate_M_hat, x)

            Cqdot = calculate_lambda(x, Mmtx_inv, Cqdot)
            Chatqdot = calculate_lambda(x, Mmtx_hat_inv, Chatqdot)

            Nvect = model.calculate_N(x)
            Nvect_hat = Nvect

            Nlamb = calculate_lambda(x, Mmtx_inv, Nvect)
            Nlamb_hat = calculate_lambda(x, Mmtx_hat_inv, Nvect_hat)

            hat = Chatqdot + Nlamb_hat
            diff = Cqdot + Nlamb - Mmtx @ Mmtx_hat_inv @ hat
            matching_tensor = B_annihilator @ diff
            return 0.5 * torch.linalg.vector_norm(matching_tensor, ord=2)

        # use torch.vmap to vectorize the transform function
        vmap_pde_loss_func = torch.vmap(matching_condition_vmap)

        # apply the vectorized transform function on data
        L1s = vmap_pde_loss_func(X)

        return L1s

    #################################################################
    # PDE loss (mean across all data points: scalar)
    def get_PDE_Loss(self, model, X):
        L1s = self.get_PDE_Loss_trajectory(model, X)
        L1_mean = L1s.mean()
        # return avg loss, normalized avg loss
        return L1_mean, L1_mean/(torch.max(L1s) + 1e-8)

    ##################################
    # Boundary Loss. 
    # KE shaping only
    # A single Loss for entire model, do not require data point. 
    def get_Boundary_Loss(self, model):
        x0 = torch.zeros(model.INPUT_DIM, dtype=torch.float32, device=device)
        y0_pred = model.calculate_M_hat(x0)
        y0_true = model.calculate_M(x0)
        out = torch.nn.functional.mse_loss(y0_pred, y0_true)
        return out

    ##################################
    # compute PDE Loss with Control Law trajectory to avoid repetitive calculation 
    def get_PDE_and_control_trajectory(self, model, X):

        def calculate_B_annihilator(x):
            M = model.calculate_M(x)
            I = torch.eye(3,device=device)
            Y = - M[0:3, 3:8] @ custom_inverse(M[3:8, 3:8])
            B = torch.cat((I, Y), dim=1)
            return B

        def calculate_AWA(x, invM):
            phase1_mask, phase2_mask, phase3_mask = determine_phase_masks(x=x, slope=slope, lf=lf, EPSILON=EPSILON, device=device)
            A1 = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0],[0, 1, 0, 0, 0, 0, 0, 0]], dtype=x.dtype, device=device)
            A1_T = torch.transpose(A1, 0, 1)
            A2 = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0],[0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0]], dtype=x.dtype, device=device)
            A2_T = torch.transpose(A2, 0, 1)
            A3 = torch.cat([
                torch.tensor([[1, 0], [0, 1]], dtype=x.dtype, device=device),
                torch.stack([lf * torch.sin(x[2]), -lf * torch.cos(x[2])]).view(2, 1),
                torch.zeros((2, 5), device=device)
            ], dim=1)
            A3_T = torch.transpose(A3, 0, 1)

            temp = (
                phase1_mask * A1_T @ custom_inverse(A1 @ invM @ A1_T) @ A1 + 
                phase2_mask * A2_T @ custom_inverse(A2 @ invM @ A2_T) @ A2 + 
                phase3_mask * A3_T @ custom_inverse(A3 @ invM @ A3_T) @ A3
                )
            return temp

        def common_factor(x, invM):
            # I - A^T W A M^-1
            temp = calculate_AWA(x, invM)
            return torch.eye(8,device=device) - temp @ invM 

        def calculate_lambda(x, invM, mat):
            # except for Mlamb = M
            com = common_factor(x, invM)
            return com @ mat

        def matching_condition_vmap(x):
            q_dot = x[8:].view(-1,1)
            
            Mmtx = model.calculate_M(x)
            Mmtx_hat = model.calculate_M_hat(x)
            Mmtx_inv = damped_pseudo_inverse(Mmtx)
            Mmtx_hat_inv = damped_pseudo_inverse(Mmtx_hat)

            B_annihilator = calculate_B_annihilator(x)

            Cqdot = calculate_Cqdot(model.calculate_M, x)
            Chatqdot = calculate_Cqdot(model.calculate_M_hat, x)

            Cqdot = calculate_lambda(x, Mmtx_inv, Cqdot)
            Chatqdot = calculate_lambda(x, Mmtx_hat_inv, Chatqdot)

            Nvect = model.calculate_N(x)
            Nvect_hat = Nvect

            Nlamb = calculate_lambda(x, Mmtx_inv, Nvect)
            Nlamb_hat = calculate_lambda(x, Mmtx_hat_inv, Nvect_hat)

            hat = Chatqdot + Nlamb_hat
            diff = Cqdot + Nlamb - Mmtx @ Mmtx_hat_inv @ hat
            matching_tensor = B_annihilator @ diff

            B_lamb = calculate_lambda(x, Mmtx_inv, self.B)
            B_T = torch.transpose(B_lamb,0,1)
            psuedo_inv_B = custom_inverse(B_T @ B_lamb) @ B_T
            control_tensor = psuedo_inv_B @ diff

            return 0.5 * torch.linalg.vector_norm(matching_tensor, ord=2), control_tensor[0][0]

        # use torch.vmap to vectorize the transform function
        vmap_pde_loss_func = torch.vmap(matching_condition_vmap)

        # apply the vectorized transform function on data
        L1s, us = vmap_pde_loss_func(X)

        return L1s, us

    ##################################
    # compute avg PDE Loss with Control Law trajectory to avoid repetitive calculation 
    # PDE loss is just avg PDE loss and should go to 0
    # control law is bounded with absolute value below 60 
    def get_PDE_and_control_loss(self, model, X):
        L1s, us = self.get_PDE_and_control_trajectory(model, X)
        L1_loss = L1s.mean()
        control_loss = bounded_quad_loss(us, CONTROL_BOUND)
        return L1_loss, control_loss

    ##################################
    # Mdot-2C constraint loss (mean across all data points: scalar)
    def get_dynamics_constraint_loss(self, model, X):
        def calculate_Cmtx_hat(x):
            dMdq = torch.func.jacrev(lambda x: model.calculate_M_hat(x))(x).reshape(8,8,16)[:,:,:8]
            qdot = x[8:].view(-1,1)
            C = torch.stack([qdot]*8, dim=1).reshape(8,8)
            # Loop over k and j for C[k, j]
            for k in range(8):
                for j in range(8):
                    # Compute the component C[k, j]
                    for i in range(8):
                        C[k, j] += 0.5 * (dMdq[k, j, i] + dMdq[k, i, j] - dMdq[i, j, k]) * qdot[i,0]
            
            return C

        # transpose(Mhatdot-2Chat)+(Mhatdot-2Chat)
        def skew_symmetric_vmap(x):
            dMdq = torch.func.jacrev(lambda x: model.calculate_M_hat(x))(x).reshape(8,8,16)[:,:,:8]
            qdot = x[8:].view(-1,1)
            Mhatdot = dMdq @ qdot
            Mhatdot = Mhatdot.reshape(8,8)
            Chat = calculate_Cmtx_hat(x)
            mat = Mhatdot - 2*Chat
            return torch.linalg.matrix_norm(torch.transpose(mat, 0, 1) + mat)


        # use torch.vmap to vectorize the transform function
        vmap_css = torch.vmap(skew_symmetric_vmap)

        out = vmap_css(X)
        
        L4_mean = out.mean()
        norm_L4mean = L4_mean/(torch.max(out) + 1e-8)
        # return avg loss, normalized avg loss
        return L4_mean, norm_L4mean

    ##################################
    def get_deviation_loss_trajectory(self, model, X):
        """
        Input:
            X: [n, 4]
        Output:
            deviations: [n]
        used avg error for difference. Alternative: max error
        """
        M_hat = model.calculate_M_hat(X)  # [n, d, d]
        M = model.calculate_M(X)          # [n, d, d]

        # Frobenius norm of difference
        diff_norm = torch.linalg.norm(M_hat - M, ord='fro', dim=(1, 2))  # [n]
        base_norm = torch.linalg.norm(M, ord='fro', dim=(1, 2)) + 1e-12  # [n]

        return diff_norm / base_norm  # [n]

    ##################################
    def get_deviation_loss(self, model, X, bound=0.5):
        """
        Computes bounded deviation loss across batched data X
        """
        diffs = self.get_deviation_loss_trajectory(model, X)  # [n]
        return bounded_quad_loss(diffs, bound)  # scalar

    ##################################
    def pos_eig_loss_trajectory(self, model, X):
        """
        Input:
            X: [n, 4]
        Output:
            eig_penalties: [n]
        """
        M_hat = model.calculate_M_hat(X)  # [n, d, d]
        eigvals = torch.linalg.eigvalsh(M_hat)  # [n, d]

        # Penalize nonpositive eigenvalues using softplus(-eigval)
        penalties = torch.nn.functional.softplus(-eigvals).sum(dim=-1)  # [n]

        return penalties

    ##################################
    def pos_eig_loss(self, model, X):
        """
        Returns:
            mean penalty across all batch points (scalar)
        """
        penalties = self.pos_eig_loss_trajectory(model, X)  # [n]
        return penalties.mean()  # scalar

    ##################################
    def eig_range_loss_trajectory(self, model, X, alpha):
        """
        Input:
            X: [n, 4]
            alpha: float (range scaling factor)
        Output:
            penalties: [n]
        """
        M_hat = model.calculate_M_hat(X)  # [n, d, d]
        M = model.calculate_M(X)          # [n, d, d]

        M_eigvals = torch.linalg.eigvalsh(M)      # [n, d]
        a = M_eigvals.min(dim=-1).values * (1.0 - alpha)  # [n]
        b = M_eigvals.max(dim=-1).values * (1.0 + alpha)  # [n]

        eigvals_hat = torch.linalg.eigvalsh(M_hat)  # [n, d]

        # Expand a, b to match eigvals shape [n, d]
        a = a.unsqueeze(-1)
        b = b.unsqueeze(-1)

        lower_violation = torch.nn.functional.softplus(a - eigvals_hat)  # [n, d]
        upper_violation = torch.nn.functional.softplus(eigvals_hat - b)  # [n, d]

        penalties = (lower_violation + upper_violation).sum(dim=-1)  # [n]

        return penalties

    ##################################
    def eig_range_loss(self, model, X, alpha=0.5):
        """
        Returns:
            Mean penalty across batch (scalar)
        """
        penalties = self.eig_range_loss_trajectory(model, X, alpha)  # [n]
        return penalties.mean()

    ##################################
    # use sparse sample for robustness
    def sparse_sample_loss(self, model, sample_size = 9):
        X = sampling_with_contact_condition(num_samples=sample_size, device=device, sampling_func=uniform_sampling)
        loss, _ = self.get_PDE_Loss(model, X)
        return loss

    ##################################
    # TODO: add adaptive sampling

    ##################################
    # def total_loss(self, model, X, weights):
    #     residual_loss, control_loss = self.get_PDE_and_control_loss(model, X)
    #     boundary_loss = self.get_Boundary_Loss(model)
    #     deviation_loss = self.get_deviation_loss(model, X, 0.5)
    #     eig_loss = self.eig_range_loss(model, X, 0.5)
    #     sparse_loss = self.sparse_sample_loss(model, 100)
    #     pos_def_loss = self.pos_eig_loss(model, X)

    #     assert len(weights) == 7
    #     [W1, W2, W3, W4, W5, W6, W7] = weights

    #     total =  W1*residual_loss + W2* control_loss + W3*boundary_loss + W4*deviation_loss + W5*eig_loss + W6*sparse_loss + W7*pos_def_loss
    #     losses = [
    #         residual_loss.detach().cpu(),
    #         control_loss.detach().cpu(),
    #         boundary_loss.detach().cpu(),
    #         deviation_loss.detach().cpu(),
    #         eig_loss.detach().cpu(),
    #         sparse_loss.detach().cpu(),
    #         pos_def_loss.detach().cpu()
    #     ]
    #     return total, losses
    ##################################
    def total_loss(self, model, X, weights):
        residual_loss, _ = self.get_PDE_Loss(model, X)

        assert len(weights) == 1
        [W1] = weights

        total =  W1*residual_loss 
        losses = [
            residual_loss.detach().cpu()
        ]
        return total, losses

########################################################################
# compute weights
# def calculate_weights(loss_funcs, model, X, print_path=None):
#     with torch.no_grad():
#         residual_loss, control_loss = loss_funcs.get_PDE_and_control_loss(model, X)
#         residual_loss = residual_loss.detach().cpu()
#         control_loss = control_loss.detach().cpu()
#         boundary_loss = loss_funcs.get_Boundary_Loss(model).detach().cpu()
#         deviation_loss = loss_funcs.get_deviation_loss(model, X, 0.5).detach().cpu()
#         eig_loss = loss_funcs.eig_range_loss(model, X, 0.5).detach().cpu()
#         sparse_loss = loss_funcs.sparse_sample_loss(model, 100).detach().cpu()
#         pos_def_loss = loss_funcs.pos_eig_loss(model, X).detach().cpu()

#         eps = 1e-10
#         n = float(X.shape[0])
#         weights = np.array([
#             1.0 / (1.0 + np.log(1.0 + residual_loss + eps)),
#             1.0 / (1.0 + np.log(1.0 + control_loss + eps)),
#             1.0 / (1.0 + np.log(1.0 + boundary_loss + eps)),
#             1.0 / (1.0 + np.log(1.0 + deviation_loss + eps)),
#             1.0 / (1.0 + np.log(1.0 + eig_loss + eps)),
#             1.0 / (1.0 + np.log(1.0 + sparse_loss + eps)),
#             1.0 / (1.0 + np.log(1.0 + pos_def_loss + eps))
#         ])
#         clamped_weights = np.clip(weights, a_min=0.5, a_max=5.0)
#         # scale boundary loss by number of samples if present
#         clamped_weights[2] *= n

#         if getattr(model, 'hard_boundary', False):
#             # boundary loss not needed for hard boundary models
#             clamped_weights[2] = 0.0
#         if getattr(model, 'pos_def', False):
#             # positive eigenvalue loss not needed for positive definite models
#             clamped_weights[6] = 0.0

#         # Optional debug printing
#         if print_path is not None:
#             with open(print_path, "a") as f:
#                 print(f"W1: {clamped_weights[0]:.6f}, W2: {clamped_weights[1]:.6f}, W3: {clamped_weights[2]:.6f}, "
#                       f"W4: {clamped_weights[3]:.6f}, W5: {clamped_weights[4]:.6f}, W6: {clamped_weights[5]:.6f}, "
#                       f"W7: {clamped_weights[6]:.6f}", file=f)
#                 print(f"L1: {residual_loss:.6f}, L2: {control_loss:.6f}, L3: {boundary_loss:.6f}, "
#                       f"L4: {deviation_loss:.6f}, L5: {eig_loss:.6f}, L6: {sparse_loss:.6f}, "
#                       f"L7: {pos_def_loss:.6f}", file=f)
#         return clamped_weights

def calculate_weights(loss_funcs, model, X, print_path=None):
    device = X.device
    with torch.no_grad():
        residual_loss, _ = loss_funcs.get_PDE_Loss(model, X)
        residual_loss = residual_loss.detach().cpu()

        eps = 1e-10
        weight_val = 1.0 / (1.0 + torch.log(1.0 + residual_loss + eps))
        clamped = torch.clamp(weight_val, min=0.5, max=5.0).item()

        # Optional debug printing
        if print_path is not None:
            with open(print_path, "a") as f:
                print(f"W1: {clamped:.6f}", file=f)
                print(f"L1: {residual_loss.item():.6f}", file=f)

        return [clamped]  # return as list


# def calculate_weights(loss_funcs, model, X, print_path=None):
#     with torch.no_grad():
#         residual_loss, _ = loss_funcs.get_PDE_Loss(model, X)
#         residual_loss = residual_loss.detach().cpu()

#         eps = 1e-10
#         n = float(X.shape[0])
#         weights = torch.tensor([
#             1.0 / (1.0 + torch.log(1.0 + residual_loss + eps))
#         ],dtype=torch.float32)
#         clamped_weights = torch.clamp(weights, min=0.5, max=5.0).to(device)


#         # Optional debug printing
#         if print_path is not None:
#             with open(print_path, "a") as f:
#                 print(f"W1: {clamped_weights[0].item():.6f}", file=f)
#                 print(f"L1: {residual_loss.item():.6f}", file=f)
#         return clamped_weights