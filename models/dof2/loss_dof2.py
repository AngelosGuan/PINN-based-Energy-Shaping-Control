import torch
from core.utils import damped_pseudo_inverse, bounded_quad_loss, uniform_sampling
from models.dof2 import dynamics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

W1, W2, W3, W4, W5, W6, W7 = 1.0, 1.0, 1.0 ,1.0, 1.0, 1.0, 1.0  # TODO: change this into parameters to total loss

class customLoss:
    def __init__(self, B_left_annihilator):
        self.B_left_annihilator = torch.tensor([[0, 0],[0, 1.0]]).to(device)
        self.B = torch.tensor([[1.0],[0]]).to(device)

    ##################################
    
    def get_PDE_Loss_trajectory(self, model, X):
        # helpers

        def func_M1(x):
            M = model.calculate_M(x)
            q_dot = torch.stack([x[2], x[3]]).view(-1,1)
            return M @ q_dot

        def func_M2(x):
            M = model.calculate_M(x)
            q_dot = torch.stack([x[2], x[3]]).view(-1,1)
            return torch.transpose(q_dot,0,1) @ M @ q_dot

        def func_Mhat1(x):
            M_hat = model.calculate_M_hat(x)
            q_dot = torch.stack([x[2], x[3]]).view(-1,1)
            return M_hat @ q_dot

        def func_Mhat2(x):
            M_hat = model.calculate_M_hat(x)
            q_dot = torch.stack([x[2], x[3]]).view(-1,1)
            return torch.transpose(q_dot,0,1) @ M_hat @ q_dot

        # x:(4,1) y(3,1)
        def matching_condition_vmap(x):
            q_dot = torch.stack([x[2], x[3]]).view(-1,1)
            M = model.calculate_M(x)
            M_hat = model.calculate_M_hat(x)

            M_inv = damped_pseudo_inverse(M)
            M_hat_inv =  damped_pseudo_inverse(M_hat)

            jacobianM1 = torch.func.jacrev(lambda x: func_M1(x))(x).reshape(2,4)[:,:2]
            jacobianM2 = torch.func.jacrev(lambda x: func_M2(x))(x).reshape(1,4)[:,:2]
            jacobianMhat1 = torch.func.jacrev(lambda x: func_Mhat1(x))(x).reshape(2,4)[:,:2]
            jacobianMhat2 = torch.func.jacrev(lambda x: func_Mhat2(x))(x).reshape(1,4)[:,:2]
            matching_tensor = (self.B_left_annihilator @ M) @ (M_inv @ (jacobianM1 @ q_dot - 0.5 * torch.transpose(jacobianM2,0,1)) - M_hat_inv @ (jacobianMhat1 @ q_dot - 0.5 * torch.transpose(jacobianMhat2,0,1)))
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
    # A single Loss for entire model, do not require data point. 
    def get_Boundary_Loss(self, model):
        x0 = torch.tensor([0., 0., 0., 0.]).to(device)
        y0_pred = model.calculate_M_hat(x0)
        y0_true = torch.stack([torch.stack([model.C_ma_tensor, model.C_mb_tensor]),
                torch.stack([model.C_mb_tensor, model.C_mc_tensor])])
        out = torch.nn.functional.mse_loss(y0_pred, y0_true)
        return out

    ##################################
    # compute PDE Loss with Control Law trajectory to avoid repetitive calculation 
    def get_PDE_and_control_trajectory(self, model, X):

        def func_M1(x):
            M = model.calculate_M(x)
            q_dot = torch.stack([x[2], x[3]]).view(-1,1)
            return M @ q_dot

        def func_M2(x):
            M = model.calculate_M(x)
            q_dot = torch.stack([x[2], x[3]]).view(-1,1)
            return torch.transpose(q_dot,0,1) @ M @ q_dot

        def func_Mhat1(x):
            M_hat = model.calculate_M_hat(x)
            q_dot = torch.stack([x[2], x[3]]).view(-1,1)
            return M_hat @ q_dot

        def func_Mhat2(x):
            M_hat = model.calculate_M_hat(x)
            q_dot = torch.stack([x[2], x[3]]).view(-1,1)
            return torch.transpose(q_dot,0,1) @ M_hat @ q_dot

        # x:(4,1) y(3,1)
        def matching_condition_vmap(x):
            q = torch.stack([x[0],x[1]]).view(-1,1)
            q_dot = torch.stack([x[2], x[3]]).view(-1,1)

            M = model.calculate_M(x)
            M_hat = model.calculate_M_hat(x)

            # pseudo inverse
            #M_hat_inv = torch.linalg.pinv(M_hat)

            # regularized pseudo inverse
            M_inv = damped_pseudo_inverse(M)
            M_hat_inv =  damped_pseudo_inverse(M_hat)

            jacobianM1 = torch.func.jacrev(lambda x: func_M1(x))(x).reshape(2,4)[:,:2]
            jacobianM2 = torch.func.jacrev(lambda x: func_M2(x))(x).reshape(1,4)[:,:2]
            jacobianMhat1 = torch.func.jacrev(lambda x: func_Mhat1(x))(x).reshape(2,4)[:,:2]
            jacobianMhat2 = torch.func.jacrev(lambda x: func_Mhat2(x))(x).reshape(1,4)[:,:2]

            common_factor = (M_inv @ (jacobianM1 @ q_dot - 0.5 * torch.transpose(jacobianM2,0,1)) - M_hat_inv @ (jacobianMhat1 @ q_dot - 0.5 * torch.transpose(jacobianMhat2,0,1)))
            
            matching_tensor = (self.B_left_annihilator @ M) @ common_factor

            B_T = torch.transpose(self.B,0,1)
            psuedo_inv_B = torch.inverse(B_T @ self.B) @ B_T
            control_tensor = (psuedo_inv_B @ M) @ common_factor
            
            return 0.5 * torch.linalg.vector_norm(matching_tensor, ord=2), control_tensor[0][0]

        # use torch.vmap to vectorize the transform function
        vmap_pde_loss_func = torch.vmap(matching_condition_vmap)

        # apply the vectorized transform function on data
        L1s, us = vmap_pde_loss_func(X)

        # get all residual errors over given input dataset
        return L1s, us

    ##################################
    # compute avg PDE Loss with Control Law trajectory to avoid repetitive calculation 
    # PDE loss is just avg PDE loss and should go to 0
    # control law is bounded with absolute value below 60 
    def get_PDE_and_control_loss(self, model, X):
        L1s, us = self.get_PDE_and_control_trajectory(model, X)
        L1_loss = L1s.mean()
        control_loss = bounded_quad_loss(us, dynamics.constants["CONTROL_BOUND"])
        return L1_loss, control_loss

    ##################################
    # condition loss (condition 1 mc_constant is trivial(does not need extra regulation) only adding condition2)
    # (mean across all data points: scalar)
    def get_condition_loss(self, model, X):
        def condition2_vmap(x):
            theta1 = x[0]
            theta2 = x[1]

            def get_ma_hat(x):
                y = model.calculate_M_hat(x)
                return y[0][0]

            def get_mb_hat(x):
                y = model.calculate_M_hat(x)
                return y[0][1]

            dma_hat_dtheata2 = torch.func.grad(lambda x: get_ma_hat(x))(x).reshape(1,4)[:,1]
            dmb_hat_dtheta1 = torch.func.grad(lambda x: get_mb_hat(x))(x).reshape(1,4)[:,0]
            part3 = 2*m*l*b*torch.sin(theta1-theta2)

            return dma_hat_dtheata2 - 2*dmb_hat_dtheta1 + part3

        # use torch.vmap to vectorize the transform function
        vmap_conditon2 = torch.vmap(condition2_vmap)

        # apply the vectorized transform function on data
        condtion2_error = vmap_conditon2(X)

        out = torch.pow(condtion2_error, 2)
        L3_mean = out.mean()
        # return avg loss, normalized avg loss
        return L3_mean, L3_mean/(torch.max(out) + 1e-8)

    ##################################
    # Mdot-2C constraint loss (mean across all data points: scalar)
    def get_dynamics_constraint_loss(self, model, X):
        def func_Mhat(x):
            return model.calculate_M_hat(x)

        def get_C(dMdq, qdot):
            C = torch.stack((qdot, qdot), dim=1).reshape(2,2)

            for k in range(2):
                for j in range(2):
                    C[k,j] = 0
                    for i in range(2):
                        C[k,j] += 0.5 * (dMdq[k,j,i] + dMdq[k,i,j] - dMdq[i,j,k])*qdot[i]
            return C

        # transpose(Mhatdot-2C)+(Mhatdot-2C)
        def skew_symmetric_vmap(x):
            dMdq = torch.func.jacrev(lambda x: func_Mhat(x))(x).reshape(2,2,4)[:,:,:2]
            qdot = torch.stack([x[2], x[3]]).view(-1,1)
            Mhatdot = dMdq @ qdot
            Mhatdot = Mhatdot.reshape(2,2)
            C = get_C(dMdq, qdot)
            mat = Mhatdot - 2*C
            return torch.linalg.matrix_norm(torch.transpose(mat, 0, 1) + mat)


        # use torch.vmap to vectorize the transform function
        vmap_css = torch.vmap(skew_symmetric_vmap)

        # apply the vectorized transform function on data
        out = vmap_css(X)
        L4_mean = out.mean()
        # return avg loss, normalized avg loss
        return L4_mean, L4_mean/(torch.max(out) + 1e-8)

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
    def sparse_sample_loss(self, model, sample_size = 100):
        X = uniform_sampling(n_samples=100)
        loss, _ = self.get_PDE_Loss(model, X)
        return loss

    ##################################
    # TODO: add adaptive sampling

    ##################################
    def total_loss(self, model, X, weights):
        residual_loss, control_loss = self.get_PDE_and_control_loss(model, X)
        boundary_loss = self.get_Boundary_Loss(model)
        deviation_loss = self.get_deviation_loss(model, X, 0.5)
        eig_loss = self.eig_range_loss(model, X, 0.5)
        sparse_loss = self.sparse_sample_loss(model, 100)
        pos_def_loss = self.pos_eig_loss(model, X)

        assert len(weights) == 7
        [W1, W2, W3, W4, W5, W6, W7] = weights

        total =  W1*residual_loss + W2* control_loss + W3*boundary_loss + W4*deviation_loss + W5*eig_loss + W6*sparse_loss + W7*pos_def_loss
        losses = [
            residual_loss.detach().cpu(),
            control_loss.detach().cpu(),
            boundary_loss.detach().cpu(),
            deviation_loss.detach().cpu(),
            eig_loss.detach().cpu(),
            sparse_loss.detach().cpu(),
            pos_def_loss.detach().cpu()
        ]
        return total, losses
