# imports for model
import  torch
from torch import nn
# import dynamic constants
from models.dof8.dynamics import Mp, Mf, g, l1, l2, la, lf, I1x, I2x, Ipx, slope, M1, M2
# import dynamic function for contact condition, M, N
from models.dof8.dynamics import determine_phase_masks, calculate_Mmtx, calculate_Nvect
from configs.config_dof8 import EPSILON, RESIDUAL_INV, HIDDEN_WIDTH
from core.utils import ResidualLinearNormBlock



########################################################################
k_a = 0.9
k_b = 1.1
k_delta = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
########################################################################
# rewritten model class to accept both batched an unbatch input

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.INPUT_DIM = 16
        self.OUTPUT_DIM = 8
        self.HIDDEN_WIDTH = HIDDEN_WIDTH
        self.residual_interval = RESIDUAL_INV
        self.hard_boundary = False
        self.pos_def = True  

        self.common = ResidualLinearNormBlock(
            input_dim=self.INPUT_DIM,
            hidden_dim=self.HIDDEN_WIDTH,
            num_repeats=8,
            output_dim=self.HIDDEN_WIDTH,
            residual_interval=2)

        self.branch1 = ResidualLinearNormBlock(
            input_dim=self.HIDDEN_WIDTH,
            hidden_dim=self.HIDDEN_WIDTH,
            num_repeats=8,
            output_dim=self.OUTPUT_DIM,
            residual_interval=2)

        self.branch2 = ResidualLinearNormBlock(
            input_dim=self.HIDDEN_WIDTH,
            hidden_dim=self.HIDDEN_WIDTH,
            num_repeats=8,
            output_dim=self.OUTPUT_DIM,
            residual_interval=2)
        self.branch3 = ResidualLinearNormBlock(
            input_dim=self.HIDDEN_WIDTH,
            hidden_dim=self.HIDDEN_WIDTH,
            num_repeats=8,
            output_dim=self.OUTPUT_DIM,
            residual_interval=2)

        self._initialize_weights()

    def forward(self, x):
        """
        Input: x shape [4] or [n, 4]
        Output: K matrix shape [2,2] or [n,2,2] (diagonal matrix or batch of diagonal matrices)
        """

        # if single data, turn into size 1 batch
        unbatched = False
        if x.ndim == 1:
            x = x.unsqueeze(0) # [4,] -> [1,4]
            unbatched = True


        phase1_mask, phase2_mask, phase3_mask = determine_phase_masks(x, slope, lf, EPSILON, device)
        out = self.common(x)
        out = (
            phase1_mask * self.branch1(out)
            + phase2_mask * self.branch2(out)
            + phase3_mask * self.branch3(out)
        )


        alpha_x = torch.sum(x**2, dim=-1, keepdim=True)  # [n,1]
        alpha_x = (alpha_x > 1e-6).float()  # [n,1]

        out = 1.0 + alpha_x * k_delta * out # [n,8]

        # change into diagonal matrix
        K = torch.diag_embed(out)  # [n,8,8]

        # if unbatched, output accordingly
        if unbatched:
            K = K.squeeze(0) # [8,8]

        return K

    def calculate_M(self, X):
        """
        Compute physical mass matrix M(q).
        Input: x shape [4] or [n,4]
        Output: M matrix [2,2] or [n,2,2]
        """

        # if single data, turn into size 1 batch
        unbatched = False
        if X.ndim == 1:
            X = X.unsqueeze(0) # [4,] -> [1,4]
            unbatched = True

        M_list = [calculate_Mmtx(x, device, Mp, Mf, g, l1, l2, la, lf, I1x, I2x, Ipx, slope, M1, M2) for x in X]
        M = torch.stack(M_list, dim=0) # [n,8,8]

        # if unbatched, output accordingly
        if unbatched:
            M = M.squeeze(0)  # [8,8]

        return M

    def calculate_N(self, X):
        """
        Compute physical N(q) vector.
        Input: X shape [8] or [n, 8]
        Output: N vector shape [8, 1] or [n, 8, 1]
        """
        unbatched = False
        if X.ndim == 1:
            X = X.unsqueeze(0)  # [8] -> [1, 8]
            unbatched = True

        N_list = [calculate_Nvect(x, device, Mp, Mf, g, l1, l2, la, lf, I1x, I2x, Ipx, slope, M1, M2) for x in X]
        N = torch.stack(N_list, dim=0)  # shape [n, 8, 1]

        if unbatched:
            N = N.squeeze(0)  # [8, 1]

        return N

    def calculate_M_hat(self, x):
        """
        Compute shaped mass matrix M_hat(x) = K^T M K
        Input: x shape [4] or [n,4]
        Output: M_hat [2,2] or [n,2,2]
        """

        # if single data, turn into size 1 batch
        unbatched = False
        if x.ndim == 1:
            unbatched = True

        M = self.calculate_M(x)  # [n,8,8] or [8,8]
        K = self.forward(x)       # [n,8,8] or [8,8]

        # construct shaped mass matrix using forward output K and mass matrix M
        M_hat =K.transpose(-2, -1) @ M @ K  # [n,8,8] or [8,8]
        return M_hat


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                # Use Xavier initialization for Tanh-compatible layers
                torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('tanh'))
                torch.nn.init.zeros_(m.bias)

                # Force final layer to output ~0 → ensure K ≈ I at init
                if m.out_features == self.OUTPUT_DIM:
                    torch.nn.init.zeros_(m.weight)