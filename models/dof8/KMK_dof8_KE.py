# imports for model
import  torch
from torch import nn
# import dynamic constants
from models.dof8.dynamics import Mp, Mf, g, l1, l2, la, lf, I1x, I2x, Ipx, slope, M1, M2
# import dynamic function for contact condition, M, N
from models.dof8.dynamics import determine_phase_masks, calculate_Mmtx, calculate_Nvect
from configs.config_dof8 import EPSILON



########################################################################
k_a = 0.7
k_b = 1.3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
########################################################################
# rewritten model class to accept both batched an unbatch input

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.INPUT_DIM = 16
        self.OUTPUT_DIM = 8
        self.HIDDEN_WIDTH = 8196
        self.hard_boundary = False
        self.pos_def = True  

        self.common = make_linear_norm_block(input_dim = self.INPUT_DIM, hidden_dim = self.HIDDEN_WIDTH, num_repeats=8, output_dim=self.HIDDEN_WIDTH)
        self.branch1 = make_linear_norm_block(input_dim = self.HIDDEN_WIDTH, hidden_dim = self.HIDDEN_WIDTH, num_repeats=8, output_dim=self.OUTPUT_DIM)
        self.branch2 = make_linear_norm_block(input_dim = self.HIDDEN_WIDTH, hidden_dim = self.HIDDEN_WIDTH, num_repeats=8, output_dim=self.OUTPUT_DIM)
        self.branch3 = make_linear_norm_block(input_dim = self.HIDDEN_WIDTH, hidden_dim = self.HIDDEN_WIDTH, num_repeats=8, output_dim=self.OUTPUT_DIM)
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

        # scale input from tanh output to [k_a, k_b] range
        out = k_a + (k_b - k_a) * 0.5 * (out+1.0)

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
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='tanh')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)