# imports for model
import  torch
from torch import nn
from models.dof4.dynamics import I1z, I2z, Ipz, Mp, Ms, Mt, l1, l2, g, calculate_Mmtx, calculate_Nvect
from core.utils import ResidualLinearNormBlock
from configs.config_dof4 import RESIDUAL_INV, HIDDEN_WIDTH, NUM_DEPTH

########################################################################
k_a = 0.7
k_b = 1.3
k_delta = 0.3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
########################################################################
# rewritten model class to accept both batched an unbatch input

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.INPUT_DIM = 8 
        self.OUTPUT_DIM = 4
        self.HIDDEN_WIDTH = HIDDEN_WIDTH
        self.residual_interval = RESIDUAL_INV
        self.hard_boundary = True
        self.pos_def = True    

        self.model = nn.Sequential(
            nn.Linear(self.INPUT_DIM, self.HIDDEN_WIDTH),
            nn.LayerNorm(self.HIDDEN_WIDTH),
            nn.Tanh(),
            nn.Linear(self.HIDDEN_WIDTH, self.HIDDEN_WIDTH),
            nn.LayerNorm(self.HIDDEN_WIDTH),
            nn.Tanh(),
            nn.Linear(self.HIDDEN_WIDTH,self.HIDDEN_WIDTH),
            nn.LayerNorm(self.HIDDEN_WIDTH),
            nn.Tanh(),
            nn.Linear(self.HIDDEN_WIDTH,self.HIDDEN_WIDTH),
            nn.LayerNorm(self.HIDDEN_WIDTH),
            nn.Tanh(),
            nn.Linear(self.HIDDEN_WIDTH, self.OUTPUT_DIM),
            nn.Tanh()
            )
        self._initialize_weights()

    def forward(self, x):
        """
        Input:
            x: [8] or [n,8]
        Output:
            K: [4,4] or [n,4,4]
        """
        unbatched = False
        if x.ndim == 1:
            x = x.unsqueeze(0)  
            unbatched = True

        N_x = self.model(x)  

        alpha_x = torch.sum(x**2, dim=-1, keepdim=True)  
        alpha_x = (alpha_x > 1e-6).float()  

        out = 1.0 + alpha_x * k_delta * N_x  

        K = torch.diag_embed(out)  # [n,4,4]

        if unbatched:
            K = K.squeeze(0)  # [4,4]

        return K

    def calculate_M(self, X):
        """
        Compute physical mass matrix M(q).
        Input: x shape [8] or [n,8]
        Output: M matrix [4,4] or [n,4,4]
        """

        # if single data, turn into size 1 batch
        unbatched = False
        if X.ndim == 1:
            X = X.unsqueeze(0) 
            unbatched = True

        M_list = [calculate_Mmtx(x, device, I1z, I2z, Ipz, Mp, Ms, Mt, l1, l2) for x in X]
        M = torch.stack(M_list, dim=0) 

        # if unbatched, output accordingly
        if unbatched:
            M = M.squeeze(0)  # [4,4]

        return M

    def calculate_N(self, X):
        """
        Compute physical N(q) vector.
        Input: X shape [8] or [n, 8]
        Output: N vector shape [4, 1] or [n, 4, 1]
        """
        unbatched = False
        if X.ndim == 1:
            X = X.unsqueeze(0)  
            unbatched = True

        N_list = [calculate_Nvect(x, device, Mp, Ms, Mt, l1, l2, g) for x in X]
        N = torch.stack(N_list, dim=0)  # shape [n, 4, 1]

        if unbatched:
            N = N.squeeze(0)  # [4, 1]

        return N

    def calculate_M_hat(self, x):
        """
        Compute shaped mass matrix M_hat(x) = K^T M K
        Input: x shape [8] or [n,8]
        Output: M_hat [4,4] or [n,4,4]
        """
        # if single data, turn into size 1 batch
        unbatched = False
        if x.ndim == 1:
            unbatched = True

        M = self.calculate_M(x)  # [n,8,8] or [8,8]
        K = self.forward(x)       # [n,8,8] or [8,8]

        # construct shaped mass matrix using forward output K and mass matrix M
        M_hat = K.transpose(-2, -1) @ M @ K  # [n,8,8] or [8,8]
        return M_hat


    def _initialize_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return