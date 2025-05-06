# imports for model
import  torch
from torch import nn
from models.dof2 import dynamics

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

        self.C_ma_tensor = torch.tensor(dynamics.constants["C_ma"], device=device, requires_grad=False)
        self.C_mb_tensor = torch.tensor(dynamics.constants["C_mb"], device=device, requires_grad=False)
        self.C_mc_tensor = torch.tensor(dynamics.constants["C_mc"], device=device, requires_grad=False)
        self.INPUT_DIM = 4 
        self.OUTPUT_DIM = 2
        self.hard_boundary = True
        self.pos_def = True    

        self.model = nn.Sequential(
            nn.Linear(self.INPUT_DIM, 1024),
            nn.LayerNorm(1024),
            nn.Tanh(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.Tanh(),
            nn.Linear(1024,1024),
            nn.LayerNorm(1024),
            nn.Tanh(),
            nn.Linear(1024, self.OUTPUT_DIM),
            nn.Tanh()
            )
        self._initialize_weights()

    def forward(self, x):
        """
        Input:
            x: [4] or [n,4]
        Output:
            K: [2,2] or [n,2,2]
        """
        unbatched = False
        if x.ndim == 1:
            x = x.unsqueeze(0)  # [1, 4]
            unbatched = True

        N_x = self.model(x)  # [n, 2]

        alpha_x = torch.sum(x**2, dim=-1, keepdim=True)  # [n,1]
        alpha_x = (alpha_x > 1e-6).float()  # [n,1]

        out = 1.0 + alpha_x * k_delta * N_x  # [n,2]

        K = torch.diag_embed(out)  # [n,2,2]

        if unbatched:
            K = K.squeeze(0)  # [2,2]

        return K

    def calculate_M(self, x):
        """
        Compute physical mass matrix M(q).
        Input: x shape [4] or [n,4]
        Output: M matrix [2,2] or [n,2,2]
        """

        # if single data, turn into size 1 batch
        unbatched = False
        if x.ndim == 1:
            x = x.unsqueeze(0) # [4,] -> [1,4]
            unbatched = True

        # get batched cosine term cos(theta1-theta2)
        cos_term = torch.cos(x[:,0] - x[:,1]).to(device) # [n]

        # get batched M: [n,2,2]
        M = torch.stack([
            torch.stack([
                self.C_ma_tensor.expand_as(cos_term),
                self.C_mb_tensor * cos_term
                ], dim=-1),
            torch.stack([
                self.C_mb_tensor * cos_term,
                self.C_mc_tensor.expand_as(cos_term)
                ], dim=-1)
            ], dim=-2)

        # if unbatched, output accordingly
        if unbatched:
            M = M.squeeze(0)

        return M

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

        M = self.calculate_M(x)  # [n,2,2] or [2,2]
        K = self.forward(x)       # [n,2,2] or [2,2]

        # construct shaped mass matrix using forward output K and mass matrix M
        M_hat =K.transpose(-2, -1) @ M @ K  # [n,2,2] or [2,2]
        return M_hat


    def _initialize_weights(self):
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='tanh')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)