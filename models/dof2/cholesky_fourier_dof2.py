# imports for model
import  torch
import math
from torch import nn
import models.dof2.dynamics as dynamics
from configs.config_dof2 import HIDDEN_WIDTH
########################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
########################################################################

# rewritten model class to accept both batched an unbatch input
class FourierFeatures(nn.Module):
    """
    Deterministic Fourier/polynomial basis for 2D input x = [q1, q2].

    Accepts:
      - unbatched: (2,) or (2,1) or (1,2)
      - batched:   (N,2) or (N,2,1)

    Returns (in this exact order):
      [ sin(q1), cos(q1), sin(q2), cos(q2),
        sin^2(q1), cos^2(q1), sin^2(q2), cos^2(q2),
        sin^3(q1), cos^3(q1), sin^3(q2), cos^3(q2),
        q1, q2 ]
    """
    def __init__(self):
        super().__init__()
        self.in_dim = 4
        self.out_dim = 28 # update this if change basis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        unbatched = False
        if x.ndim == 1:
            x = x.unsqueeze(0)
            unbatched = True

        q1 = x[:, 0]
        q2 = x[:, 1]

        s1 = torch.sin(q1)
        c1 = torch.cos(q1)
        s2 = torch.sin(q2)
        c2 = torch.cos(q2)

        basis = torch.cat(
            [
                s1, c1, s2, c2,
                s1 * s1, c1 * c1, s2 * s2, c2 * c2,
                s1 * s1 * s1, c1 * c1 * c1, s2 * s2 * s2, c2 * c2 * c2,
                q1, q2, 1/s1, 1/s2, 1/c1, 1/c2, s1/c1, s2/c2, c1/s1, c2/s2,
                1/(s1*s1), 1/(s2*s2), 1/(c1*c1), 1/(c2*c2), torch.log(1/c1+s1/c1),torch.log(1/c2+s2/c2)
            ],
            dim=-1,
        )  # (N,26)

        if unbatched:
            return basis.squeeze(0)  # (26,)
        return basis  # (N,26)

# TODO: divide model into two subnetwork
class MdNet(nn.Module):
    def __init__(self):  # out_dim depends on how you parameterize Md
        super().__init__()
        self.fourier = FourierFeatures()
        in0 = self.fourier.out_dim 
        self.OUTPUT_DIM = 3
        self.model = nn.Sequential(
            nn.Linear(in0, self.OUTPUT_DIM),
            )
        self._initialize_weights()

    def forward(self, x):
        unbatched = False
        if x.ndim == 1:
            x = x.unsqueeze(0)
            unbatched = True

        feats = self.fourier(x)
        out = self.model(feats)  

        if unbatched:
            out = out.squeeze(0)
        return out

    def _initialize_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return

class VdNet(nn.Module):
    def __init__(self):  # out_dim depends on how you parameterize Md
        super().__init__()
        self.fourier = FourierFeatures()
        in0 = self.fourier.out_dim 
        self.OUTPUT_DIM  = 1
        self.model = nn.Sequential(
            nn.Linear(in0, HIDDEN_WIDTH),
            nn.Tanh(),
            nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH),
            nn.Tanh(),
            nn.Linear(HIDDEN_WIDTH, self.OUTPUT_DIM)
            )
        self._initialize_weights()

    def forward(self, x):
        unbatched = False
        if x.ndim == 1:
            x = x.unsqueeze(0)
            unbatched = True

        feats = self.fourier(x)
        out = self.model(feats) 

        if unbatched:
            out = out.squeeze(0)
        return out

    def _initialize_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return
# just linear regression
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.INPUT_DIM = 4
        self.OUTPUT_DIM = 4
        self.hard_boundary = False # boundary: 
        self.pos_def = True 


        self.md = MdNet()
        self.vd = VdNet()


    def forward(self, x):
        return self.md(x), self.vd(x)


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

        M_list = [dynamics.calculate_M(x) for x in X]
        M = torch.stack(M_list, dim=0) 

        # if unbatched, output accordingly
        if unbatched:
            M = M.squeeze(0)  # [4,4]

        return M

    def calculate_Vd(self, X):
        """
        Compute physical N(q) vector.
        Input: X shape [8] or [n, 8]
        Output: N vector shape [4, 1] or [n, 4, 1]
        """
        return self.vd(X)

    def calculate_Md_hat(self, X):
        """
        Compute shaped mass matrix M_hat(x) = K^T M K
        Input: x shape [8] or [n,8]
        Output: M_hat [4,4] or [n,4,4]
        """
        # if single data, turn into size 1 batch
        out= self.md(X)
        unbatched = False
        if X.ndim == 1:
            out = out.unsqueeze(0) 
            unbatched = True

        n = out.size(0)
        L = torch.zeros((n, 2, 2), device=X.device, dtype=X.dtype)
        L[:, 0, 0] = out[:, 0]   # o1
        L[:, 1, 0] = out[:, 1]   # o2
        L[:, 1, 1] = out[:, 2]   # o3

        # M =LL^T+eps*I
        eps = 1e-6
        Md_hat = L @ L.transpose(-1,-2) + eps*torch.eye(2, device=X.device, dtype=X.dtype).unsqueeze(0)
        
        if unbatched:
            Md_hat = Md_hat.squeeze(0)  # [2,2]
        return Md_hat


    def _initialize_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return