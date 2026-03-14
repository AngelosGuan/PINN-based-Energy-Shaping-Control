# imports for model
import  torch
import math
from torch import nn
from models.dof3.dynamics import calculate_Mmtx, calculate_Nvect

########################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
########################################################################
class FourierFeatures(nn.Module):
    """
    Deterministic Fourier/polynomial basis for 3dof input X

    Accepts:
      - unbatched: (6,) 
      - batched:   (N,6) 

    """
    def __init__(self):
        super().__init__()
        self.in_dim = 6
        self.out_dim = 21 # update this if change basis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        unbatched = False
        if x.ndim == 1:
            x = x.unsqueeze(0)
            unbatched = True

        q1 = x[:, 0]
        q2 = x[:, 1]
        q3 = x[:, 2]

        s1 = torch.sin(q1)
        c1 = torch.cos(q1)
        s2 = torch.sin(q2)
        c2 = torch.cos(q2)
        s3 = torch.sin(q3)
        c3 = torch.cos(q3)


        basis = torch.stack(
            [
                s1, c1, s2, c2, s3, c3,
                s1 * s1, c1 * c1, s2 * s2, c2 * c2, s3 * s3, c3 * c3,
                s1 ** 3, c1 ** 3, s2 ** 3, c2 ** 3, s3 ** 3, c3 ** 3,
                q1, q2, q3
            ],
            dim=-1,
        )  # (N, 28)

        if unbatched:
            return basis.squeeze(0)  
        return basis  

# # rewritten model class to accept both batched an unbatch input
# class FourierFeatures(nn.Module):
#     """
#     Random Fourier features (Rahimi & Recht, 2007).
#     Maps x in R^d to [sin(2π B x), cos(2π B x)] in R^{2m}.
#     B ~ N(0, sigma^2) if gaussian=True, otherwise use fixed frequency bands (NeRF/Tancik style).
#     """
#     def __init__(self, in_dim, mapping_size=64, sigma=1.0, gaussian=True, num_bands=None, logspace=True):
#         super().__init__()
#         self.in_dim = in_dim
#         self.gaussian = gaussian

#         if gaussian:
#             # B: (m, d); fixed by default so it moves with device but doesn't train
#             B = torch.randn(mapping_size, in_dim) * sigma
#             self.register_buffer("B", B)
#             self.out_dim = 2 * mapping_size
#         else:
#             # NeRF/Tancik positional encoding style per-dimension frequency bands
#             # num_bands controls how many frequencies per input dim
#             assert num_bands is not None and num_bands > 0
#             if logspace:
#                 # [1, 2, 4, ..., 2^{num_bands-1}]
#                 freq_bands = 2.0 ** torch.arange(num_bands)
#             else:
#                 # linearly spaced in [1, 2^{num_bands-1}]
#                 freq_bands = torch.linspace(1.0, 2.0 ** (num_bands - 1), num_bands)
#             self.register_buffer("freq_bands", freq_bands)
#             self.out_dim = 2 * in_dim * num_bands

#     def forward(self, x):
#         # x: (N, d)
#         if self.gaussian:
#             # (N, m)
#             proj = 2.0 * math.pi * (x @ self.B.t())
#             return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
#         else:
#             # Broadcast: for each dim apply all bands
#             # x_expanded: (N, d, 1) * (bands,) -> (N, d, bands)
#             x_expanded = x.unsqueeze(-1) * self.freq_bands  # (N, d, K)
#             x_expanded = 2.0 * math.pi * x_expanded
#             sin_enc = torch.sin(x_expanded)
#             cos_enc = torch.cos(x_expanded)
#             # flatten last two dims: (N, d*K)
#             sin_enc = sin_enc.reshape(x.shape[0], -1)
#             cos_enc = cos_enc.reshape(x.shape[0], -1)
#             return torch.cat([sin_enc, cos_enc], dim=-1)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.INPUT_DIM = 6
        self.OUTPUT_DIM = 3
        self.fourier = FourierFeatures()

        in0 = self.fourier.out_dim 

        self.model = nn.Sequential(
            nn.Linear(in0, 32, bias=False),
            nn.Tanh(),
            nn.Linear(32, 32, bias=False),
            nn.Tanh(),
            nn.Linear(32, 32, bias=False),
            nn.Tanh(),
            nn.Linear(32, self.OUTPUT_DIM, bias=False),
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

    def calculate_M(self, X):
        """
        Compute physical mass matrix M(q).
        Input: x shape [8] or [n,8]
        Output: M matrix [4,4] or [n,4,4]
        """
        return calculate_Mmtx(X)

    def calculate_N(self, X):
        """
        Compute physical N(q) vector.
        Input: X shape [8] or [n, 8]
        Output: N vector shape [4, 1] or [n, 4, 1]
        """
        return calculate_Nvect(X)

    def calculate_M_hat(self, X):
        """
        Compute shaped mass matrix M_hat(x) = K^T M K
        Input: x shape [8] or [n,8]
        Output: M_hat [4,4] or [n,4,4]
        """
        # if single data, turn into size 1 batch
        M = self.calculate_M(X)      # (3,3) or (B,3,3)
        out =  torch.abs(self.forward(X))        # (3,) or (B,3)

        if X.ndim == 1:
            D = torch.diag(out)      # (3,3)
        else:
            D = torch.diag_embed(out)  # (B,3,3)

        M_hat = M + D
        return M_hat


    def _initialize_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return