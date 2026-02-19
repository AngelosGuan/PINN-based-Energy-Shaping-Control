# imports for model
import  torch
import math
from torch import nn
import models.dof2.dynamics as dynamics
from core.utils import assert_finite
########################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
########################################################################

# rewritten model class to accept both batched an unbatch input
class FourierFeatures_Md(nn.Module):
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
        self.out_dim = 14 # update this if change basis

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

        basis = torch.stack(
            [
                s1, c1, s2, c2,
                s1 * s1, c1 * c1, s2 * s2, c2 * c2,
                s1 ** 3, c1 ** 3, s2 ** 3, c2 ** 3,
                q1, q2
            ],
            dim=-1,
        )  # (N, 28)

        if unbatched:
            return basis.squeeze(0)  # (28,)
        return basis  # (N, 28)


# rewritten model class to accept both batched an unbatch input
class FourierFeatures_Vd(nn.Module):
    """
    Random Fourier features (Rahimi & Recht, 2007).
    Maps x in R^d to [sin(2π B x), cos(2π B x)] in R^{2m}.
    B ~ N(0, sigma^2) if gaussian=True, otherwise use fixed frequency bands (NeRF/Tancik style).
    """
    def __init__(self, in_dim, mapping_size=64, sigma=1.0, gaussian=True, num_bands=None, logspace=True):
        super().__init__()
        self.in_dim = in_dim
        self.gaussian = gaussian

        if gaussian:
            # B: (m, d); fixed by default so it moves with device but doesn't train
            B = torch.randn(mapping_size, in_dim) * sigma
            self.register_buffer("B", B)
            self.out_dim = 2 * mapping_size
        else:
            # NeRF/Tancik positional encoding style per-dimension frequency bands
            # num_bands controls how many frequencies per input dim
            assert num_bands is not None and num_bands > 0
            if logspace:
                # [1, 2, 4, ..., 2^{num_bands-1}]
                freq_bands = 2.0 ** torch.arange(num_bands)
            else:
                # linearly spaced in [1, 2^{num_bands-1}]
                freq_bands = torch.linspace(1.0, 2.0 ** (num_bands - 1), num_bands)
            self.register_buffer("freq_bands", freq_bands)
            self.out_dim = 2 * in_dim * num_bands + in_dim

    def forward(self, x):
        # x: (N, d)
        if self.gaussian:
            # (N, m)
            proj = 2.0 * math.pi * (x @ self.B.t())
            return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        else:
            # Broadcast: for each dim apply all bands
            # x_expanded: (N, d, 1) * (bands,) -> (N, d, bands)
            x_expanded = x.unsqueeze(-1) * self.freq_bands  # (N, d, K)
            x_expanded = 2.0 * math.pi * x_expanded
            sin_enc = torch.sin(x_expanded)
            cos_enc = torch.cos(x_expanded)
            # flatten last two dims: (N, d*K)
            sin_enc = sin_enc.reshape(x.shape[0], -1)
            cos_enc = cos_enc.reshape(x.shape[0], -1)
            return torch.cat([sin_enc, cos_enc, x], dim=-1)

# TODO: divide model into two subnetwork
class MdNet(nn.Module):
    def __init__(self):  # out_dim depends on how you parameterize Md
        super().__init__()
        self.fourier = FourierFeatures_Md()
        in0 = self.fourier.out_dim 
        self.OUTPUT_DIM = 3
        self.HIDDEN_WIDTH = 16 # change here
        self.model = nn.Sequential(
            nn.Linear(in0, self.HIDDEN_WIDTH),
            nn.LayerNorm(self.HIDDEN_WIDTH),
            nn.Tanh(),
            nn.Linear(self.HIDDEN_WIDTH, self.HIDDEN_WIDTH),
            nn.LayerNorm(self.HIDDEN_WIDTH),
            nn.Tanh(),
            nn.Linear(self.HIDDEN_WIDTH, self.OUTPUT_DIM),
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
        self.fourier = FourierFeatures_Vd(in_dim=4,
                mapping_size=64,
                sigma=1.0,
                gaussian=True,
                num_bands=None,
                logspace=True)  # config here
        in0 = self.fourier.out_dim 
        self.OUTPUT_DIM  = 1
        self.HIDDEN_WIDTH = 16
        self.model = nn.Sequential(
            nn.Linear(in0, self.HIDDEN_WIDTH),
            nn.LayerNorm(self.HIDDEN_WIDTH),
            nn.Tanh(),
            nn.Linear(self.HIDDEN_WIDTH, self.HIDDEN_WIDTH),
            nn.LayerNorm(self.HIDDEN_WIDTH),
            nn.Tanh(),
            nn.Linear(self.HIDDEN_WIDTH,self.HIDDEN_WIDTH),
            nn.LayerNorm(self.HIDDEN_WIDTH),
            nn.Tanh(),
            nn.Linear(self.HIDDEN_WIDTH, self.OUTPUT_DIM),
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

        #assert_finite('NN out', out)
        
        n = out.size(0)
        o1, o2, o3 = out[:, 0], out[:, 1], out[:, 2]
        o1_p = torch.nn.functional.softplus(o1) + 1e-3
        o3_p = torch.nn.functional.softplus(o3) + 1e-3
        
        L = torch.stack(
        [
            torch.stack([o1_p, torch.zeros_like(o1, device = X.device)], dim=-1),
            torch.stack([o2, o3_p],    dim=-1),
        ],dim=-2)  # (n,2,2)

        #assert_finite('L', L)


        Md_hat = L @ L.transpose(-1,-2)
        
        if unbatched:
            Md_hat = Md_hat.squeeze(0)  # [2,2]

        #assert_finite('Md_hat', Md_hat)
        return Md_hat


    def _initialize_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return