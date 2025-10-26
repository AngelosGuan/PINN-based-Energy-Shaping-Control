# imports for model
import  torch
import math
from torch import nn
from models.dof4.dynamics import I1z, I2z, Ipz, Mp, Ms, Mt, l1, l2, g, calculate_Mmtx, calculate_Nvect
from configs.config_dof4 import RESIDUAL_INV, HIDDEN_WIDTH, NUM_DEPTH

########################################################################
k_a = 0.7
k_b = 1.3
k_delta = 0.3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
########################################################################

# rewritten model class to accept both batched an unbatch input
class FourierFeatures(nn.Module):
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
            self.out_dim = 2 * in_dim * num_bands

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
            return torch.cat([sin_enc, cos_enc], dim=-1)

class MLP(nn.Module):
    def __init__(self,
                 HIDDEN_WIDTH=256,
                 RESIDUAL_INV=1,
                 use_fourier=True,
                 concat_raw=True,   # keep original x
                 fourier_mapping_size=64,
                 fourier_sigma=1.0,
                 fourier_gaussian=True,
                 fourier_num_bands=None,
                 fourier_logspace=True):
        super().__init__()
        self.INPUT_DIM = 8 
        self.OUTPUT_DIM = 4
        self.HIDDEN_WIDTH = HIDDEN_WIDTH
        self.residual_interval = RESIDUAL_INV
        self.hard_boundary = True
        self.pos_def = True 
        #fourier
        self.use_fourier = use_fourier
        self.concat_raw = concat_raw

        if use_fourier:
            self.fourier = FourierFeatures(
                in_dim=self.INPUT_DIM,
                mapping_size=fourier_mapping_size,
                sigma=fourier_sigma,
                gaussian=fourier_gaussian,
                num_bands=fourier_num_bands,
                logspace=fourier_logspace
            )
            in0 = self.fourier.out_dim + (self.INPUT_DIM if concat_raw else 0)
        else:
            self.fourier = None
            in0 = self.INPUT_DIM


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
            #nn.Tanh()
            )
        self._initialize_weights()

    def encode(self, x):
        """
        Returns the features fed into the backbone.
        Keeps raw x available to the caller for any other heads/constraints.
        """
        if self.use_fourier:
            z = self.fourier(x)
            return torch.cat([x, z], dim=-1) if self.concat_raw else z
        return x

    def forward(self, x):
        unbatched = False
        if x.ndim == 1:
            x = x.unsqueeze(0)
            unbatched = True

        feats = self.encode(x)
        N_x = self.model(feats)  # shape: [n,4]

        # Soft masking based on x-norm
        ep = 1e-6
        x_norm2 = torch.sum(x**2, dim=-1, keepdim=True)
        alpha_x = torch.sigmoid(100.0 * (x_norm2 - ep))  # shape: [n,1]

        # Bounded shaping component (from sigmoid)
        bounded_output = 0.6 * torch.sigmoid(N_x) - 0.3  # ∈ [−0.3, 0.3]
        K_diag = 1.0 + alpha_x * bounded_output          # ∈ [0.7, 1.3] with alpha_x ≈ 1

        K = torch.diag_embed(K_diag)

        if unbatched:
            K = K.squeeze(0)
        return K

    # def project_positive_definite(self, M_hat, λ_min=1e-3):
    #     """
    #     Project shaped mass matrix to ensure it is positive definite
    #     """
    #     eigvals, eigvecs = torch.linalg.eigh(M_hat)
    #     eigvals_clamped = torch.clamp(eigvals, min=λ_min)
    #     M_hat_safe = eigvecs @ torch.diag_embed(eigvals_clamped) @ eigvecs.transpose(-2, -1)
    #     return M_hat_safe


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
        M_hat = K @ M @ K  # [n,8,8] or [8,8]
        # K transpose = K for diagonal matrix
        #M_hat = self.project_positive_definite(M_hat)
        
        return M_hat


    def _initialize_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return