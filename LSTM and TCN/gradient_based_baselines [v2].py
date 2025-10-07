# ============================================================
#  Causal TCN baseline for 3-D Lorenz forecasting (PyTorch)
# ============================================================
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

class TCNBaseline3D(nn.Module):
    """
    2-layer causal TCN       (kernel=3, dilation=1 & 2, padding chosen
    so receptive field = 5 time-steps, identical to NVAR window length).
    ----------------------
    • input_dim  = 3
    • hidden_dim = 32  → total ≈ 4.9 k parameters
    • output_dim = 3    (one-step prediction)
    """
    def __init__(self,
                 input_dim:  int = 3,
                 hidden_dim: int = 32,
                 output_dim: int = 3,
                 lr: float = 1e-3,
                 epochs: int = 40,
                 device: str = "cpu",
                 seed: int = 0):
        super().__init__()
        torch.manual_seed(seed); np.random.seed(seed)

        k = 3  # kernel
        # layer 1: dilation 1  → pad 2 to keep length
        self.conv1 = nn.Conv1d(input_dim, hidden_dim,
                               kernel_size=k,
                               dilation=1,
                               padding=2,
                               bias=True)
        # layer 2: dilation 2  → pad 4
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim,
                               kernel_size=k,
                               dilation=2,
                               padding=4,
                               bias=True)
        self.relu  = nn.ReLU()
        self.head  = nn.Conv1d(hidden_dim, output_dim,
                               kernel_size=1, bias=True)

        self.lr, self.epochs = lr, epochs
        self.to(device)
        self.optim = Adam(self.parameters(), lr=lr)
        self.crit  = nn.MSELoss()

    # ---------------------------------------------------------
    def forward(self, x):
        """
        x shape  [B, T, 3]  (batch, time, channels)
        return  [B, T, 3]
        """
        # reshape to Conv1d convention: (B, C, T)
        x = x.permute(0, 2, 1)
        y = self.conv1(x); y = self.relu(y[:, :, :-2])     # remove look-ahead pad
        y = self.conv2(y); y = self.relu(y[:, :, :-4])     # remove look-ahead pad
        out = self.head(y).permute(0, 2, 1)                # back to (B,T,C)
        return out

    # ---------------------------------------------------------
    def fit(self, x_np: np.ndarray, y_np: np.ndarray):
        """
        Teacher-forcing on entire sequence (batch size = 1).
        x_np, y_np shape [T, 3]
        """
        x = torch.tensor(x_np[None], dtype=torch.float32, device=next(self.parameters()).device)
        y = torch.tensor(y_np[None], dtype=torch.float32, device=next(self.parameters()).device)

        for _ in range(self.epochs):
            self.optim.zero_grad()
            pred = self.forward(x)
            loss = self.crit(pred[:, :-1], y[:, 1:])  # predict next step
            loss.backward()
            self.optim.step()

    # ---------------------------------------------------------
    @torch.no_grad()
    def predict(self, init_window: np.ndarray, n_steps: int):
        """
        Autoregressive roll-out.
        init_window : length L≥5, shape [L,3] (latest samples, earliest first)
        Returns      : [n_steps,3]
        """
        device = next(self.parameters()).device
        window = init_window.copy()
        preds  = np.empty((n_steps, 3), dtype=np.float32)

        for t in range(n_steps):
            inp = torch.tensor(window[None], dtype=torch.float32, device=device)
            y   = self.forward(inp)[0, -1].cpu().numpy()
            preds[t] = y
            window   = np.vstack([window[1:], y])  # slide window

        return preds

# ============================================================
#  Causal Transformer baseline for 3-D Lorenz forecasting
#  (PyTorch ≥ 1.9)
# ============================================================
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

class SmallCausalTransformer3D(nn.Module):
    """
    Single-layer causal Transformer:
      • d_model = 24,   nhead = 1,   d_ff = 4·d_model
      • receptive field  = sequence length L (set in fit / predict)
      • total parameters ≈ 4 900
    """
    def __init__(self,
                 d_model: int = 24,
                 nhead: int = 1,
                 d_ff: int = 96,        # 4 × d_model
                 lr: float = 2e-3,
                 epochs: int = 60,
                 device: str = "cpu",
                 seed: int = 0):
        super().__init__()
        torch.manual_seed(seed); np.random.seed(seed)
        self.device, self.epochs = device, epochs

        self.in_proj   = nn.Linear(3, d_model)     # 3-dim input → tokens
        encoder_layer  = nn.TransformerEncoderLayer(
                             d_model=d_model,
                             nhead=nhead,
                             dim_feedforward=d_ff,
                             batch_first=True,
                             activation="gelu",
                             norm_first=True)
        self.encoder   = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.pos_embed = None                      # built on first call
        self.head      = nn.Linear(d_model, 3)     # back to 3-dim output

        self.to(device)
        self.opt  = Adam(self.parameters(), lr=lr)
        self.crit = nn.MSELoss()

    # ----------------------------------------
    def _get_posembed(self, L: int, d: int):
        """Fixed sinusoidal positional embedding (same as Vaswani et al.)."""
        pos = torch.arange(L, dtype=torch.float32, device=self.device)
        i   = torch.arange(d//2, dtype=torch.float32, device=self.device)
        angles = pos[:, None] / (10000 ** (2*i/d))
        pe = torch.zeros(L, d, device=self.device)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)
        return pe[None]                                # shape (1,L,d)

    # ----------------------------------------
    def fit(self, x_np: np.ndarray, y_np: np.ndarray, L: int = 20):
        """
        Teacher-forcing with sliding windows of length L.
        x_np, y_np  shape [T, 3];  y_np[t] is the desired prediction for x_np[t].
        """
        x = torch.tensor(x_np, dtype=torch.float32, device=self.device)
        y = torch.tensor(y_np, dtype=torch.float32, device=self.device)

        if self.pos_embed is None or self.pos_embed.size(1) != L:
            self.pos_embed = self._get_posembed(L, self.in_proj.out_features)

        # build training batches as overlapping windows (stride 1)
        windows   = x.unfold(0, L, 1)        # shape [T-L+1, L, 3]
        targets   = y[L-1:]                  # predict the last step
        dataset   = torch.utils.data.TensorDataset(windows, targets)
        loader    = torch.utils.data.DataLoader(dataset,
                                                batch_size=64,
                                                shuffle=True)

        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                self.opt.zero_grad()
                z   = self.in_proj(batch_x) + self.pos_embed
                out = self.encoder(z)
                pred = self.head(out[:, -1])          # last token
                loss = self.crit(pred, batch_y)
                loss.backward(); self.opt.step()

    # ----------------------------------------
    @torch.no_grad()
    def predict(self, init_window: np.ndarray, n_steps: int):
        """
        Autoregressive roll-out.
        init_window : numpy (L,3)  – most recent L samples (old → new)
        Returns      : numpy (n_steps,3)
        """
        L = init_window.shape[0]
        if self.pos_embed is None or self.pos_embed.size(1) != L:
            self.pos_embed = self._get_posembed(L, self.in_proj.out_features)

        window = torch.tensor(init_window, dtype=torch.float32,
                              device=self.device)
        preds  = np.empty((n_steps, 3), dtype=np.float32)

        for t in range(n_steps):
            z   = self.in_proj(window[None]) + self.pos_embed
            y   = self.head(self.encoder(z)[:, -1])[0]
            preds[t] = y.cpu().numpy()

            window = torch.vstack([window[1:], y])

        return preds

# ============================================================
#  Non-linear Vector Auto-Regression (NVAR) baseline for 3-D Lorenz
# ============================================================
import numpy as np
from itertools import combinations_with_replacement
from sklearn.linear_model import Ridge

class NVARBaseline3D:
    """
    • delay window length k   (default 5 samples)
    • quadratic polynomial lift (all monomials up to degree 2)
    • closed-form ridge regression read-out
    """
    def __init__(self,
                 k: int = 5,
                 ridge_alpha: float = 1e-4):
        self.k          = k
        self.alpha      = ridge_alpha
        self.scaler_mu  = None
        self.scaler_sig = None
        self.reg        = Ridge(alpha=self.alpha, fit_intercept=False)

        # indices for quadratic terms
        L  = 3 * k                 # length of flattened delay vector
        self.idxs_quad = list(combinations_with_replacement(range(L), 2))

    # ---------------------------------------------------------
    def _build_feature(self, window: np.ndarray) -> np.ndarray:
        """
        window: shape (k, 3)  -> returns (F,) where
          F = 1 + 3k + (3k)(3k+1)/2
        """
        lin = window.flatten()                 # linear terms
        quad = np.array([lin[i]*lin[j] for i, j in self.idxs_quad])
        return np.concatenate(([1.0], lin, quad), dtype=np.float32)

    # ---------------------------------------------------------
    def fit(self, x_np: np.ndarray, y_np: np.ndarray):
        """
        x_np shape [T, 3] (driver)
        y_np shape [T, 3] (target 1-step ahead)
        Assumes x_np[t] predicts y_np[t].
        """
        k = self.k
        assert len(x_np) == len(y_np)
        # normalise inputs
        self.scaler_mu  = x_np.mean(0, keepdims=True)
        self.scaler_sig = x_np.std (0, keepdims=True) + 1e-9
        x_norm = (x_np - self.scaler_mu)/self.scaler_sig

        feats, targets = [], []
        for t in range(k, len(x_norm)):
            window = x_norm[t-k:t]              # shape (k,3)
            feats.append(self._build_feature(window))
            targets.append(y_np[t])

        X = np.vstack(feats)
        Y = np.vstack(targets)
        self.reg.fit(X, Y)

    # ---------------------------------------------------------
    def predict(self, init_window: np.ndarray, n_steps: int):
        """
        Autoregressive roll-out.
        init_window : array (k,3)  – most recent k inputs (y-values).
        Returns array (n_steps,3)
        """
        k = self.k
        window = init_window.copy()
        preds  = np.empty((n_steps, 3), dtype=np.float32)

        for t in range(n_steps):
            w_norm  = (window - self.scaler_mu)/self.scaler_sig
            phi     = self._build_feature(w_norm)
            y_hat   = self.reg.predict(phi[None, :])[0]
            preds[t] = y_hat
            # slide window: drop oldest, append new prediction
            window = np.vstack([window[1:], y_hat])

        return preds

# ============================================================
#  LSTM baseline with ~4.8 k parameters  (PyTorch ≥ 1.9)
# ============================================================
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

class LSTMBaseline3D:
    """
    Lightweight single-layer LSTM for 3-dim Lorenz forecasting.
    * hidden_size=32 → ~4.8k trainable parameters
    * fit() trains in teacher-forcing mode
    * predict() produces autoregressive roll-out
    """

    def __init__(self,
                 input_dim:  int = 3,
                 hidden_size: int = 32,
                 output_dim: int = 3,
                 lr: float = 1e-3,
                 epochs: int = 30,
                 device: str = 'cpu',
                 seed: int = 0):
        torch.manual_seed(seed); np.random.seed(seed)

        self.device  = torch.device(device)
        self.epochs  = epochs
        self.model   = nn.LSTM(input_dim, hidden_size,
                               batch_first=True).to(self.device)
        self.head    = nn.Linear(hidden_size, output_dim).to(self.device)
        self.crit    = nn.MSELoss()
        self.optim   = Adam(list(self.model.parameters())+
                            list(self.head.parameters()), lr=lr)

    # ---------------------------------------------------------
    @torch.no_grad()
    def _init_hidden(self, batch_sz=1):
        h0 = torch.zeros(1, batch_sz,
                         self.model.hidden_size,
                         device=self.device)
        c0 = torch.zeros_like(h0)
        return (h0, c0)

    # ---------------------------------------------------------
    def fit(self, x_np: np.ndarray, y_np: np.ndarray):
        """
        x_np shape [T, 3]  (input  at t)
        y_np shape [T, 3]  (target at t)
        """
        x = torch.tensor(x_np, dtype=torch.float32,
                         device=self.device).unsqueeze(0)  # [1,T,3]
        y = torch.tensor(y_np, dtype=torch.float32,
                         device=self.device).unsqueeze(0)

        for _ in range(self.epochs):
            self.optim.zero_grad()
            out, _ = self.model(x, self._init_hidden())
            pred   = self.head(out)
            loss   = self.crit(pred, y)
            loss.backward()
            self.optim.step()

    # ---------------------------------------------------------
    @torch.no_grad()
    def predict(self, init_u: np.ndarray, n_steps: int):
        """
        Autoregressive roll-out.
        init_u : initial 3-vector (last known sample)
        Returns array of shape [n_steps, 3].
        """
        self.model.eval(); self.head.eval()

        inp     = torch.tensor(init_u[None, None, :],
                               dtype=torch.float32, device=self.device)
        h, c    = self._init_hidden()
        preds   = np.empty((n_steps, 3), dtype=np.float32)

        for t in range(n_steps):
            out, (h, c) = self.model(inp, (h, c))
            y           = self.head(out)
            preds[t]    = y.squeeze(0).cpu().numpy()
            inp         = y.detach()    # feed prediction back

        return preds