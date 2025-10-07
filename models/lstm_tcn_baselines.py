# ============================================================
#  LSTM baseline with ~4.8 k parameters  (PyTorch > 1.9)
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


lstm_baseline = LSTMBaseline3D(
                    hidden_size=32,         # parameter budget ~ 4800
                    lr=1e-3,
                    epochs=40,
                    device='cpu',
                    seed=45)
lstm_baseline.fit(train_input, train_target)

# one-step roll-out to build an initial vector for auto-regressive mode
init_vec = train_target[-1]                # last teacher-forced target
lstm_preds = lstm_baseline.predict(init_vec,
                                   n_steps=len(test_input))







# ============================================================
#  Causal TCN baseline for 3-D Lorenz forecasting (PyTorch)
# ============================================================


class TCNBaseline3D(nn.Module):
    """
    2-layer causal TCN       (kernel=3, dilation=1 & 2, padding chosen
    so receptive field = 5 time-steps).
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


tcn = TCNBaseline3D(hidden_dim=32, epochs=50, lr=1e-3, device="cpu", seed=46)
tcn.fit(train_input, train_target)

# initial window must be >4 samples:
init_win = test_input[:5].copy()
tcn_preds = tcn.predict(init_win, n_steps=len(test_target))