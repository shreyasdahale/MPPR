import numpy as np
from sklearn.linear_model import Ridge


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    return 1.0 / (1.0 + np.exp(-x))


def augment_state_with_squares(x):
    """
    Given state vector x in R^N, return [ x, x^2, 1 ] in R^(2N+1).
    We'll use this for both training and prediction.
    """
    x_sq = x**2
    return np.concatenate([x, x_sq, [1.0]])  # shape: 2N+1


class MPPRN:
    """
    Swirl-Gated k-Cycle Echo-State Network (SG-kC-ESN).

    Parameters
    ----------
    reservoir_size : int
        Total number of neurons N (must be divisible by n_cycles).
    n_cycles       : int, ≥ 2
        How many simple cycles to create.
    cycle_weight   : float
        Weight r on each ring edge.
    bridge_weight  : float
        Weight s on the k sparse bridges (one per cycle).
    input_scale    : float
        Scaling of random input matrix W_in.
    leak_rate      : float in (0,1]
        leaky-integrator update; 1 recovers standard ESN.
    ridge_alpha    : float
        ℓ₂ penalty used in the ridge read-out.
    swirl_beta, swirl_frequency, swirl_sigmoid
        Parameters of the static per-neuron swirl gate
            g_k = σ[β sin(ω·q + φ_c)]      if swirl_sigmoid
                  β sin(ω·q + φ_c)         otherwise
        with φ_c = 2πc / k   (c = ring id).
    """

    # ------------------------------- init ------------------------------ #
    def __init__(
        self,
        reservoir_size: int = 600,
        n_cycles: int = 4,
        cycle_weight: float = 0.9,
        bridge_weight: float = 0.25,
        input_scale: float = 0.5,
        leak_rate: float = 1.0,
        ridge_alpha: float = 1e-6,
        swirl_beta: float = 2.0,
        swirl_frequency: float | None = None,
        swirl_sigmoid: bool = True,
        seed: int = 42,
        use_polynomial_readout: bool = True,
    ):
        if n_cycles < 2:
            raise ValueError("n_cycles must be at least 2")
        if reservoir_size % n_cycles:
            raise ValueError("reservoir_size must be divisible by n_cycles")

        # -------------- basic bookkeeping --------------------------------
        self.N = reservoir_size
        self.k = n_cycles
        self.m = reservoir_size // n_cycles          # neurons per ring

        # -------------- hyper-parameters ---------------------------------
        self.r = cycle_weight
        self.s = bridge_weight
        self.input_scale = input_scale
        self.alpha = leak_rate
        self.ridge_alpha = ridge_alpha
        self.beta = swirl_beta
        self.omega = (
            swirl_frequency if swirl_frequency is not None else 2.0 * np.pi / self.m
        )
        self.swirl_sigmoid = swirl_sigmoid
        self.seed = seed
        self.use_poly = use_polynomial_readout

        # -------------- placeholders to be filled ------------------------
        self.W_res: np.ndarray | None = None
        self.W_in: np.ndarray | None = None
        self.W_out: np.ndarray | None = None
        self.gate: np.ndarray | None = None
        self.x = np.zeros(self.N, dtype=np.float32)

        # -------------- one-off construction -----------------------------
        self._build_reservoir()
        self._build_swirl_gate()

    # =========================== builders =============================== #
    def _build_reservoir(self):
        """Construct the k-cycle recurrent matrix W_res (shape N × N)."""
        m, r, s, k = self.m, self.r, self.s, self.k

        # 1) ring block C_r : unidirectional permutation matrix scaled by r
        C_r = np.zeros((m, m), dtype=np.float32)
        for i in range(m):
            C_r[(i + 1) % m, i] = r

        # 2) bridge block S : rank-1 matrix with single non-zero entry (0,0)
        S = np.zeros((m, m), dtype=np.float32)
        S[0, 0] = s

        # 3) assemble full block matrix
        W = np.zeros((self.N, self.N), dtype=np.float32)

        def put_block(row_ring: int, col_ring: int, block: np.ndarray):
            i0, j0 = row_ring * m, col_ring * m
            W[i0 : i0 + m, j0 : j0 + m] = block

        for c in range(k):
            put_block(c, c, C_r)                   # diagonal ring
            put_block(c, (c - 1) % k, S)           # bridge to predecessor

        self.W_res = W

    def _build_swirl_gate(self):
        """Pre-compute static gain vector g (length N)."""
        g = np.empty(self.N, dtype=np.float32)
        for k_idx in range(self.N):
            ring_id = k_idx // self.m
            local_q = k_idx % self.m
            phi_c = 2.0 * np.pi * ring_id / self.k
            raw = self.beta * np.sin(self.omega * local_q + phi_c)
            g[k_idx] = sigmoid(raw) if self.swirl_sigmoid else raw
        self.gate = g

    # ====================== low-level reservoir ops ===================== #
    def _apply_gate(self, vec: np.ndarray) -> np.ndarray:
        return self.gate * vec

    def _update_state(self, u_t: np.ndarray):
        """Single ESN step with optional leakage."""
        pre = self.W_res @ self.x + self.W_in @ u_t
        gated = self._apply_gate(pre)
        new_x = np.tanh(gated)
        self.x = (1.0 - self.alpha) * self.x + self.alpha * new_x

    def reset_state(self):
        self.x.fill(0.0)

    # ====================== read-out training (ridge) =================== #
    def fit_readout(self, inputs: np.ndarray, targets: np.ndarray, discard: int = 100):
        """
        Teacher forcing pass • inputs [T, d_in] → states, then fit ridge.
        """
        T, d_in = inputs.shape
        if T <= discard + 1:
            raise ValueError("Not enough data for training")

        rng = np.random.default_rng(self.seed)
        self.W_in = (
            rng.uniform(-1.0, 1.0, size=(self.N, d_in)) * self.input_scale
        ).astype(np.float32)

        self.reset_state()
        states = []
        for t in range(T):
            self._update_state(inputs[t])
            if t >= discard:
                states.append(self.x.copy())

        states = np.asarray(states, dtype=np.float32)          # [T-discard, N]
        Y = targets[discard:]                                  # same length

        if self.use_poly:
            feats = np.concatenate(
                [states, states * states, np.ones((states.shape[0], 1), dtype=np.float32)],
                axis=1,
            )
        else:
            feats = states

        reg = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
        reg.fit(feats, Y)
        self.W_out = reg.coef_.astype(np.float32)             # d_out × feat_dim

    # ======================== autoregressive roll-out =================== #
    def predict_autoregressive(
        self, initial_input: np.ndarray, n_steps: int
    ) -> np.ndarray:
        if self.W_out is None:
            raise RuntimeError("Call fit_readout() before prediction.")

        d_in = initial_input.shape[0]
        preds = np.empty((n_steps, self.W_out.shape[0]), dtype=np.float32)

        #self.reset_state()
        current_in = initial_input.astype(np.float32).copy()

        for t in range(n_steps):
            self._update_state(current_in)

            if self.use_poly:
                big_x = np.concatenate(
                    [self.x, self.x * self.x, np.ones(1, dtype=np.float32)]
                )
            else:
                big_x = self.x

            y_t = (self.W_out @ big_x).astype(np.float32)
            preds[t] = y_t
            current_in = y_t[:d_in]  # feedback: assume d_in ≤ d_out

        return preds
    
    # def predict_open_loop(self, test_input):
    #     preds = []
    #     for true_input in test_input:
    #         self._update_state(true_input)
    #         if self.use_poly:
    #             big_x = np.concatenate(
    #                 [self.x, self.x * self.x, np.ones(1, dtype=np.float32)]
    #             )
    #         else:
    #             big_x = self.x
    #         out = self.W_out @ big_x
    #         preds.append(out)
    #     return np.array(preds)
    
    def predict_open_loop(self, test_input):
        preds = []
        for true_input in test_input:
            self._update(true_input)
            x_aug = augment_state_with_squares(self.x)
            out = self.W_out @ x_aug
            preds.append(out)
        return np.array(preds)
