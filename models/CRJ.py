import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from scipy.integrate import odeint
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def scale_spectral_radius(W, target_radius=0.95):
    """
    Scales a matrix W so that its largest eigenvalue magnitude = target_radius.
    """
    eigvals = np.linalg.eigvals(W)
    radius = np.max(np.abs(eigvals))
    if radius == 0:
        return W
    return (W / radius) * target_radius

def augment_state_with_squares(x):
    """
    Given state vector x in R^N, return [ x, x^2, 1 ] in R^(2N+1).
    We'll use this for both training and prediction.
    """
    x_sq = x**2
    return np.concatenate([x, x_sq, [1.0]])  # shape: 2N+1

class CRJ3D:
    """
    Cycle Reservoir with Jumps (CRJ) for 3D->3D single-step tasks.
    We form a ring adjacency with an extra 'jump' edge in each row.
    This can help capture multiple timescales or delayed memory
    while retaining the easy ring structure.

    The adjacency is built as follows (reservoir_size = mod N):
      For each i in [0..N-1]:
        W[i, (i+1) % mod N] = 1.0
        W[i, (i+jump) % mod N] = 1.0
    Then we scale by 'spectral_radius.' We do an ESN update
    with readout [ x, x^2, 1 ] -> next step in R^3.
    """

    def __init__(self,
                 reservoir_size=300,
                 jump=10,                # offset for the jump
                 spectral_radius=0.95,
                 input_scale=1.0,
                 leaking_rate=1.0,
                 ridge_alpha=1e-6,
                 seed=42):
        """
        reservoir_size: how many nodes in the ring
        jump            : the offset for the 2nd connection from node i
        spectral_radius : scale adjacency
        input_scale     : scale factor for W_in
        leaking_rate    : ESN 'alpha'
        ridge_alpha     : ridge penalty for readout
        seed            : random seed
        """
        self.reservoir_size = reservoir_size
        self.jump = jump
        self.spectral_radius = spectral_radius
        self.input_scale = input_scale
        self.leaking_rate = leaking_rate
        self.ridge_alpha = ridge_alpha
        self.seed = seed

        # build adjacency
        np.random.seed(self.seed)
        W = np.zeros((reservoir_size, reservoir_size))
        for i in range(reservoir_size):
            # cycle edge: i -> (i+1)%N
            W[i, (i+1) % reservoir_size] = 1.0
            # jump edge: i -> (i+jump)%N
            W[i, (i + self.jump) % reservoir_size] = 1.0

        # scale spectral radius
        W = scale_spectral_radius(W, self.spectral_radius)
        self.W = W

        # input weights => shape [N,3]
        np.random.seed(self.seed+100)
        W_in = (np.random.rand(reservoir_size, 3) - 0.5)*2.0*self.input_scale
        self.W_in = W_in

        # readout
        self.W_out = None
        self.x = np.zeros(self.reservoir_size)

    def reset_state(self):
        self.x = np.zeros(self.reservoir_size)

    def _update(self, u):
        """
        Single-step ESN update:
          x(t+1) = (1-alpha)*x(t) + alpha*tanh( W x(t) + W_in u(t) )
        """
        pre_activation = self.W @ self.x + self.W_in @ u
        x_new = np.tanh(pre_activation)
        alpha = self.leaking_rate
        self.x = (1.0 - alpha)*self.x + alpha*x_new

    def collect_states(self, inputs, discard=100):
        """
        Teacher forcing => feed the real 3D inputs => gather states.
        Return (states_after_discard, states_discarded).
        """
        self.reset_state()
        states = []
        for val in inputs:
            self._update(val)
            states.append(self.x.copy())
        states = np.array(states)
        return states[discard:], states[:discard]

    def fit_readout(self, train_input, train_target, discard=100):
        """
        gather states => polynomial readout => solve ridge
        """
        states_use, _ = self.collect_states(train_input, discard=discard)
        target_use = train_target[discard:]
        X_list = []
        for s in states_use:
            # polynomial expansion => [ x, x^2, 1 ]
            X_list.append(augment_state_with_squares(s))
        X_aug = np.array(X_list)

        reg = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
        reg.fit(X_aug, target_use)
        self.W_out = reg.coef_  # shape => (3, 2N+1)

    def predict_autoregressive(self, initial_input, n_steps):
        """
        fully autoregressive => feed last output => next input
        """
        preds = []
        #self.reset_state()
        current_in = np.array(initial_input)
        for _ in range(n_steps):
            self._update(current_in)
            big_x = augment_state_with_squares(self.x)
            out = self.W_out @ big_x  # shape => (3,)
            preds.append(out)
            current_in = out
        return np.array(preds)
        
    def predict_open_loop(self, test_input):
        preds = []
        for true_input in test_input:
            self._update(true_input)
            x_aug = augment_state_with_squares(self.x)
            out = self.W_out @ x_aug
            preds.append(out)
        return np.array(preds)
