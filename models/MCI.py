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

class MCI3D:
    """
    Minimum Complexity Interaction ESN (MCI-ESN).

    This class implements the approach described in:
      "A Minimum Complexity Interaction Echo State Network"
        by Jianming Liu, Xu Xu, Eric Li (2024).
    
    The model structure:
      - We maintain two 'simple cycle' reservoirs (each of size N).
      - Each reservoir is a ring with weight = l, i.e. 
            W_res[i, (i+1)%N] = l
        plus the corner wrap from (N-1)->0, also = l. ##(unnecessary as already called for in the prev. line)
      - The two reservoirs interact via a minimal connection matrix: 
         exactly 2 cross-connections with weight = g. 
         (One might connect x2[-1], x2[-2], ... 
          But we do where reservoir1 sees x2[-1] 
          in one location, and reservoir2 sees x1[-1] likewise.)
      - Activation function in reservoir1 is cos(·), and in reservoir2 is sin(·).
      - They each have a separate input weight matrix: Win1 and Win2. 
        The final state is a linear combination 
           x(t) = h*x1(t) + (1-h)*x2(t).
      - Then we do a polynomial readout [x, x^2, 1] -> output.
      - We feed teacher forcing in collect_states, 
        then solve readout with Ridge regression.

    References:
      - Liu, J., Xu, X., & Li, E. (2024). 
        "A minimum complexity interaction echo state network," 
         Neural Computing and Applications.
    
    notes:
      - The reservoir_size is N for each reservoir, 
        so total param dimension is 2*N for states, 
        but we produce a single final "combined" state x(t) in R^N for readout.
      - The activation f1=cos(...) for reservoir1, f2=sin(...) for reservoir2, 
        as recommended by the paper for MCI-ESN.

    """

    def __init__(
        self,
        reservoir_size=500,
        cycle_weight=0.9,      # 'l' in the paper
        connect_weight=0.9,    # 'g' in the paper
        input_scale=0.2,
        leaking_rate=1.0,
        ridge_alpha=1e-6,
        combine_factor=0.1,    # 'h' in the paper
        seed=47,
        v1=0.6, v2=0.6         # fixed values for v1, v2
    ):
        """
        reservoir_size: N, size of each cycle reservoir 
        cycle_weight : l, ring adjacency weight in [0,1), ensures cycle synergy
        connect_weight: g, cross-connection weight between the two cycle reservoirs
        input_scale   : scale factor for input->reservoir weights
        leaking_rate  : ESN update alpha 
        ridge_alpha   : readout ridge penalty
        combine_factor: h in [0,1], to form x(t)= h*x1(t)+(1-h)*x2(t) as final combined state
        seed          : random seed
        """
        self.reservoir_size = reservoir_size
        self.cycle_weight   = cycle_weight
        self.connect_weight = connect_weight
        self.input_scale    = input_scale
        self.leaking_rate   = leaking_rate
        self.ridge_alpha    = ridge_alpha
        self.combine_factor = combine_factor
        self.seed           = seed
        self.v1 = v1
        self.v2 = v2

        # We'll define (and build) adjacency for each cycle, 
        # plus cross-connection for two sub-reservoirs.
        # We'll define 2 input weight mats: Win1, Win2.
        # We'll define states x1(t), x2(t).
        # We'll define readout W_out after training.

        self._build_mci_esn()

    def _build_mci_esn(self):
        """
        Build all the internal parameters: 
         - ring adjacency for each reservoir
         - cross-reservoir connection
         - input weights for each reservoir
         - initial states
        """
        np.random.seed(self.seed)

        N = self.reservoir_size

        # Build ring adjacency W_res in shape [N, N], with cycle_weight on ring
        W_res = np.zeros((N, N))
        for i in range(N):
            j = (i+1) % N
            W_res[j, i] = self.cycle_weight
        self.W_res = W_res  # shared by both sub-reservoirs

        # Build cross-connection W_cn for shape [N,N], 
        # minimal 2 nonzero elements. 
        # For the simplest approach from the paper:
        #   W_cn[0, N-1] = g, W_cn[1, N-2] = g or similar.
        # The paper's eq(7) suggests the last 2 elements in x(t) cross to first 2 in the other reservoir:
        # We'll do the simplest reference: if i=0 or i=1, we connect from the other reservoir's last or second-last. 
        # We'll define a function for each sub-res to pick up from the other sub-res. 
        # We can store them in separate arrays, or define them in code. 
        # We'll just store "We want index 0 to see x2[-1], index 1 to see x2[-2]."

        # But as done in the original code snippet from the paper:
        #   Wcn has
        # effectively 2 nonzero positions. We'll define that pattern:
        W_cn = np.zeros((N, N))
        # e.g. W_cn[0, N-1] = g, W_cn[N-1, N-2] = g or something. 
        # The paper example used W_cn = diag(0,g,...) plus the corner. We'll do the simplest:
        # let W_cn[0, N-1]=g, W_cn[1, N-2]=g.
        # This matches the minimal cross. 
        # For clarity we do:
        W_cn[0, N-1] = self.connect_weight
        if N>1:
            # W_cn[1, N-2] = self.connect_weight
            W_cn[N-1, 0] = self.connect_weight
        self.W_cn = W_cn

        # We'll define input weights for each sub-reservoir, shape [N, dim_input].
        # The paper sets them as eq(10) in the snippet, with different signs. 
        # We'll define them as parted. 
        # We define V1, V2 => shape [N, dim_input], with constant magnitude t1, t2, random sign. 
        # We'll do random. Need to check this in the paper again
        # We'll keep "two" separate. user can define input_scale but not two separate. 
        # We'll do the simplest approach: the absolute value is the same => input_scale, 
        # sign is random. Then we define Win1 = V1 - V2, Win2 = V1 + V2.
        # This is consistent with eq(10) from the paper.

        self.Win1 = None
        self.Win2 = None

        # We'll define states x1(t), x2(t). We'll do them after dimension known. 
        self.x1 = None
        self.x2 = None

        self.W_out = None

    def _init_substates(self):
        """
        Once we know reservoir_size, we define x1, x2 as zeros. 
        We'll call this in reset_state or at fit time.
        """
        N = self.reservoir_size
        self.x1 = np.zeros(N)
        self.x2 = np.zeros(N)

    def reset_state(self):
        if self.x1 is not None:
            self.x1[:] = 0.0
        if self.x2 is not None:
            self.x2[:] = 0.0

    def _update(self, u):
        """
        Single-step reservoir update.
        x1(t+1) = cos( Win1*u(t+1) + W_res*x1(t) + W_cn*x2(t) )
        x2(t+1) = sin( Win2*u(t+1) + W_res*x2(t) + W_cn*x1(t) )
        Then x(t)= h*x1(t+1) + (1-h)* x2(t+1).
        We'll define the leaky integration. 
        But the paper uses an approach with no leak? Be careful.
        We'll do the approach: x1(t+1)= (1-alpha)* x1(t) + alpha*cos(...).
        """
        alpha = self.leaking_rate

        # pre activation for reservoir1
        pre1 = self.Win1 @ u + self.W_res @ self.x1 + self.W_cn @ self.x2
        # reservoir1 uses cos
        new_x1 = np.cos(pre1)

        # reservoir2 uses sin
        pre2 = self.Win2 @ u + self.W_res @ self.x2 + self.W_cn @ self.x1
        new_x2 = np.sin(pre2)

        self.x1 = (1.0 - alpha)*self.x1 + alpha*new_x1
        self.x2 = (1.0 - alpha)*self.x2 + alpha*new_x2

    def _combine_state(self):
        """
        Combine x1(t), x2(t) => x(t) = h*x1 + (1-h)*x2
        """
        h = self.combine_factor
        return h*self.x1 + (1.0 - h)*self.x2

    def collect_states(self, inputs, discard=100):
        # We reset the reservoir to zero
        self.reset_state()
        states = []
        for t in range(len(inputs)):
            self._update(inputs[t])   # feed the REAL input from the dataset
            combined = self._combine_state()
            states.append(combined.copy())
        states = np.array(states)  # shape => [T, N]
        return states[discard:], states[:discard]


    def fit_readout(self, train_input, train_target, discard=100):
        """
        Build input weights if needed, gather states on the training data (teacher forcing),
        then solve a polynomial readout [x, x^2, 1]->train_target(t).

        train_input : shape [T, d_in]
        train_target: shape [T, d_out]
        discard     : # of states to discard for warmup
        """
        T = len(train_input)
        if T<2:
            raise ValueError("Not enough training data")

        d_in = train_input.shape[1]
        # d_out = train_target.shape[1]

        # built Win1, Win2
        if self.Win1 is None or self.Win2 is None:
            np.random.seed(self.seed+100)
            # build V1, V2 in shape [N, d_in]
            N = self.reservoir_size
            # V1 = (np.random.rand(N, d_in)-0.5)*2.0*self.input_scale
            # V2 = (np.random.rand(N, d_in)-0.5)*2.0*self.input_scale

            sign_V1 = np.random.choice([-1, 1], size=(N, d_in))
            sign_V2 = np.random.choice([-1, 1], size=(N, d_in))

            v1, v2 = self.v1, self.v2  # fixed values for V1, V2

            V1 = v1 * sign_V1 * self.input_scale
            V2 = v2 * sign_V2 * self.input_scale

            # eq(10): Win1= V1 - V2, Win2= V1 + V2
            self.Win1 = V1 - V2
            self.Win2 = V1 + V2

        # define x1, x2
        self._init_substates()

        # gather states
        states_use, _ = self.collect_states(train_input, discard=discard)
        target_use = train_target[discard:]  # shape => [T-discard, d_out]

        # polynomial readout
        X_list = []
        for s in states_use:
            X_list.append(augment_state_with_squares(s))
        X_aug = np.array(X_list)  # shape => [T-discard, 2N+1]

        # Solve ridge
        reg = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
        reg.fit(X_aug, target_use)
        # W_out => shape [d_out, 2N+1]
        self.W_out = reg.coef_

    def predict_autoregressive(self, initial_input, n_steps):
        """
        Fully autoregressive: 
          We do not use teacher forcing, 
          we feed the model's last output as the next input 
        Typically, for MCI-ESN the paper does input(t+1) in R^d. 
        We do the test_input
        For multi-step chaotic forecast, we feed the model's output as input? 
        That means the system dimension d_in must match d_out. 
        """
        preds = []
        # re-init states
        #self._init_substates()

        # we assume initial_input => shape (d_in,)
        current_in = np.array(initial_input)

        for _ in range(n_steps):
            self._update(current_in)
            # read out
            combined = self._combine_state()
            big_x = augment_state_with_squares(combined)
            out = self.W_out @ big_x  # shape => (d_out,)

            preds.append(out)
            current_in = out  # feed output back as next input

        return np.array(preds)
        
    def predict_open_loop(self, test_input):
        preds = []
        for true_input in test_input:
            self._update(true_input)
            combined = self._combine_state()
            x_aug = augment_state_with_squares(combined)
            out = self.W_out @ x_aug
            preds.append(out)
        return np.array(preds)
