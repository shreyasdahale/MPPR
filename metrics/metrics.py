import numpy as np
from scipy.signal import welch
from sklearn.neighbors import NearestNeighbors
from scipy.stats import linregress

def compute_valid_prediction_time(y_true, y_pred, t_vals, threshold, lambda_max, dt):
    """
    Compute the Valid Prediction Time (VPT) and compare it to Lyapunov time T_lambda = 1 / lambda_max.
    
    Parameters
    ----------
    y_true : ndarray of shape (N, dim)
        True trajectory over time.
    y_pred : ndarray of shape (N, dim)
        Model's predicted trajectory over time (closed-loop).
    t_vals : ndarray of shape (N,)
        Time values corresponding to the trajectory steps.
    threshold : float, optional
        The error threshold, default is 0.4 as in your snippet.
    lambda_max : float, optional
        Largest Lyapunov exponent. Default=0.9 for Lorenz.
        
    Returns
    -------
    T_VPT : float
        Valid prediction time. The earliest time at which normalized error surpasses threshold
        (or the last time if never surpassed).
    T_lambda : float
        Lyapunov time = 1 / lambda_max
    ratio : float
        How many Lyapunov times the model prediction remains valid, i.e. T_VPT / T_lambda.
    """
    # 1) Average of y_true
    y_mean = np.mean(y_true, axis=0)  # shape (dim,)
    
    # 2) Time-averaged norm^2 of (y_true - y_mean)
    y_centered = y_true - y_mean
    denom = np.mean(np.sum(y_centered**2, axis=1))  # scalar
    
    # 3) Compute the normalized error delta_gamma(t) = ||y_true - y_pred||^2 / denom
    diff = y_true - y_pred
    err_sq = np.sum(diff**2, axis=1)  # shape (N,)
    delta_gamma = err_sq / denom      # shape (N,)
    
    # 4) Find the first time index where delta_gamma(t) exceeds threshold
    idx_exceed = np.where(delta_gamma > threshold)[0]
    if len(idx_exceed) == 0:
        # never exceeds threshold => set T_VPT to the final time
        T_VPT = t_vals[-1]
    else:
        T_VPT = t_vals[idx_exceed[0]]
    
    # 5) Compute T_lambda and ratio
    T_lambda = 1.0 / lambda_max

    print(f"\n--- Valid Prediction Time (VPT) with threshold={threshold}, lambda_max={lambda_max} ---")

    T_VPT = (T_VPT - t_vals[0])  # Adjust T_VPT to be relative to the start time
    ratio = T_VPT / T_lambda

    return T_VPT, T_lambda, ratio

# def compute_valid_prediction_time(y_true, y_pred, t_vals, threshold=0.4, lambda_max=0.9):
#     """
#     Compute the Valid Prediction Time (VPT) and compare it to Lyapunov time T_lambda = 1 / lambda_max.
    
#     Parameters
#     ----------
#     y_true : ndarray of shape (N, dim)
#         True trajectory over time.
#     y_pred : ndarray of shape (N, dim)
#         Model's predicted trajectory over time (closed-loop).
#     t_vals : ndarray of shape (N,)
#         Time values corresponding to the trajectory steps.
#     threshold : float, optional
#         The error threshold, default is 0.4 as in your snippet.
#     lambda_max : float, optional
#         Largest Lyapunov exponent. Default=0.9 for Lorenz.
        
#     Returns
#     -------
#     T_VPT : float
#         Valid prediction time. The earliest time at which normalized error surpasses threshold
#         (or the last time if never surpassed).
#     T_lambda : float
#         Lyapunov time = 1 / lambda_max
#     ratio : float
#         How many Lyapunov times the model prediction remains valid, i.e. T_VPT / T_lambda.
#     """
#     # 1) Average of y_true
#     y_mean = np.mean(y_true, axis=0)  # shape (dim,)
    
#     # 2) Time-averaged norm^2 of (y_true - y_mean)
#     y_centered = y_true - y_mean
#     denom = np.mean(np.sum(y_centered**2, axis=1))  # scalar
    
#     # 3) Compute the normalized error delta_gamma(t) = ||y_true - y_pred||^2 / denom
#     diff = y_true - y_pred
#     err_sq = np.sum(diff**2, axis=1)  # shape (N,)
#     delta_gamma = err_sq / denom      # shape (N,)
    
#     # 4) Find the first time index where delta_gamma(t) exceeds threshold
#     idx_exceed = np.where(delta_gamma > threshold)[0]
#     if len(idx_exceed) == 0:
#         # never exceeds threshold => set T_VPT to the final time
#         T_VPT = t_vals[-1]
#     else:
#         T_VPT = t_vals[idx_exceed[0]]
    
#     # 5) Compute T_lambda and ratio
#     T_lambda = 1.0 / lambda_max
#     ratio = T_VPT / T_lambda
    
#     return T_VPT, T_lambda, ratio

# def mse_dimwise(pred, truth):
#     length = min(len(pred), len(truth))
#     return np.mean((pred[:length] - truth[:length])**2, axis=0)

def mse_dimwise(pred, truth):
    length = min(len(pred), len(truth))
    mse = np.mean((pred[:length] - truth[:length])**2, axis=0)
    return np.linalg.norm(mse)

def nrmse_dimwise(pred, truth):
    length = min(len(pred), len(truth))
    pred = pred[:length]
    truth = truth[:length]
    mse = np.mean((pred - truth) ** 2, axis=0)
    std = np.std(truth, axis=0)
    std[std == 0] = 1e-8
    nrmse = np.sqrt(mse) / std
    return np.linalg.norm(nrmse)

def compute_attractor_deviation(predictions, targets, cube_size=(0.1, 0.1, 0.1)):
    """
    Compute the Attractor Deviation (ADev) metric.

    Parameters:
        predictions (numpy.ndarray): Predicted trajectories of shape (n, 3).
        targets (numpy.ndarray): True trajectories of shape (n, 3).
        cube_size (tuple): Dimensions of the cube (dx, dy, dz).

    Returns:
        float: The ADev metric.
    """
    # Define the cube grid based on the range of the data and cube size
    min_coords = np.min(np.vstack((predictions, targets)), axis=0)
    max_coords = np.max(np.vstack((predictions, targets)), axis=0)

    # Create a grid of cubes
    grid_shape = ((max_coords - min_coords) / cube_size).astype(int) + 1

    # Initialize the cube occupancy arrays
    pred_cubes = np.zeros(grid_shape, dtype=int)
    target_cubes = np.zeros(grid_shape, dtype=int)

    # Map trajectories to cubes
    pred_indices = ((predictions - min_coords) / cube_size).astype(int)
    target_indices = ((targets - min_coords) / cube_size).astype(int)

    # Mark cubes visited by predictions and targets
    for idx in pred_indices:
        pred_cubes[tuple(idx)] = 1
    for idx in target_indices:
        target_cubes[tuple(idx)] = 1

    # Compute the ADev metric
    adev = np.sum(np.abs(pred_cubes - target_cubes))

    return adev

def compute_psd(y, dt=0.01):
    z = y[:, 2]  # Extract Z-component
    x = y[:, 0]  # Extract X-component
    y1 = y[:, 1]  # Extract Y-component
    # Compute PSD using Welch’s method
    freqs_z, psd_z = welch(z, fs=1/dt, window='hamming', nperseg=len(y)//4)  # Using Hamming window
    freqs_x, psd_x = welch(x, fs=1/dt, window='hamming', nperseg=len(y)//4)  # Using Hamming window
    freqs_y, psd_y = welch(y1, fs=1/dt, window='hamming', nperseg=len(y)//4)  # Using Hamming window

    return freqs_z, psd_z, freqs_x, psd_x, freqs_y, psd_y

# def compute_relative_psd(y_true, y_pred, dt=0.01):
#     _, psd_z_true, _, psd_x_true, _, psd_y_true = compute_psd(y_true, dt)
#     _, psd_z_pred, _, psd_x_pred, _, psd_y_pred = compute_psd(y_pred, dt)

#     D_psd_z = np.sqrt(np.sum((psd_z_true - psd_z_pred) ** 2))
#     D_psd_x = np.sqrt(np.sum((psd_x_true - psd_x_pred) ** 2))
#     D_psd_y = np.sqrt(np.sum((psd_y_true - psd_y_pred) ** 2))

#     return D_psd_z, D_psd_x, D_psd_y

def compute_relative_psd(y_true, y_pred, dt=0.01):
    freqs_z, psd_z_true, freqs_x, psd_x_true, freqs_y, psd_y_true = compute_psd(y_true, dt)
    _, psd_z_pred, _, psd_x_pred, _, psd_y_pred = compute_psd(y_pred, dt)

    D_psd_z = np.sqrt(np.trapz((psd_z_true - psd_z_pred)**2, freqs_z))
    D_psd_x = np.sqrt(np.trapz((psd_x_true - psd_x_pred)**2, freqs_x))
    D_psd_y = np.sqrt(np.trapz((psd_y_true - psd_y_pred)**2, freqs_y))

    PSD = D_psd_z + D_psd_x + D_psd_y

    return PSD

def compute_max_lyapunov(
    data: np.ndarray,
    emb_dim: int = 10,
    delay: int = 1,
    min_tsep: int = 10,
    fit_range: tuple = (0, None)
) -> float:
    """
    Estimate the maximal Lyapunov exponent using Rosenstein et al.'s algorithm.
    
    Parameters
    ----------
    data : ndarray, shape (N,) or (N, d)
        Scalar or multivariate time series.
    emb_dim : int
        Embedding dimension.
    delay : int
        Time delay for embedding.
    min_tsep : int
        The Theiler window to exclude temporally correlated neighbors.
    fit_range : (int, int or None)
        Time indices (start, end) over which to fit the divergence curve.
    
    Returns
    -------
    lambda_max : float
        Estimated maximal Lyapunov exponent (per time step).
    """
    # 1. Build delay embedding
    N = len(data)
    M = N - (emb_dim - 1) * delay
    embedded = np.empty((M, emb_dim))
    for i in range(emb_dim):
        embedded[:, i] = data[i * delay : i * delay + M]
    
    # 2. Find nearest neighbors with temporal exclusion
    nbrs = NearestNeighbors(n_neighbors=2).fit(embedded)
    distances, indices = nbrs.kneighbors(embedded)
    # discard self-match (distance zero) -> take second nearest
    neigh_idx = indices[:, 1]
    # enforce Theiler window
    valid = np.abs(neigh_idx - np.arange(M)) > min_tsep
    neigh_idx = np.where(valid, neigh_idx, np.arange(M))
    
    # 3. Track divergence over time
    max_t = M - max(neigh_idx)
    div = np.zeros(max_t)
    counts = np.zeros(max_t)
    for i in range(M):
        j = neigh_idx[i]
        max_len = min(M - i, M - j)
        for k in range(max_len):
            div[k] += np.log(np.linalg.norm(embedded[i + k] - embedded[j + k]))
            counts[k] += 1
    # average log divergence
    avg_log_div = div[counts > 0] / counts[counts > 0]
    
    # 4. Linear fit over specified range
    t = np.arange(len(avg_log_div))
    start, end = fit_range
    end = end or len(t)
    slope, _, _, _, _ = linregress(t[start:end], avg_log_div[start:end])
    return slope


def lyapunov_deviation(
    true_traj: np.ndarray,
    pred_traj: np.ndarray,
    **lyap_kwargs
) -> float:
    """
    Compute the absolute difference between the maximal Lyapunov exponents
    of the true and predicted trajectories.
    
    Parameters
    ----------
    true_traj : ndarray
        Ground‑truth time series.
    pred_traj : ndarray
        Model‑generated time series.
    **lyap_kwargs :
        Keyword arguments passed to compute_max_lyapunov.
    
    Returns
    -------
    delta_lambda : float
        |lambda_max(true_traj) - lambda_max(pred_traj)|.
    """
    lambda_true = compute_max_lyapunov(true_traj, **lyap_kwargs)
    lambda_pred = compute_max_lyapunov(pred_traj, **lyap_kwargs)
    return abs(lambda_true - lambda_pred)

