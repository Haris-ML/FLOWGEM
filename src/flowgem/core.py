"""Core FLOWGEM sampler: the discretised Wasserstein gradient flow.

This module orchestrates the whole method — extracting missingness patterns,
selecting bandwidths, and evolving the particle ensemble over time — using the
solver from `optim` and the bandwidth utilities from `bandwidth`.
"""

import logging
import os

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

from flowgem.optim import opt_wb
from flowgem.bandwidth import sigma_heuristic_pairs, evaluate_sigma

logger = logging.getLogger(__name__)

def sample_flowgem(X0, X_obs, M, T=1000, eta=0.01, grad_tol=0.01, min_iter=10, sigma_fix=None, sigma_vals=None, cv_every=10, dtype=torch.float64):
    """Evolve a particle ensemble along the approximate Wasserstein gradient flow.

    This is the core sampler. It extracts the distinct missingness patterns,
    selects the kernel bandwidth, and iteratively updates the particles using
    a per-pattern local-linear velocity estimate (Equation 9), until the final
    time ``T`` or early stopping is reached.

    Parameters
    ----------
    X0 : torch.Tensor of shape (n, d)
        Initial particle ensemble (a complete array, no NaN).
    X_obs : torch.Tensor of shape (n, d)
        Observed data, with missing entries filled by zero.
    M : torch.Tensor of shape (n, d)
        Observation mask; 1 marks an observed entry, 0 a missing one.
    T : int, default=1000
        Maximum number of gradient-flow steps.
    eta : float, default=0.01
        Step size of the forward-Euler update.
    grad_tol : float, default=0.01
        Early-stopping tolerance on the relative gradient norm.
    min_iter : int, default=10
        Minimum number of steps before early stopping may trigger.
    sigma_fix : float or None, default=None
        Fixed kernel bandwidth. If None, a median heuristic is used (unless
        ``sigma_vals`` is given).
    sigma_vals : list of float or None, default=None
        Candidate bandwidths for cross-validation. If given, the bandwidth is
        re-selected by CV every ``cv_every`` steps.
    cv_every : int, default=10
        How often (in steps) to re-run bandwidth cross-validation.
    dtype : torch.dtype, default=torch.float64
        Numerical precision used throughout.

    Returns
    -------
    list of numpy.ndarray
        The per-step snapshots of the particle ensemble; the last element is
        the final generated sample.
    """
    # device and dtype are taken from / controlled by the caller,
    ...
    device = X0.device 

    Xt = X0.clone().to(dtype = dtype, device = device)
    M = M.clone().to(dtype = dtype, device = device)
    X = X_obs.to(dtype = dtype, device = device)

    # determine all possible missingness patterns M
    unique_M, inverse = torch.unique(M, dim=0, return_inverse=True)

    # precompute observed indices for each m (improves computation efficiency)
    mask_idx = {tuple(m.tolist()): (m == 1).nonzero(as_tuple=True)[0] for m in unique_M}

    # samples drawn from p, i.e. conditional distribution X^(m)|M=m for different values of m
    zpi = {m: X[inverse == i][:, mask_idx[m]].clone().to(dtype = dtype, device = device) for i,m in enumerate(mask_idx)}

    # prepare cross validation for sigma
    if sigma_vals is not None:
        n_jobs = min(len(sigma_vals), os.cpu_count()-1)
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        splits_pi = {m: list(kf.split(zpi[m])) for m in mask_idx}
        splits_rho = list(kf.split(Xt))

    grads = {}

    # store current sigma for each pattern m
    if sigma_fix is None:
        sigma_heur = sigma_heuristic_pairs(Xt)
        sigma_current = {m: sigma_heur for m in mask_idx}
        logger.info("Heuristic bandwidth sigma: %s", sigma_heur)
    else:
        sigma_current = {m: sigma_fix for m in mask_idx}

    mean_grad_prev = torch.inf

    Xhats = []
    for t in tqdm(range(T), desc="WGF iterations"):

        run_cv = sigma_vals is not None and t % cv_every == 0

        grad_sum = 0
        for m in mask_idx:
            if run_cv:
                # choose sigma by cross validation (see App. I.1 in MIRI paper)
                logger.debug("Running CV: pattern=%s, size=%s", m, zpi[m].shape[0])
                logger.debug("Candidate sigmas: %s", sigma_vals)
                sigma_scores = Parallel(n_jobs=n_jobs)(
                    delayed(evaluate_sigma)(zpi[m], Xt[:, mask_idx[m]], sig, splits_pi[m], splits_rho)
                    for sig in sigma_vals
                )
                sigma_current[m] = sigma_vals[np.argmax(sigma_scores)]
                logger.debug("Chosen sigma: %s", sigma_current[m])

            # optimize (w,b) according to the objective in equation (8)
            grads[m] = opt_wb(zpi[m], Xt[:, mask_idx[m]], Xt[:, mask_idx[m]], sigma_current[m], dtype=dtype)[0]

            # enlarge grad to R^d
            grad_full = torch.zeros_like(Xt)
            grad_full.index_copy_(1, mask_idx[m], grads[m])
            grad_sum += (zpi[m].shape[0]/X.shape[0]) * grad_full

        # update all particles according to WGF
        Xt = Xt + eta * grad_sum

        Xhats.append(Xt.clone().cpu().numpy())

        # early stopping and eta halving dependent on mean grad value
        mean_grad = (torch.mean(torch.norm(grad_sum, dim=1)) / torch.mean(torch.norm(Xt, dim=1))).item()
        if mean_grad < grad_tol and t+1 > min_iter:
            logger.info("Stopped early after %d iterations (mean grad %.4f < grad_tol %s).", t+1, mean_grad, grad_tol)
            break
        if mean_grad > mean_grad_prev:
            eta = eta*0.5
            logger.info("Mean gradient increased; halving step size eta to %s.", eta)
        mean_grad_prev = mean_grad

    return Xhats