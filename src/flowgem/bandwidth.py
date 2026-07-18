"""Bandwidth (sigma) selection utilities for FLOWGEM.

Everything related to choosing the RBF kernel bandwidth sigma lives here:
computing pairwise distances, the median heuristic for a quick data-adaptive
estimate, and cross-validation scoring of a candidate sigma.
"""

import torch

from flowgem.optim import opt_wb


def comp_pairwise_dist(x, y):
    # in case x and y are not vectors
    x = x.view(x.shape[0], -1)
    y = y.view(y.shape[0], -1)

    t1 = torch.tile(torch.sum(x**2, dim=1, keepdim=True), (1, y.shape[0]))
    t2 = -2 * torch.matmul(x, y.T)
    t3 = torch.tile(torch.sum(y**2, dim=1, keepdim=True).T, (x.shape[0], 1))

    return t1 + t2 + t3


def sigma_heuristic_pairs(X, num_pairs=1000000):
    n = X.shape[0]

    i = torch.randint(0, n, (num_pairs,))
    j = torch.randint(0, n, (num_pairs,))

    dvals = torch.norm(X[i] - X[j], dim=1)

    return (0.5 * dvals.median()).sqrt().item()


def evaluate_sigma(xpi, xrho, sigma, splits_pi, splits_rho):
    # See thesis: intentional discrepancy with paper's Remark 1; preserved as-is.
    psicon = lambda d: 0.5*d*d + d

    obj = 0

    for (train_pi, val_pi), (train_rho, val_rho) in zip(splits_pi, splits_rho):
        w, b = opt_wb(xpi[train_pi, :], xrho[train_rho, :], torch.cat([xpi[val_pi, :], xrho[val_rho, :]], dim=0), sigma)
        obj += torch.mean((w[:len(val_pi), :] * xpi[val_pi, :]).sum(dim=1) + b[:len(val_pi)].T).item()
        obj -= torch.mean(psicon((w[len(val_pi):, :] * xrho[val_rho, :]).sum(dim=1) + b[len(val_pi):].T)).item()

    return obj