"""Local linear system solver for the FLOWGEM velocity field."""

import torch


def opt_wb(xpi, xrho, x, sigma, block_size=1000, dtype=torch.float64):
    # cast inputs to the requested dtype so all internal computations are consistent
    xpi = xpi.to(dtype)
    xrho = xrho.to(dtype)
    x = x.to(dtype)

    n, d = x.shape
    n_pi = xpi.shape[0]
    n_rho = xrho.shape[0]

    device = x.device

    w = torch.empty(n, d, device=device, dtype=dtype)
    b = torch.empty(n, 1, device=device, dtype=dtype)

    xpi_norm = (xpi**2).sum(dim=1, keepdim=True).T  # (1,n_pi)
    xrho_norm = (xrho**2).sum(dim=1, keepdim=True).T  # (1,n_rho)

    # solve systems blockwise to ensure memory usage linear in sample size
    # (otherwise it might crash for large n)
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        xb = x[start:end]
        bs = xb.shape[0]

        xb_norm = (xb**2).sum(dim=1, keepdim=True)

        # ---------- xpi kernel block ----------
        dist2_pi = xb_norm - 2 * xb @ xpi.T + xpi_norm
        kpix = torch.exp(-dist2_pi / (2 * sigma**2))

        kpix_mean = kpix.mean(dim=1, keepdim=True)
        Xpikpix = (kpix @ xpi) / n_pi

        # ---------- xrho kernel block ----------
        dist2_rho = xb_norm - 2 * xb @ xrho.T + xrho_norm
        krhox = torch.exp(-dist2_rho / (2 * sigma**2))

        krhox_mean = krhox.mean(dim=1, keepdim=True)
        Xrhokrhox = (krhox @ xrho) / n_rho

        # ---------- build c ----------
        c1 = Xpikpix - Xrhokrhox
        c2 = kpix_mean - krhox_mean
        c = torch.cat([c1, c2], dim=1)

        # ---------- build A ----------
        ## A11 = 1/n_rho sum_{i,j} k(x_i,xrho_j) xrho_j xrho_j^T
        A11 = torch.einsum('bk,kd,ke->bde', krhox, xrho, xrho) / n_rho
        A12 = Xrhokrhox
        A22 = krhox_mean.squeeze(1)

        A = torch.zeros(bs, d+1, d+1, dtype=dtype, device=device)
        A[:, :d, :d] = A11
        A[:, :d, d] = A12
        A[:, d, :d] = A12
        A[:, d, d] = A22

        # ---------- solve ----------
        # regularize to avoid (near-)singular system matrices A
        eps = 1e-5
        A += eps * torch.eye(d+1, device=A.device)
        res = torch.linalg.solve(A, c)

        w[start:end] = res[:, :d]
        b[start:end] = res[:, d:]

    return w, b