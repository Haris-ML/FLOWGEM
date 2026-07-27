"""Golden (regression) tests for sample_flowgem (the core WGF sampler).

Unlike the per-function tests, sample_flowgem is stateful and returns a full
trajectory of particle snapshots. The golden test therefore compares the entire
trajectory, snapshot by snapshot, against the reference implementation.
"""

import sys
from pathlib import Path

import numpy as np
import torch

from flowgem.core import sample_flowgem as sample_flowgem_new

# make the vendored reference implementation importable
sys.path.insert(0, str(Path(__file__).parent))
from reference_flowgem import sample_flowgem as sample_flowgem_original


def make_toy_problem(seed=0, n=40, d=3):
    """Build a small, fixed MAR problem: X0, X_obs and mask M.

    Convention (as used by the code): M == 1 means OBSERVED, M == 0 means MISSING.
    Column 0 is forced observed so that no pattern is completely empty.
    """
    torch.manual_seed(seed)

    X_full = torch.randn(n, d, dtype=torch.float64)

    M = (torch.rand(n, d) > 0.3).to(torch.float64)
    M[:, 0] = 1.0  # avoid an all-missing row/pattern

    X_obs = X_full * M
    X0 = X_full.clone()

    return X0, X_obs, M

def test_sample_flowgem_matches_reference():
    """Packaged sample_flowgem must reproduce the reference trajectory exactly."""
    X0, X_obs, M = make_toy_problem()

    # short run: 8 steps is enough to confirm behaviour (T=1000 not needed)
    T = 8

    Xhats_old = sample_flowgem_original(X0, X_obs, M, T=T, eta=0.01, sigma_fix=1.0)
    Xhats_new = sample_flowgem_new(X0, X_obs, M, T=T, eta=0.01, sigma_fix=1.0)

    # 1. same number of snapshots (same number of steps taken)
    assert len(Xhats_new) == len(Xhats_old)

    # 2. every snapshot must match, step by step
    for t, (snap_new, snap_old) in enumerate(zip(Xhats_new, Xhats_old)):
        assert np.allclose(snap_new, snap_old, atol=1e-12, rtol=1e-12), \
            f"Trajectory diverges at step {t}"

def test_early_stopping_triggers():
    """With a loose gradient tolerance, the sampler should stop well before T."""
    X0, X_obs, M = make_toy_problem()

    Xhats = sample_flowgem_new(
        X0, X_obs, M,
        T=100,            # large ceiling
        eta=0.01,
        sigma_fix=1.0,
        grad_tol=1.0,     # very loose -> easy to satisfy
        min_iter=2,       # allow stopping early
    )

    # it should have stopped early, not run the full 100 steps
    assert len(Xhats) < 100


def test_output_structure():
    """The trajectory has the right length and each snapshot the right shape."""
    X0, X_obs, M = make_toy_problem()
    n, d = X0.shape
    T = 6

    Xhats = sample_flowgem_new(X0, X_obs, M, T=T, eta=0.01, sigma_fix=1.0)

    # no early stopping expected here (min_iter default 10 > T), so full T steps
    assert len(Xhats) == T
    for snap in Xhats:
        assert snap.shape == (n, d)