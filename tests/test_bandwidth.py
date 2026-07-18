"""Golden (regression) tests for the bandwidth module.

Verifies that comp_pairwise_dist, sigma_heuristic_pairs and evaluate_sigma
reproduce the original reference implementation exactly.
"""

import sys
from pathlib import Path

import torch
from sklearn.model_selection import KFold

from flowgem.bandwidth import (
    comp_pairwise_dist as comp_pairwise_dist_new,
    sigma_heuristic_pairs as sigma_heuristic_pairs_new,
    evaluate_sigma as evaluate_sigma_new,
)

# make the vendored reference implementation importable
sys.path.insert(0, str(Path(__file__).parent))
from reference_flowgem import (
    comp_pairwise_dist as comp_pairwise_dist_original,
    sigma_heuristic_pairs as sigma_heuristic_pairs_original,
    evaluate_sigma as evaluate_sigma_original,
)


def test_comp_pairwise_dist_matches_reference():
    """Deterministic: same distances from both implementations."""
    torch.manual_seed(0)
    x = torch.randn(15, 3, dtype=torch.float64)
    y = torch.randn(12, 3, dtype=torch.float64)

    d_old = comp_pairwise_dist_original(x, y)
    d_new = comp_pairwise_dist_new(x, y)

    assert torch.allclose(d_new, d_old, atol=1e-12, rtol=1e-12)


def test_sigma_heuristic_pairs_matches_reference():
    """Random function: seed is reset before EACH call so both see
    the same random pairs and must return the same sigma."""
    torch.manual_seed(0)
    X = torch.randn(40, 3, dtype=torch.float64)

    torch.manual_seed(123)
    sigma_old = sigma_heuristic_pairs_original(X, num_pairs=500)

    torch.manual_seed(123)
    sigma_new = sigma_heuristic_pairs_new(X, num_pairs=500)

    assert abs(sigma_new - sigma_old) < 1e-12


def test_evaluate_sigma_matches_reference():
    """Deterministic given fixed splits; also exercises opt_wb via the
    cross-module import."""
    torch.manual_seed(0)
    xpi = torch.randn(30, 3, dtype=torch.float64)
    xrho = torch.randn(30, 3, dtype=torch.float64)
    sigma = 1.0

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    splits_pi = list(kf.split(xpi))
    splits_rho = list(kf.split(xrho))

    obj_old = evaluate_sigma_original(xpi, xrho, sigma, splits_pi, splits_rho)
    obj_new = evaluate_sigma_new(xpi, xrho, sigma, splits_pi, splits_rho)

    assert abs(obj_new - obj_old) < 1e-10