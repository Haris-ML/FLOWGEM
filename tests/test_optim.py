"""Golden (regression) tests for opt_wb.

These tests verify that the refactored implementation in `flowgem.optim`
reproduces the original reference implementation exactly. This guards the
refactoring: the code was relocated, but its behaviour must not change.
"""

import sys
from pathlib import Path

import torch

from flowgem.optim import opt_wb as opt_wb_new

# make the vendored reference implementation importable
sys.path.insert(0, str(Path(__file__).parent))
from reference_flowgem import opt_wb as opt_wb_original


def make_inputs(seed=0, n_pi=30, n_rho=25, n=20, d=3):
    """Create small, fixed inputs for comparison."""
    torch.manual_seed(seed)
    xpi = torch.randn(n_pi, d, dtype=torch.float64)
    xrho = torch.randn(n_rho, d, dtype=torch.float64)
    x = torch.randn(n, d, dtype=torch.float64)
    sigma = 1.0
    return xpi, xrho, x, sigma


def test_opt_wb_matches_reference():
    """Packaged opt_wb must reproduce the reference output exactly."""
    xpi, xrho, x, sigma = make_inputs()

    w_old, b_old = opt_wb_original(xpi, xrho, x, sigma)
    w_new, b_new = opt_wb_new(xpi, xrho, x, sigma)

    assert torch.allclose(w_new, w_old, atol=1e-12, rtol=1e-12)
    assert torch.allclose(b_new, b_old, atol=1e-12, rtol=1e-12)


def test_opt_wb_output_shapes():
    """w has shape (n, d) and b has shape (n, 1)."""
    xpi, xrho, x, sigma = make_inputs()
    n, d = x.shape

    w, b = opt_wb_new(xpi, xrho, x, sigma)

    assert w.shape == (n, d)
    assert b.shape == (n, 1)


def test_opt_wb_blocking_does_not_change_result():
    """block_size affects memory only, not the mathematics."""
    xpi, xrho, x, sigma = make_inputs()

    w_one_block, b_one_block = opt_wb_new(xpi, xrho, x, sigma, block_size=1000)
    w_small_blocks, b_small_blocks = opt_wb_new(xpi, xrho, x, sigma, block_size=7)

    assert torch.allclose(w_one_block, w_small_blocks, atol=1e-12, rtol=1e-12)
    assert torch.allclose(b_one_block, b_small_blocks, atol=1e-12, rtol=1e-12)


def test_opt_wb_respects_dtype():
    """The dtype parameter must control the output precision."""
    xpi, xrho, x, sigma = make_inputs()

    # default: float64
    w64, b64 = opt_wb_new(xpi, xrho, x, sigma)
    assert w64.dtype == torch.float64
    assert b64.dtype == torch.float64

    # explicit float32
    w32, b32 = opt_wb_new(xpi, xrho, x, sigma, dtype=torch.float32)
    assert w32.dtype == torch.float32
    assert b32.dtype == torch.float32