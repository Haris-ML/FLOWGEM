"""Tests for the public FlowGEM class (api.py).

These are behavioural tests: the class is new code with no reference to
compare against, so they check that it does what the interface promises
rather than matching an original implementation.
"""

import numpy as np
import pytest
import torch

from flowgem.api import FlowGEM


def make_incomplete(seed=0, n=40, d=3):
    """A small dataset with NaN marking missing values."""
    torch.manual_seed(seed)
    X = torch.randn(n, d, dtype=torch.float64)
    mask = torch.rand(n, d) > 0.3
    mask[:, 0] = True                 # keep column 0 observed
    X[~mask] = float("nan")
    return X


# ---- core behaviour -------------------------------------------------------

def test_fit_generate_returns_complete_sample():
    """generate() returns a same-shape sample with no NaN."""
    X = make_incomplete()
    model = FlowGEM(T=5, sigma=1.0, random_state=0)
    X_out = model.fit(X).generate()

    assert X_out.shape == X.shape
    assert not torch.isnan(X_out).any()


def test_fit_returns_self():
    """fit() returns self, so fit(...).generate() chains."""
    X = make_incomplete()
    model = FlowGEM(T=3, sigma=1.0, random_state=0)
    assert model.fit(X) is model


def test_mask_is_inferred_from_nan():
    """The internal mask marks observed (non-NaN) entries as 1."""
    X = torch.tensor([[1.0, float("nan"), 3.0],
                      [4.0, 5.0, float("nan")]], dtype=torch.float64)
    model = FlowGEM().fit(X)
    expected = torch.tensor([[1.0, 0.0, 1.0],
                             [1.0, 1.0, 0.0]], dtype=torch.float64)
    assert torch.equal(model.mask_, expected)


# ---- initialisation modes -------------------------------------------------

@pytest.mark.parametrize("init", ["mice", "sample"])
def test_init_modes_produce_complete_X0(init):
    """Both string init modes yield an X0 with no missing values."""
    X = make_incomplete()
    model = FlowGEM(init=init, random_state=0).fit(X)
    assert not torch.isnan(model.X0_).any()
    assert model.X0_.shape == X.shape


def test_init_array_is_used():
    """A supplied init array is used directly as X0."""
    X = make_incomplete(n=10, d=3)
    my_X0 = torch.zeros(10, 3, dtype=torch.float64)
    model = FlowGEM(init=my_X0).fit(X)
    assert torch.equal(model.X0_, my_X0)


# ---- validation and guards ------------------------------------------------

def test_1d_input_raises():
    X = torch.tensor([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        FlowGEM().fit(X)


def test_no_nan_input_raises():
    X = torch.randn(10, 3, dtype=torch.float64)  # no NaN -> nothing missing
    with pytest.raises(ValueError):
        FlowGEM().fit(X)


def test_generate_before_fit_raises():
    with pytest.raises(RuntimeError):
        FlowGEM().generate()


# ---- reproducibility ------------------------------------------------------

def test_random_state_is_reproducible():
    """Same random_state -> identical output."""
    X = make_incomplete()
    out1 = FlowGEM(T=5, sigma=1.0, random_state=42).fit(X).generate()
    out2 = FlowGEM(T=5, sigma=1.0, random_state=42).fit(X).generate()
    assert torch.allclose(out1, out2)


# ---- trajectory option ----------------------------------------------------

def test_generate_trajectory_returns_list():
    """return_trajectory=True yields the full per-step trajectory."""
    X = make_incomplete()
    T = 5
    model = FlowGEM(T=T, sigma=1.0, random_state=0).fit(X)

    traj = model.generate(return_trajectory=True)
    final = model.generate(return_trajectory=False)

    # a list of snapshots, each the right shape
    assert isinstance(traj, list)
    assert len(traj) == T
    assert all(snap.shape == X.shape for snap in traj)

    # the last snapshot equals the default (final-only) output
    assert torch.allclose(traj[-1], final)