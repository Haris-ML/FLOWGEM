"""Test that the eta-halving branch fires when the step size is too large."""

import logging
import numpy as np
from flowgem import FlowGEM


def test_eta_halving_triggers(caplog):
    # small dataset with correlation + some missing values
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.7], [0.7, 1.0]], size=60)
    missing = rng.random(60) < 1.0 / (1.0 + np.exp(-X[:, 0]))
    X[missing, 1] = np.nan

    # a deliberately large eta makes the particles overshoot, so the
    # gradient grows and the step size gets halved.
    model = FlowGEM(T=40, eta=5.0, sigma=1.0, random_state=0)

    with caplog.at_level(logging.INFO):
        result = model.fit(X).generate()

    # it still finishes with a complete, finite sample
    assert np.isfinite(np.asarray(result)).all()

    # the halving branch actually fired (message contains "halving")
    assert any("halving" in rec.message.lower() for rec in caplog.records), \
        "eta-halving did not trigger -- increase eta (try 10.0, 20.0, ...)"