import numpy as np

from src.portfolio_sim import _bootstrap_slice


def test_bootstrap_slice_compounds():
    rng = np.random.default_rng(0)
    capital = 1000.0
    rets = [0.01, -0.02, 0.03]
    out = [_bootstrap_slice(rets, rng, capital) for _ in range(200)]
    arr = np.array(out)
    assert np.isfinite(arr).all()
    assert arr.min() > 0
