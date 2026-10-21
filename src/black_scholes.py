"""Black-Scholes helpers for a hold-horizon cone around spot.

Sigma is realized volatility, not an options-implied surface. The backtest has no listed chain.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _year_fraction(horizon_days: float) -> float:
    return max(float(horizon_days), 1.0) / 252.0


def expected_move(spot, sigma, horizon_days: float):
    """One-standard-deviation move, S * sigma * sqrt(T)."""
    t = _year_fraction(horizon_days)
    spot_arr = np.asarray(spot, dtype=float)
    sigma_arr = np.clip(np.asarray(sigma, dtype=float), 0.0, None)
    return spot_arr * sigma_arr * np.sqrt(t)


def _d1_d2(spot, strike, sigma, horizon_days: float, rate: float = 0.02):
    t = _year_fraction(horizon_days)
    spot_arr = np.clip(np.asarray(spot, dtype=float), 1e-8, None)
    strike_arr = np.clip(np.asarray(strike, dtype=float), 1e-8, None)
    sigma_arr = np.clip(np.asarray(sigma, dtype=float), 1e-4, None)
    vol_sqrt = sigma_arr * np.sqrt(t)
    d2 = (np.log(spot_arr / strike_arr) + (rate - 0.5 * sigma_arr**2) * t) / vol_sqrt
    d1 = d2 + vol_sqrt
    return d1, d2, t


def black_scholes_up_proba(spot, strike, sigma, horizon_days: float, rate: float = 0.02):
    """Risk-neutral probability that price finishes above the strike, N(d2)."""
    _, d2, _ = _d1_d2(spot, strike, sigma, horizon_days, rate)
    return norm.cdf(d2)


def black_scholes_call(spot, strike, sigma, horizon_days: float, rate: float = 0.02):
    """European call with no dividend. Price is in the same units as spot."""
    d1, d2, t = _d1_d2(spot, strike, sigma, horizon_days, rate)
    spot_arr = np.clip(np.asarray(spot, dtype=float), 1e-8, None)
    strike_arr = np.clip(np.asarray(strike, dtype=float), 1e-8, None)
    return spot_arr * norm.cdf(d1) - strike_arr * np.exp(-rate * t) * norm.cdf(d2)
