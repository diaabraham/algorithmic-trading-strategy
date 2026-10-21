"""Brinson-Fachler attribution against a buy-and-hold benchmark."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd


def brinson_fachler(
    portfolio_weights: Sequence[float],
    benchmark_weights: Sequence[float],
    portfolio_returns: Sequence[float],
    benchmark_returns: Sequence[float],
) -> Dict[str, float]:
    """One-period Brinson-Fachler across groups (symbols or sectors)."""
    w_p = np.asarray(portfolio_weights, dtype=float)
    w_b = np.asarray(benchmark_weights, dtype=float)
    r_p = np.asarray(portfolio_returns, dtype=float)
    r_b = np.asarray(benchmark_returns, dtype=float)
    if not (len(w_p) == len(w_b) == len(r_p) == len(r_b)):
        raise ValueError("Brinson inputs must be the same length.")
    total_b = float(np.sum(w_b * r_b))
    allocation = float(np.sum((w_p - w_b) * (r_b - total_b)))
    selection = float(np.sum(w_b * (r_p - r_b)))
    interaction = float(np.sum((w_p - w_b) * (r_p - r_b)))
    return {
        "allocation": allocation,
        "selection": selection,
        "interaction": interaction,
        "benchmark_return": total_b,
        "portfolio_return": float(np.sum(w_p * r_p)),
        "excess_return": float(np.sum(w_p * r_p) - total_b),
    }


def brinson_single_asset(position: pd.Series, asset_returns: pd.Series) -> Dict[str, float]:
    """
    Split a single-name book versus always-long buy-and-hold.

    Allocation is the effect of being flat instead of long.
    Selection is the effect of being short instead of long.
    Interaction is the residual so the three sum to the arithmetic excess.
    Compounded portfolio and benchmark returns are reported separately.
    """
    frame = pd.concat({"w": position, "r": asset_returns}, axis=1).dropna()
    if frame.empty:
        return {
            "allocation": 0.0,
            "selection": 0.0,
            "interaction": 0.0,
            "arithmetic_excess": 0.0,
            "benchmark_return": 0.0,
            "portfolio_return": 0.0,
            "excess_return": 0.0,
        }
    w = frame["w"].astype(float)
    r = frame["r"].astype(float)
    arithmetic_excess = float(((w - 1.0) * r).sum())
    allocation = float(((w.abs() - 1.0) * r).sum())
    selection = float(((w - w.abs()) * r).sum())
    interaction = float(arithmetic_excess - allocation - selection)
    benchmark = float(np.prod(1.0 + r.to_numpy()) - 1.0)
    portfolio = float(np.prod(1.0 + (w * r).to_numpy()) - 1.0)
    return {
        "allocation": allocation,
        "selection": selection,
        "interaction": interaction,
        "arithmetic_excess": arithmetic_excess,
        "benchmark_return": benchmark,
        "portfolio_return": portfolio,
        "excess_return": portfolio - benchmark,
    }
