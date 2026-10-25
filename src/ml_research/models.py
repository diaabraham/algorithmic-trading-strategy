"""Models and purged / embargoed cross-validation (Lopez de Prado, 2018, ch. 7)."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

SEED = 7


def purged_kfold(n: int, horizon: int, folds: int = 5, embargo: int = None) -> List[tuple]:
    """Contiguous folds; training rows whose label window touches the test block are purged."""
    embargo = horizon if embargo is None else embargo
    bounds = np.linspace(0, n, folds + 1).astype(int)
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        test = np.arange(a, b)
        keep = np.ones(n, bool)
        keep[max(0, a - horizon - 1) : min(n, b + embargo)] = False
        out.append((np.flatnonzero(keep), test))
    return out


class Model:
    """Common wrapper: impute with training medians, scale (linear/neural), fit, predict."""

    def __init__(self, kind: str, params: dict):
        self.kind, self.params = kind, params

    def _prep(self, X, fit=False):
        if fit:
            self.med = np.nanmedian(X, axis=0)
            self.med = np.where(np.isfinite(self.med), self.med, 0.0)
        X = np.where(np.isfinite(X), X, self.med)
        if self.kind in ("ridge", "mlp"):
            if fit:
                self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
            X = np.clip((X - self.mu) / self.sd, -5, 5)
        return X

    def fit(self, X, y):
        Xp = self._prep(X, fit=True)
        if self.kind == "ridge":
            from sklearn.linear_model import Ridge

            self.m = Ridge(alpha=self.params["alpha"]).fit(Xp, y)
        elif self.kind == "lgbm":
            import lightgbm as lgb

            self.m = lgb.LGBMRegressor(
                n_estimators=200, learning_rate=0.03, num_leaves=self.params["num_leaves"],
                min_child_samples=self.params["min_child_samples"], subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, n_jobs=4, verbose=-1,
            ).fit(X, y)  # trees take raw features with NaN
        elif self.kind == "mlp":
            from sklearn.neural_network import MLPRegressor

            self.m = MLPRegressor(hidden_layer_sizes=(32, 16), alpha=self.params["alpha"], learning_rate_init=1e-3,
                                  max_iter=200, early_stopping=True, validation_fraction=0.15, n_iter_no_change=10,
                                  random_state=SEED).fit(Xp, y)
        else:
            raise ValueError(self.kind)
        self.train_pred_sd = float(np.std(self.predict(X))) or 1e-6
        return self

    def predict(self, X):
        if self.kind == "lgbm":
            return self.m.predict(X)
        return self.m.predict(self._prep(X))


GRIDS = {
    "ridge": [{"alpha": 10.0}, {"alpha": 1000.0}],
    "lgbm": [{"num_leaves": 7, "min_child_samples": 200}, {"num_leaves": 15, "min_child_samples": 50}],
    "mlp": [{"alpha": 1e-3}, {"alpha": 1e-1}],
}


def choose_params(kind: str, X, y, horizon: int) -> dict:
    """Pick the grid point with the lowest purged-CV mean squared error."""
    best, best_err = None, np.inf
    for params in GRIDS[kind]:
        errs = []
        for tr, te in purged_kfold(len(y), horizon):
            m = Model(kind, params).fit(X[tr], y[tr])
            errs.append(np.mean((m.predict(X[te]) - y[te]) ** 2))
        if np.mean(errs) < best_err:
            best, best_err = params, float(np.mean(errs))
    return best
