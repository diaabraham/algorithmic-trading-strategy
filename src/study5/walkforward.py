"""Expanding-window, monthly-refit walk-forward ("recursive learning") on a panel.

Self-contained (numpy plus lightgbm or torch) so it can run on another machine.
At decision month t: train on rows with month <= t - 1 (their target month is
<= t) whose asset is usable at t (pub_year < calendar year of t), then predict
rows with month == t and usable assets.
"""

from __future__ import annotations

import time

import numpy as np

LGBM_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=500, subsample=0.7,
                   subsample_freq=1, colsample_bytree=0.7, random_state=7, verbose=-1)
MLP_HIDDEN = (512, 256, 128)
MLP_EPOCHS, MLP_BATCH, MLP_LR, MLP_WD, MLP_DROPOUT, SEED = 6, 2048, 1e-3, 1e-4, 0.1, 7


def _year(month: int) -> int:
    return 1900 + month // 12


def fit_predict_lgbm(Xtr, ytr, Xte, n_jobs=8):
    import lightgbm as lgb

    m = lgb.LGBMRegressor(n_jobs=n_jobs, **LGBM_PARAMS).fit(Xtr, ytr)
    return m.predict(Xte)


def fit_predict_mlp(Xtr, ytr, Xte, device="cpu"):
    import torch

    mu = np.nanmean(Xtr, 0)
    sd = np.nanstd(Xtr, 0) + 1e-8
    prep = lambda X: np.nan_to_num(np.clip((X - mu) / sd, -5, 5)).astype(np.float32)
    Xt, Xe = prep(Xtr), prep(Xte)
    ysd = float(np.std(ytr)) or 1.0
    torch.manual_seed(SEED)
    layers, d = [], Xt.shape[1]
    for h in MLP_HIDDEN:
        layers += [torch.nn.Linear(d, h), torch.nn.ReLU(), torch.nn.Dropout(MLP_DROPOUT)]
        d = h
    layers.append(torch.nn.Linear(d, 1))
    net = torch.nn.Sequential(*layers).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=MLP_LR, weight_decay=MLP_WD)
    X_t = torch.tensor(Xt).to(device)
    y_t = torch.tensor((ytr / ysd).astype(np.float32)).reshape(-1, 1).to(device)
    g = torch.Generator().manual_seed(SEED)
    n = len(Xt)
    net.train()
    for _ in range(MLP_EPOCHS):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, MLP_BATCH):
            b = perm[i : i + MLP_BATCH].to(device) if device == "cpu" else perm[i : i + MLP_BATCH].to(device)
            opt.zero_grad()
            loss = ((net(X_t[b]) - y_t[b]) ** 2).mean()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        out = net(torch.tensor(Xe).to(device)).cpu().numpy().ravel() * ysd
    return out


def run(arrays: dict, first_decision_month: int, last_decision_month: int, model: str, device="cpu", log_every=60):
    X, y, month, pub = arrays["X"], arrays["y"], arrays["month"], arrays["pub_year"]
    pred = np.full(len(y), np.nan, dtype=np.float64)
    timings = []
    has_y = np.isfinite(y)
    for t in range(first_decision_month, last_decision_month + 1):
        usable = pub < _year(t)
        tr = (month <= t - 1) & usable & has_y
        te = (month == t) & usable
        if tr.sum() < 1000 or te.sum() == 0:
            continue
        t0 = time.time()
        if model == "lgbm":
            p = fit_predict_lgbm(X[tr], y[tr], X[te])
        else:
            p = fit_predict_mlp(X[tr], y[tr], X[te], device=device)
        timings.append(time.time() - t0)
        pred[te] = p
        if log_every and (t - first_decision_month) % log_every == 0:
            print(f"{model} month {t} ({_year(t)}) train {tr.sum()} test {te.sum()} {timings[-1]:.2f}s", flush=True)
    return pred, np.array(timings)
