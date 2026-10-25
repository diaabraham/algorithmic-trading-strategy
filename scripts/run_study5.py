"""Study 5 driver (see results/study5_protocol.md).

    python scripts/run_study5.py build    --phase dev|holdout
    python scripts/run_study5.py predict  --phase dev|holdout --model lgbm --kind long|ls
    python scripts/study5_gpu_driver.py ...   # MLP, run on the GPU machine
    python scripts/run_study5.py evaluate --phase dev|holdout
"""

import argparse
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.study5 import walkforward  # noqa: E402
from src.study5.data import CACHE, load  # noqa: E402
from src.study5.panel import build, to_arrays  # noqa: E402

PHASES = {
    # decision months (months since 1900-01): dev predicts 1975-01..2014-12, holdout 2015-01..2024-12
    "dev": {"end": "2014-12-31", "first": (1974 - 1900) * 12 + 11, "last": (2014 - 1900) * 12 + 10},
    "holdout": {"end": "2024-12-31", "first": (2014 - 1900) * 12 + 11, "last": (2024 - 1900) * 12 + 10},
}


def path(name):
    os.makedirs(CACHE, exist_ok=True)
    return os.path.join(CACHE, name)


def cmd_build(phase):
    d = load(PHASES[phase]["end"])
    for kind in ("long", "ls"):
        panel = build(d, kind)
        panel.to_pickle(path(f"panel_{phase}_{kind}.pkl"))
        np.savez_compressed(path(f"arrays_{phase}_{kind}.npz"), **to_arrays(panel))
        print(kind, panel.shape)


def cmd_predict(phase, model, kind, device):
    arr = dict(np.load(path(f"arrays_{phase}_{kind}.npz")))
    p = PHASES[phase]
    pred, timings = walkforward.run(arr, p["first"], p["last"], model, device=device)
    np.save(path(f"pred_{phase}_{model}_{kind}.npy"), pred)
    with open(path(f"timing_{phase}_{model}_{kind}_{device}.json"), "w") as fh:
        json.dump({"refits": len(timings), "total_s": float(timings.sum()), "mean_s": float(timings.mean())}, fh)
    print(f"{model}/{kind}: {len(timings)} refits, {timings.sum():.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["build", "predict", "evaluate"])
    ap.add_argument("--phase", choices=list(PHASES), required=True)
    ap.add_argument("--model", default="lgbm")
    ap.add_argument("--kind", default="long")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.cmd == "build":
        cmd_build(a.phase)
    elif a.cmd == "predict":
        cmd_predict(a.phase, a.model, a.kind, a.device)
    else:
        from src.study5.evaluate import evaluate

        evaluate(a.phase)
