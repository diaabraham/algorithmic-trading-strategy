"""Run the study 5 MLP walk-forward on a DirectML (AMD) or CPU device.

    python study5_gpu_driver.py arrays_dev_long.npz FIRST LAST dml|cpu [max_refits]

Needs walkforward.py next to it (copied from src/study5/walkforward.py).
Writes pred_<input>_<device>.npy and timing_<input>_<device>.json.
"""

import json
import sys
import time

import numpy as np

import walkforward

npz, first, last, dev = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
limit = int(sys.argv[5]) if len(sys.argv) > 5 else None
if limit:
    last = min(last, first + limit - 1)
if dev == "dml":
    import torch_directml

    names = [torch_directml.device_name(i) for i in range(torch_directml.device_count())]
    idx = next(i for i, n in enumerate(names) if "RX" in n)
    device = torch_directml.device(idx)
    print("device:", names[idx], flush=True)
else:
    device = "cpu"
arr = dict(np.load(npz))
t0 = time.time()
pred, timings = walkforward.run(arr, first, last, "mlp", device=device, log_every=60)
stem = npz.replace(".npz", "")
np.save(f"pred_{stem}_{dev}.npy", pred)
json.dump({"device": dev, "refits": int(len(timings)), "total_s": float(timings.sum()), "mean_s": float(timings.mean()),
           "wall_s": time.time() - t0}, open(f"timing_{stem}_{dev}.json", "w"))
print(f"done {len(timings)} refits, {timings.sum():.1f}s, mean {timings.mean():.2f}s", flush=True)
