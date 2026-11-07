"""Speed vs accuracy of live (causal) goal alerts for spotters with different look-ahead budgets.

For each model and target precision, the alert threshold is picked on the validation split and
the live alerts are scored on test: precision, recall within 2/5/10/... s of the goal, and the
median delay after the ball crosses the line (SoccerNet's Goal timestamp).

    python scripts/goal_latency.py models/sweep/goal_causal_d*_s0.pt models/live_d10.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from varflash.paths import OUTPUTS
from varflash.spotting.data import FPS, available_games, load_halves
from varflash.spotting.infer import load_spotter
from varflash.spotting.metrics import latency_report, pick_threshold, spotting_report
from varflash.spotting.train import _live_inputs, predict_halves

LABEL = "Goal"
GRID = np.r_[np.linspace(0.05, 0.95, 37), 1 - np.geomspace(0.04, 0.0005, 20)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+")
    p.add_argument("--precisions", nargs="+", type=float, default=[0.5, 0.7, 0.9])
    p.add_argument("--out", default=str(OUTPUTS / "goal_latency.csv"))
    a = p.parse_args()

    cache: dict[str, list] = {}
    rows = []
    for path in a.models:
        model, ckpt = load_spotter(path, "cuda")
        classes, cfg = ckpt["classes"], ckpt.get("config", {})
        key = cfg.get("classes", "all")
        if key not in cache:
            cache[key] = [load_halves(available_games(s), radius=0, classes=key) for s in ("valid", "test")]
        valid, test = cache[key]
        c = classes.index(LABEL)
        window = cfg.get("window", 120)
        _, _, live_v = predict_halves(model, valid, classes, window, ckpt["delay"])
        preds, gts, live_t = predict_halves(model, test, classes, window, ckpt["delay"], ckpt.get("calibrators"))
        ap = spotting_report(preds, gts, classes).set_index("class").loc[LABEL]
        for target in a.precisions:
            th = pick_threshold(_live_inputs(valid, live_v, c), target, grid=GRID)
            r = latency_report(_live_inputs(test, live_t, c), th)
            rows.append({"model": Path(path).stem, "lookahead_s": ckpt["delay"] / FPS, "causal": cfg.get("causal"),
                         "target_precision": target, **r,
                         **{k: ap[k] for k in ap.index if str(k).startswith("AP")}})
        print(pd.DataFrame(rows[-len(a.precisions):]).round(3).to_string(index=False), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    cols = ["model", "lookahead_s", "target_precision", "threshold", "precision", "median_delay_s",
            "recall@2s", "recall@5s", "recall@10s", "recall@15s", "recall@30s"]
    print("\n" + df[cols].round(3).to_string(index=False))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
