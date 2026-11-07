"""Error analysis for one spotter class (default Offside) on a split.

    python scripts/offside_errors.py models/sweep/all17_bce_s0.pt [--split test] [--cls Offside]

Prints: AP by label visibility, what ground-truth events sit near false alarms, which
neighbouring events surround missed events, and score distributions.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd

from varflash.spotting.data import FPS, LABEL_FILE, available_games, load_halves
from varflash.spotting.infer import event_scores, frame_scores, load_spotter, peak_pick
from varflash.spotting.metrics import average_precision, match_predictions


def all_labels(game_dir) -> pd.DataFrame:
    a = json.loads((game_dir / LABEL_FILE).read_text())["annotations"]
    df = pd.DataFrame(a)
    df["half"] = df["gameTime"].str.split(" - ").str[0].astype(int)
    df["time_s"] = df["position"].astype(int) / 1000
    df["video"] = str(game_dir) + "#" + df["half"].astype(str)
    return df[["video", "time_s", "label", "visibility", "team"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--split", default="test")
    ap.add_argument("--cls", default="Offside")
    ap.add_argument("--max-games", type=int)
    args = ap.parse_args()

    model, ck = load_spotter(args.model)
    classes, delay = ck["classes"], ck["delay"]
    window = ck.get("config", {}).get("window", 120)
    c = classes.index(args.cls)
    games = available_games(args.split)[: args.max_games]
    halves = load_halves(games, radius=0, classes=ck.get("config", {}).get("classes", "all"))
    labels = pd.concat([all_labels(g) for g in games], ignore_index=True)

    preds, raw_by_video = [], {}
    for h in halves:
        s = event_scores(frame_scores(model, h.features, window), delay)
        v = f"{h.game}#{h.half}"
        raw_by_video[v] = s[:, c]
        p = peak_pick(s, classes)
        p = p[p["label"] == args.cls].copy()
        p["video"] = v
        preds.append(p)
    preds = pd.concat(preds, ignore_index=True)
    gts = labels[labels["label"] == args.cls].reset_index(drop=True)
    others = labels[labels["label"] != args.cls]

    print(f"{args.cls}: {len(gts)} events, {len(preds)} peaks, visibility {dict(Counter(gts['visibility']))}")
    for vis in ["visible", "not shown"]:
        g = gts[gts["visibility"] == vis]
        # Peaks matched to the other visibility group must not count as false positives here.
        other = gts[gts["visibility"] != vis]
        keep = ~match_predictions(preds, other, 5.0) if len(other) else np.ones(len(preds), bool)
        print(f"  AP@5s {vis:>9}: {average_precision(preds[keep], g, 5):.3f}  "
              f"AP@30s {average_precision(preds[keep], g, 30):.3f}  (n={len(g)})")

    # Peak score at each true event (max frame score within +-5 s), recall at thresholds.
    peak_at = []
    for _, r in gts.iterrows():
        s = raw_by_video[r["video"]]
        f = int(r["time_s"] * FPS)
        peak_at.append(s[max(f - 10, 0): f + 11].max())
    gts["score"] = peak_at
    print("\nmax score within +-5 s of true events, quantiles:",
          np.round(np.quantile(peak_at, [0.1, 0.25, 0.5, 0.75, 0.9]), 3))
    for th in (0.1, 0.3, 0.5):
        print(f"  recall at score >= {th}: {(np.array(peak_at) >= th).mean():.3f}  "
              f"(visible {(gts[gts.visibility == 'visible'].score >= th).mean():.3f}, "
              f"not shown {(gts[gts.visibility == 'not shown'].score >= th).mean():.3f})")

    # False alarms: confident peaks with no true event within 10 s. What is nearby?
    preds["hit"] = match_predictions(preds, gts, 10.0)
    fa = preds[(~preds["hit"]) & (preds["confidence"] >= 0.3)]
    print(f"\nfalse alarms (conf >= 0.3, no {args.cls} within 10 s): {len(fa)}  vs true hits {preds['hit'].sum()}")
    near = Counter()
    for _, r in fa.iterrows():
        o = others[(others["video"] == r["video"]) & ((others["time_s"] - r["time_s"]).abs() <= 10)]
        near[o.loc[(o["time_s"] - r["time_s"]).abs().idxmin(), "label"] if len(o) else "(nothing within 10 s)"] += 1
    for k, v in near.most_common(10):
        print(f"  nearest label {k:<22} {v}")

    # What annotated events follow / precede a true event (context the model can use)?
    after, before = Counter(), Counter()
    for _, r in gts.iterrows():
        o = others[others["video"] == r["video"]]
        nxt = o[(o["time_s"] > r["time_s"]) & (o["time_s"] <= r["time_s"] + 60)].sort_values("time_s")
        prv = o[(o["time_s"] < r["time_s"]) & (o["time_s"] >= r["time_s"] - 30)].sort_values("time_s")
        after[nxt.iloc[0]["label"] if len(nxt) else "(none in 60 s)"] += 1
        before[prv.iloc[-1]["label"] if len(prv) else "(none in 30 s)"] += 1
        if len(nxt):
            gts.loc[_, "next_label"], gts.loc[_, "next_gap"] = nxt.iloc[0]["label"], nxt.iloc[0]["time_s"] - r["time_s"]
    print("\nnext label after a true event:", after.most_common(6))
    print("previous label before a true event:", before.most_common(6))
    ifk = gts[gts.get("next_label") == "Indirect free-kick"]["next_gap"]
    if len(ifk):
        print("gap to following Indirect free-kick, s, quantiles:", np.round(np.quantile(ifk, [0.1, 0.5, 0.9]), 1))
    miss = gts[gts["score"] < 0.3]
    print(f"\nmissed (score < 0.3): {len(miss)}; next label for missed:",
          Counter(miss.get("next_label", pd.Series(dtype=str)).fillna("(none)")).most_common(5))


if __name__ == "__main__":
    main()
