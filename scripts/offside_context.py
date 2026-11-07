"""Does the context around an Offside peak separate real offsides from other stoppages?

    python scripts/offside_context.py models/sweep/all17_bce_s0.pt

Rescores every Offside peak with a logistic regression (fitted on valid) over the spotter's
own scores for all classes in time bins around the peak, and with an oracle feature (an
Indirect free-kick label 3-45 s later). Prints test Offside AP for each.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from varflash.spotting.data import FPS, LABEL_FILE, available_games, load_halves
from varflash.spotting.infer import frame_scores, load_spotter, peak_pick
from varflash.spotting.metrics import average_precision

BINS_S = [(-30, -8), (-8, -2), (-2, 2), (2, 8), (8, 20), (20, 45)]
TOLS = (5, 10, 30, 60)


def peaks_with_context(model, ck, split):
    classes = ck["classes"]
    c = classes.index("Offside")
    window = ck.get("config", {}).get("window", 120)
    rows = []
    for h in load_halves(available_games(split), radius=0, classes=ck.get("config", {}).get("classes", "all")):
        s = frame_scores(model, h.features, window)
        p = peak_pick(s[:, [c]], ["Offside"])
        labels = json.loads((__import__("pathlib").Path(h.game) / LABEL_FILE).read_text())["annotations"]
        ifk = np.array([int(a["position"]) / 1000 for a in labels
                        if a["label"] == "Indirect free-kick" and a["gameTime"].startswith(f"{h.half} ")])
        for r in p.itertuples():
            f = r.frame
            feat = {"score": r.confidence}
            for a, b in BINS_S:
                lo, hi = max(f + a * FPS, 0), max(min(f + b * FPS, len(s)), 0)
                m = s[lo:hi].max(0) if hi > lo else np.zeros(len(classes))
                feat.update({f"{k}_{a}_{b}": m[i] for i, k in enumerate(classes)})
            feat["oracle_ifk"] = float(((ifk > r.time_s + 3) & (ifk < r.time_s + 45)).any())
            rows.append({"video": f"{h.game}#{h.half}", "time_s": r.time_s, **feat})
    gts = pd.DataFrame([{"video": f"{h.game}#{h.half}", "time_s": f / FPS}
                        for h in load_halves(available_games(split), radius=0) for f, k in h.events
                        if k == classes.index("Offside")])
    return pd.DataFrame(rows), gts


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def mean_ap(df, conf, gts):
    d = df[["video", "time_s"]].assign(confidence=conf)
    return {f"AP@{t}s": round(average_precision(d, gts, t), 3) for t in TOLS} | {
        "mAP": round(np.mean([average_precision(d, gts, t) for t in TOLS]), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    args = ap.parse_args()
    model, ck = load_spotter(args.model)
    va, va_gt = peaks_with_context(model, ck, "valid")
    te, te_gt = peaks_with_context(model, ck, "test")
    from varflash.spotting.metrics import match_predictions
    y = match_predictions(va.assign(confidence=va["score"]), va_gt, 5.0)
    ctx = [c for c in va.columns if c not in ("video", "time_s", "oracle_ifk")]

    def fit(cols):
        X = lambda d: _logit(d[cols].to_numpy())
        m = LogisticRegression(C=0.3, max_iter=3000).fit(X(va), y)
        return m.predict_proba(X(te))[:, 1]

    print(f"valid peaks {len(va)} (hits {y.sum()}), test peaks {len(te)}")
    print("spotter alone      ", mean_ap(te, te["score"], te_gt))
    print("+ oracle IFK label ", mean_ap(te, te["score"] * (0.1 + 0.9 * te["oracle_ifk"]), te_gt))
    print("+ context (all cls)", mean_ap(te, fit(ctx), te_gt))
    ifk_cols = ["score"] + [c for c in ctx if c.startswith("Indirect free-kick_")]
    print("+ context (IFK)    ", mean_ap(te, fit(ifk_cols), te_gt))


if __name__ == "__main__":
    main()
