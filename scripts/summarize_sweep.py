"""Card scorecard across the sweep: test AP (mean over 5/10/30/60 s), calibration, live latency.

    python scripts/summarize_sweep.py [models/sweep]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from varflash.paths import OUTPUTS
from varflash.spotting.data import FPS, available_games, load_halves
from varflash.spotting.infer import frame_scores, load_spotter, peak_pick
from varflash.spotting.metrics import spotting_report

TOL = [f"AP@{t}s" for t in (5, 10, 30, 60)]


def all17_as_cards(path: Path, device: str) -> dict:
    """Score the 17-class model on the card task: Sending off = max(Red card, Yellow->red card)."""
    model, ckpt = load_spotter(path, device)
    cls = ckpt["classes"]
    halves = load_halves(available_games("test"), radius=0, classes="cards")
    preds, gts = [], []
    for h in halves:
        s = frame_scores(model, h.features)
        cards = np.stack([s[:, cls.index("Yellow card")],
                          np.maximum(s[:, cls.index("Red card")], s[:, cls.index("Yellow->red card")])], 1)
        p = peak_pick(cards, ["Yellow card", "Sending off"])
        p["video"] = f"{h.game}#{h.half}"
        preds.append(p)
        gts += [{"video": f"{h.game}#{h.half}", "time_s": f / FPS, "label": ["Yellow card", "Sending off", "Foul"][c]}
                for f, c in h.events]
    rep = spotting_report(pd.concat(preds), pd.DataFrame(gts), ["Yellow card", "Sending off"])
    return {f"test_AP_{r['class']}": float(np.mean([r[t] for t in TOL])) for _, r in rep.iterrows()}


def main(sweep=Path("models/sweep")):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for m in sorted(sweep.glob("*.pt")):
        ck = torch.load(m, map_location="cpu", weights_only=False)
        row = {"run": m.stem, "epoch": ck["epoch"], "valid_map": ck["valid_map"]}
        if ck["config"]["classes"] == "all":
            row.update(all17_as_cards(m, device))
        else:
            d = OUTPUTS / f"spotting_eval_{m.stem}_test"
            if not (d / "ap.csv").exists():
                continue
            ap = pd.read_csv(d / "ap.csv").set_index("class")
            for c in ("Yellow card", "Sending off"):
                row[f"test_AP_{c}"] = float(ap.loc[c, TOL].mean())
            cal = pd.read_csv(d / "calibration.csv")
            for c in ("Yellow card", "Sending off"):
                r = cal[(cal["class"] == c) & (cal["which"] == "calibrated")]
                row[f"ECE_{c}"] = float(r["ece"].iloc[0]) if len(r) else np.nan
                r = cal[(cal["class"] == c) & (cal["which"] == "raw")]
                row[f"ECE_raw_{c}"] = float(r["ece"].iloc[0]) if len(r) else np.nan
            if (d / "latency.csv").exists():
                lat = pd.read_csv(d / "latency.csv").set_index("class")
                for c in lat.index:
                    for k in ("precision", "median_delay_s", "recall@45s", "recall@75s"):
                        row[f"live_{c}_{k}"] = lat.loc[c, k]
        rows.append(row)
    df = pd.DataFrame(rows)
    df["variant"] = df["run"].str.replace(r"_s\d+$", "", regex=True)
    out = OUTPUTS / "sweep_summary.csv"
    df.to_csv(out, index=False)
    agg = df.drop(columns=["run"]).groupby("variant").agg(["mean", "std"]).round(3)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    keep = [c for c in agg.columns.levels[0] if c.startswith(("valid_map", "test_AP", "ECE_Sending", "ECE_raw_Send"))]
    print(agg[keep].to_string())
    live = [c for c in agg.columns.levels[0] if c.startswith("live_")]
    if live:
        print("\nlive (causal) alerts on test, threshold set for 50% precision on valid:")
        print(agg[live].xs("mean", axis=1, level=1).dropna(how="all").round(3).T.to_string())
    print(f"\n-> {out}")


if __name__ == "__main__":
    main(*(Path(a) for a in sys.argv[1:]))
