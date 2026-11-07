"""Seed-averaged comparison of spotter sweep variants (e.g. augmentation vs baseline).

    python scripts/compare_spotting.py                 # all families in models/sweep
    python scripts/compare_spotting.py cards_bce all17_bce

Model names are ``<family>[_<variant>]_s<seed>``; variant ``base`` means no suffix. Seed ensembles
(``spotting_ensemble.sbatch``) show up as variant ``<variant>+ens<N>`` with one row each.
Test numbers come from ``outputs/spotting_eval_<name>_test``; valid mAP from the checkpoint.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
VARIANT_TOKENS = {"mix", "tm", "mixtm", "ema", "g2", "g4", "calf", "yc", "sm3"}
TOLS = ["AP@5s", "AP@10s", "AP@30s", "AP@60s"]


def parse(name: str) -> tuple[str, str, int] | None:
    m = re.fullmatch(r"(.+)_s(\d+)", name)
    if not m:
        return None
    parts, seed = m.group(1).split("_"), int(m.group(2))
    k = len(parts)
    while k > 1 and parts[k - 1] in VARIANT_TOKENS:
        k -= 1
    return "_".join(parts[:k]), "_".join(parts[k:]) or "base", seed


def runs():
    """(eval name, family, variant, seed, checkpoint with focus/valid_map) for single models and seed ensembles."""
    for pt in sorted((ROOT / "models/sweep").glob("*.pt")):
        if p := parse(pt.stem):
            yield pt.stem, *p, torch.load(pt, map_location="cpu", weights_only=False)
    for ev in sorted((ROOT / "outputs").glob("spotting_eval_*_ens*_test")):
        name = ev.name[len("spotting_eval_"): -len("_test")]
        stem, n = re.fullmatch(r"(.+)_ens(\d+)", name).groups()
        fam, var, _ = parse(f"{stem}_s0")
        ck = torch.load(ROOT / f"models/sweep/{stem}_s0.pt", map_location="cpu", weights_only=False)
        log = ROOT / f"logs/eval_{name}.log"
        m = re.search(r"valid mAP ([0-9.]+)", log.read_text()) if log.exists() else None
        yield name, fam, f"{var}+ens{n}", -1, {"focus": ck.get("focus"), "valid_map": float(m.group(1)) if m else None}


def rows():
    for name, fam, var, seed, ck in runs():
        ev = ROOT / f"outputs/spotting_eval_{name}_test"
        if not (ev / "ap.csv").exists():
            continue
        ap = pd.read_csv(ev / "ap.csv")
        focus = ck.get("focus") or ap["class"].tolist()
        f = ap[ap["class"].isin(focus)]
        r = {"family": fam, "variant": var, "seed": seed, "valid_mAP": ck.get("valid_map"),
             "best_epoch": ck.get("epoch"), "test_mAP": f[TOLS].mean(axis=1).mean(),
             **{f"test_{t}": f[t].mean() for t in ("AP@5s", "AP@60s")}}
        for _, c in f.iterrows():
            if len(focus) <= 3:
                r[f"{c['class']}"] = c[TOLS].mean()
        lat = ev / "latency.csv"
        if lat.exists():
            for _, c in pd.read_csv(lat).iterrows():
                if c["class"] == "Yellow card":
                    r["YC_live_prec"], r["YC_live_recall@15s"] = c["precision"], c["recall@15s"]
        yield r


def main(families: list[str]):
    df = pd.DataFrame(list(rows()))
    if families:
        df = df[df["family"].isin(families)]
    fams = df.groupby("family")["variant"].transform(lambda v: (v != "base").any())
    df = df[fams].dropna(axis=1, how="all")  # only families with an augmented or ensemble run
    num = [c for c in df.columns if c not in ("family", "variant", "seed")]
    g = df.groupby(["family", "variant"])
    out = g[num].mean().round(4)
    out.insert(0, "n_seeds", g.size())
    out["test_mAP_std"] = g["test_mAP"].std().round(4)
    base = out.xs("base", level="variant")["test_mAP"]
    out["delta_vs_base"] = (out["test_mAP"] - out.index.get_level_values(0).map(base)).round(4)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(out.reset_index().to_string(index=False))
    path = ROOT / "outputs/spotting_aug_compare.csv"
    out.to_csv(path)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
