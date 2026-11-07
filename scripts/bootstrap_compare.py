"""95% confidence intervals (match-level bootstrap) for test AP, and paired gains over a baseline.

    # spotter families in models/sweep (all seeds of each are averaged); first one is the baseline
    python scripts/bootstrap_compare.py spotter multitask_bce multitask_bce_calf
    # card rescore outputs: "" = current hindsight verifier, others are `cards rescore --tag` suffixes
    python scripts/bootstrap_compare.py cards "" _live_d0 _live_d2 _live_d4

Spotter predictions are recomputed from the checkpoints once and cached in outputs/bootstrap/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

from varflash.cards.clips import CARD_CLASSES
from varflash.cards.rescore import _gts, fit_fusion
from varflash.cards.rescore import _features as fusion_features
from varflash.paths import MODELS, OUTPUTS
from varflash.spotting.bootstrap import compare
from varflash.spotting.data import available_games, load_halves
from varflash.spotting.infer import load_spotter
from varflash.spotting.train import predict_halves

OUT = OUTPUTS / "bootstrap"
N_BOOT = 1000


def _rel(path: str) -> str:
    """Game key independent of the home-dir alias (/home/... vs /orcd/home/...)."""
    return path.split("/soccernet/")[-1]


def _test_games() -> list[str]:
    return [_rel(str(g)) for g in available_games("test")]


def _rel_videos(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(video=df["video"].map(_rel))


def spotter_preds(path: Path, device: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cache = OUT / f"preds_{path.stem}_test.parquet"
    gcache = OUT / f"gts_{path.stem}_test.parquet"
    model, ck = load_spotter(path, device)
    if cache.exists() and gcache.exists():
        return pd.read_parquet(cache), pd.read_parquet(gcache), ck
    cfg = ck["config"]
    halves = load_halves(available_games("test"), radius=0, classes=cfg["classes"])
    preds, gts, _ = predict_halves(model, halves, ck["classes"], cfg["window"], ck.get("delay", 0),
                                   ck.get("calibrators"))
    OUT.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(cache, index=False)
    gts.to_parquet(gcache, index=False)
    return preds, gts, ck


def spotter_mode(families: list[str]):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sets, gts, focus = {}, None, None
    for fam in families:
        paths = sorted((MODELS / "sweep").glob(f"{fam}_s[0-9].pt"))
        if not paths:
            raise SystemExit(f"no checkpoints models/sweep/{fam}_s*.pt")
        sets[fam] = []
        for p in paths:
            preds, g, ck = spotter_preds(p, device)
            f = ck.get("focus") or ck["classes"]
            if focus is not None and f != focus:
                raise SystemExit(f"{p.stem}: focus classes {f} differ from {focus}")
            focus, gts = f, g
            sets[fam].append(preds)
        print(f"{fam}: {len(paths)} seeds")
    return sets, gts, focus, families[0]


def cards_mode(tags: list[str]):
    sets, gts = {}, _gts("test")
    for tag in tags:
        valid = pd.read_parquet(OUTPUTS / "cards" / f"scored_valid{tag}.parquet")
        test = pd.read_parquet(OUTPUTS / "cards" / f"scored_test{tag}.parquet")
        fused = [m.predict_proba(fusion_features(test))[:, 1] for m in fit_fusion(valid)]
        video = test["game"] + "#" + test["half"].astype(str)
        name = tag.lstrip("_") or "hindsight"

        def table(cols):
            return pd.concat([pd.DataFrame({"video": video, "time_s": test["center_s"], "label": c, "confidence": col})
                              for c, col in zip(CARD_CLASSES, cols)], ignore_index=True)

        if not sets:
            sets["spotter_only"] = [table([test["sp_yellow"], test["sp_red"]])]
        sets[f"visual_{name}"] = [table([test["pv_yellow"], test["pv_red"]])]
        sets[f"fused_{name}"] = [table(fused)]
    return sets, gts, CARD_CLASSES, f"fused_{tags[0].lstrip('_') or 'hindsight'}"


def main(mode: str, names: list[str]):
    sets, gts, classes, baseline = (spotter_mode if mode == "spotter" else cards_mode)(names)
    sets = {k: [_rel_videos(t) for t in v] for k, v in sets.items()}
    aps, diffs = compare(sets, _rel_videos(gts), classes, _test_games(), baseline, n_boot=N_BOOT)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"{mode}_" + "_vs_".join(n.strip("_") or "hindsight" for n in names)
    aps.to_csv(OUT / f"{stem}_ap.csv", index=False)
    diffs.to_csv(OUT / f"{stem}_diff.csv", index=False)
    pd.set_option("display.width", 200)
    show = aps if len(classes) <= 4 else aps[aps["class"] == "mean"]
    print(f"\ntest AP (mean over tolerances 5/10/30/60 s), 95% CI from {N_BOOT} match resamples:")
    print(show.round(3).to_string(index=False))
    print(f"\nminus {baseline}, same resamples (p_not_better < 0.05: unlikely to be noise):")
    print((diffs if len(classes) <= 4 else diffs[diffs["class"] == "mean"]).round(3).to_string(index=False))
    print(f"\n-> {OUT / stem}_{{ap,diff}}.csv")


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("spotter", "cards"):
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2:])
