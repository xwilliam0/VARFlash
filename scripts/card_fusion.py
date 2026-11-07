"""Which second-stage card scorer ranks spotter candidates best? Fusion variants with bootstrap CIs.

Every variant is a per-class logistic regression on logit features of the same candidates
(from ``cards build-clips``), so differences come only from the scores fed to it:

    base       spotter yellow/red + visual yellow/red (what ``cards rescore`` fits today;
               the untagged visual scores are seed 1, card_vision_s1.pt)
    tta        base with seed 1 also scored on the mirrored clip
    ens2       visual scores averaged over seeds 0 and 1
    ens3       ... over seeds 0, 1 and 2
    ens_tta    seed ensemble with mirroring
    ctx        base + what the spotter sees around the candidate (foul, direct free kick,
               penalty, substitution) + how many yellows came earlier in the match
    newspot    base with yellow/red from the current deployed spotter instead of the one
               that proposed the candidates
    bothspot   base + the deployed spotter's yellow/red as two more features
    best       newspot with the largest available visual ensemble

Two evaluations, both with match-level 95% CIs and paired gains over ``base``:
    split      fit on valid, score test (the protocol behind the published numbers)
    pooled     5-fold cross-fit by match over valid + test, about twice the sending-offs

    python scripts/card_fusion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bootstrap_compare import _rel  # noqa: E402

from varflash.cards.clips import CARD_CLASSES  # noqa: E402
from varflash.cards.rescore import _gts, _truth  # noqa: E402
from varflash.cards.rescore import spotter_at_candidates as _spotter_at  # noqa: E402
from varflash.paths import MODELS, OUTPUTS  # noqa: E402
from varflash.spotting.bootstrap import compare  # noqa: E402
from varflash.spotting.data import available_games  # noqa: E402

CARDS = OUTPUTS / "cards"
OUT = OUTPUTS / "bootstrap"
PROPOSER = MODELS / "sweep" / "multitask_bce_s1.pt"  # models/cards.pt when the candidates were built
DEPLOYED = MODELS / "cards.pt"
CONTEXT = ["Foul", "Direct free-kick", "Penalty", "Substitution"]
N_BOOT = 1000


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def spotter_at_candidates(path: Path, split: str, cands: pd.DataFrame) -> pd.DataFrame:
    """Every spotter class's score around each candidate, cached per split and spotter."""
    cache = CARDS / f"context_{split}_{path.stem}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    out = _spotter_at(path, split, cands)
    out.to_parquet(cache)
    return out


def prior_yellows(cands: pd.DataFrame, gap_s: float = 30.0) -> np.ndarray:
    """Spotter yellow mass earlier in the same match (more than ``gap_s`` before): a second yellow needs a first."""
    out = np.zeros(len(cands))
    key = cands["half"] * 10_000 + cands["center_s"]
    for _, d in cands.assign(key=key).groupby("game"):
        d = d.sort_values("key")
        k, y = d["key"].to_numpy(), d["sp_yellow"].to_numpy()
        out[d.index] = [y[k < kk - gap_s].sum() for kk in k]
    return out


def load_split(split: str) -> pd.DataFrame:
    df = pd.read_parquet(CARDS / f"scored_{split}.parquet").reset_index(drop=True)
    for tag in ("_s0", "_s2", "_s0_tta", "_s1_tta", "_s2_tta"):
        p = CARDS / f"scored_{split}{tag}.parquet"
        if p.exists():
            o = pd.read_parquet(p).reset_index(drop=True)
            if not (o["clip_id"].to_numpy() == df["clip_id"].to_numpy()).all():
                raise SystemExit(f"{p} has different candidates")
            df[f"pv_yellow{tag}"], df[f"pv_red{tag}"] = o["pv_yellow"], o["pv_red"]
    proposer = spotter_at_candidates(PROPOSER, split, df)
    gap = np.abs(proposer.loc[df.index, "Yellow card"].to_numpy() - df["sp_yellow"].to_numpy()).max()
    if gap > 1e-3:
        raise SystemExit(f"{PROPOSER.name} does not reproduce the saved candidate scores (max gap {gap:.3g})")
    for c in CONTEXT:
        df[f"ctx_{c}"] = proposer.loc[df.index, c].to_numpy()
    deployed = spotter_at_candidates(DEPLOYED, split, df)
    df["new_yellow"], df["new_red"] = deployed.loc[df.index, "Yellow card"], deployed.loc[df.index, "Sending off"]
    df["prior_yellow"] = prior_yellows(df)
    df["split"] = split
    y = _truth(df)
    df["y_yellow"], df["y_red"] = y[:, 0], y[:, 1]
    return df


def _visual(df: pd.DataFrame, tags: list[str]) -> tuple[np.ndarray, np.ndarray]:
    return (np.mean([df[f"pv_yellow{t}"] for t in tags], 0), np.mean([df[f"pv_red{t}"] for t in tags], 0))


def variants(df: pd.DataFrame) -> dict[str, np.ndarray]:
    sp = [logit(df["sp_yellow"]), logit(df["sp_red"])]
    new = [logit(df["new_yellow"]), logit(df["new_red"])]
    ctx = [logit(df[f"ctx_{c}"]) for c in CONTEXT] + [np.log1p(df["prior_yellow"].to_numpy())]
    vis = lambda tags: [logit(v) for v in _visual(df, tags)]  # noqa: E731
    have = lambda *tags: all(t == "" or f"pv_red{t}" in df for t in tags)  # noqa: E731
    visual_sets = {"base": [""], "tta": ["_s1_tta"], "ens2": ["", "_s0"], "ens3": ["", "_s0", "_s2"],
                   "ens_tta": [t for t in ("_s0_tta", "_s1_tta", "_s2_tta") if have(t)]}
    out, best = {}, [""]
    for name, tags in visual_sets.items():
        if have(*tags) and (len(tags) >= 2 or name in ("base", "tta")):
            out[name] = sp + vis(tags)
            if name != "tta" and len(tags) >= len(best):
                best = tags
    out["ctx"] = sp + vis([""]) + ctx
    out["newspot"] = new + vis([""])
    out["bothspot"] = sp + new + vis([""])
    if best != [""]:
        out["best"] = new + vis(best)
    return {k: np.column_stack(v) for k, v in out.items()}


def fit_predict(X_fit, y_fit, X_new) -> np.ndarray:
    m = LogisticRegression(C=1.0, max_iter=2000).fit(X_fit, y_fit)
    return m.predict_proba(X_new)[:, 1]


def predictions(df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    video = (df["game"] + "#" + df["half"].astype(str)).map(_rel).to_numpy()
    return pd.concat([pd.DataFrame({"video": video, "time_s": df["center_s"].to_numpy(), "label": c,
                                    "confidence": scores[:, k]}) for k, c in enumerate(CARD_CLASSES)],
                     ignore_index=True)


def run(valid: pd.DataFrame, test: pd.DataFrame, gts: pd.DataFrame, games: list[str], how: str):
    both = pd.concat([valid, test], ignore_index=True)
    feats = variants(both)
    ys = both[["y_yellow", "y_red"]].to_numpy()
    sets = {}
    for name, X in feats.items():
        if how == "split":
            fit, new = (both["split"] == "valid").to_numpy(), (both["split"] == "test").to_numpy()
            p = np.column_stack([fit_predict(X[fit], ys[fit, k], X[new]) for k in range(2)])
            sets[name] = [predictions(both[new], p)]
        else:
            game_keys = both["game"].map(_rel)
            uniq = np.array(sorted(game_keys.unique()))
            fold_of = dict(zip(uniq, np.random.default_rng(0).permutation(len(uniq)) % 5))
            fold = game_keys.map(fold_of).to_numpy()
            p = np.zeros((len(both), 2))
            for f in range(5):
                tr, te = fold != f, fold == f
                for k in range(2):
                    p[te, k] = fit_predict(X[tr], ys[tr, k], X[te])
            sets[name] = [predictions(both, p)]
    aps, diffs = compare(sets, gts, CARD_CLASSES, games, "base", n_boot=N_BOOT)
    OUT.mkdir(parents=True, exist_ok=True)
    aps.to_csv(OUT / f"card_fusion_{how}_ap.csv", index=False)
    diffs.to_csv(OUT / f"card_fusion_{how}_diff.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"\n=== {how}: AP (mean over 5/10/30/60 s), 95% CI over {len(games)} matches")
    print(aps.pivot_table(index="model", columns="class", values="value").round(3).to_string())
    print(f"\n--- gain over base (same resamples)")
    print(diffs.round(3).to_string(index=False))


def main():
    valid, test = load_split("valid"), load_split("test")
    gts = {s: _gts(s).assign(video=lambda d: d["video"].map(_rel)) for s in ("valid", "test")}
    games = {s: [_rel(str(g)) for g in available_games(s)] for s in ("valid", "test")}
    print(f"valid: {len(valid)} candidates ({valid.y_red.sum()} near a sending-off); "
          f"test: {len(test)} ({test.y_red.sum()})")
    run(valid, test, gts["test"], games["test"], "split")
    run(valid, test, pd.concat(gts.values(), ignore_index=True), games["valid"] + games["test"], "pooled")


if __name__ == "__main__":
    main()
