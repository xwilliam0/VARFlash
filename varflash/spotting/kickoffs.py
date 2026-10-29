"""Propose kickoff / second-half video timestamps for a full-match video from spotter output.

A full broadcast (e.g. a FIFA+ replay) has its own clock; the backtest needs to know
at which video second each half kicked off. SoccerNet's "Kick-off" class fires at the
start of each half and after every goal, so we rank kick-off peaks mostly by how quiet
the preceding minutes were (pre-match build-up / half-time break), then by confidence,
then require the second half to start 45-75 minutes after the first. These are
*proposals*: check them in the video before recording them as verified.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import ROOT

VIDEO_KICKOFFS_CSV = ROOT / "config" / "video_kickoffs.csv"
COLUMNS = ["game", "video", "video_kickoff_s", "video_second_half_s", "verified", "note"]


def _quietness(events: pd.DataFrame, t: float, lookback_s: float = 300.0) -> float:
    """1 / (1 + number of confident non-kick-off events in the preceding ``lookback_s``)."""
    busy = events[(events["label"] != "Kick-off") & (events["confidence"] >= 0.3)
                  & (events["time_s"] < t) & (events["time_s"] >= t - lookback_s)]
    return 1.0 / (1.0 + len(busy))


def propose_kickoffs(events: pd.DataFrame, top: int = 3) -> dict[str, pd.DataFrame]:
    """``events``: spotter output over the whole video (time_s, label, confidence)."""
    ko = events[events["label"] == "Kick-off"].copy()
    if ko.empty:
        return {"first_half": ko, "second_half": ko}
    ko["quiet"] = [_quietness(events, t) for t in ko["time_s"]]
    # Quietness dominates: a half starts after minutes without play, and the spotter's
    # kick-off score right after a break is weak (no preceding play for context).
    ko["score"] = (ko["confidence"] + 0.02) * ko["quiet"]
    # The first half kick-off is the earliest strong one; prefer early candidates.
    first = ko.assign(score=ko["score"] / (1 + ko["time_s"] / 1800)).nlargest(top, "score")
    t1 = float(first["time_s"].iloc[0])
    window = ko[(ko["time_s"] >= t1 + 45 * 60) & (ko["time_s"] <= t1 + 75 * 60)]
    second = window.nlargest(top, "score")
    cols = ["time_s", "confidence", "quiet", "score"]
    return {"first_half": first[cols].reset_index(drop=True), "second_half": second[cols].reset_index(drop=True)}


def record(game: str, video: str, kickoff_s: float, second_half_s: float | None, verified: bool,
           note: str = "", path: Path = VIDEO_KICKOFFS_CSV) -> None:
    df = pd.read_csv(path, dtype=str) if path.exists() else pd.DataFrame(columns=COLUMNS)
    df = df[~((df["game"] == game) & (df["video"] == video))]
    row = {"game": game, "video": video, "video_kickoff_s": f"{kickoff_s:.1f}",
           "video_second_half_s": "" if second_half_s is None else f"{second_half_s:.1f}",
           "verified": str(bool(verified)).lower(), "note": note}
    pd.concat([df, pd.DataFrame([row])], ignore_index=True)[COLUMNS].to_csv(path, index=False)


def lookup(game: str, path: Path = VIDEO_KICKOFFS_CSV) -> tuple[float | None, float | None]:
    """Verified (video_kickoff_s, video_second_half_s) for a game key such as '26JUL11ARGSUI'."""
    if not path.exists():
        return None, None
    df = pd.read_csv(path, dtype=str).fillna("")
    df = df[(df["game"] == game) & (df["verified"] == "true")]
    if df.empty:
        return None, None
    r = df.iloc[-1]
    return float(r["video_kickoff_s"]), float(r["video_second_half_s"]) if r["video_second_half_s"] else None


def evaluate_on_soccernet(model, ckpt, games, gap_s: float = 900.0) -> pd.DataFrame:
    """Sanity check: glue each SoccerNet game's halves into one video with a fake half-time
    break, propose kickoffs, and compare against the labelled first kick-off of each half."""
    from .data import FPS, half_events, load_halves
    from .infer import spot

    rows = []
    for g in games:
        h1, h2 = load_halves([g], radius=0)
        gap = np.zeros((int(gap_s * FPS), h1.features.shape[1]), np.float32)
        feats = np.concatenate([np.asarray(h1.features), gap, np.asarray(h2.features)])
        prop = propose_kickoffs(spot(model, feats, ckpt, min_score=0.05))
        ev = half_events(g)
        ko = [min((f for f, c in ev[h] if c == 1), default=None) for h in (1, 2)]  # class 1 = Kick-off
        offset2 = (len(h1.features) + len(gap)) / FPS
        truth = [None if ko[0] is None else ko[0] / FPS, None if ko[1] is None else ko[1] / FPS + offset2]
        got = [p["time_s"].iloc[0] if len(p) else None for p in (prop["first_half"], prop["second_half"])]
        rows.append({"game": Path(g).name, "true_1": truth[0], "pred_1": got[0], "true_2": truth[1], "pred_2": got[1]})
    df = pd.DataFrame(rows)
    for h in (1, 2):
        df[f"err_{h}_s"] = (df[f"pred_{h}"] - df[f"true_{h}"]).abs()
    return df
