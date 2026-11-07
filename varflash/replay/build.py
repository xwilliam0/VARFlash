"""Assemble a replay session for the GUI (``varflash replay build``).

A session is one browser-playable ``video.mp4`` plus ``session.json`` in which every
time is in seconds of that video: the live (causal) model's per-class scores and
alerts, optional hindsight-model scores, reference events, and, when a Kalshi event
is given, the market's quotes/trades and the strategy's decisions and fills, mapped
onto the video with ``MatchClock``.

The spotter does not say which team scored or was sent off, so an alert is only
traded when it matches a reference event (SoccerNet labels or a match-report CSV) of
the same type within ``match_tol_s``. Unmatched alerts are shown but not traded.
"""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import MODELS, OUTPUTS
from ..spotting.data import FEATURE_FILE, FPS, LABEL_FILE

log = logging.getLogger(__name__)

SESSIONS = OUTPUTS / "replay"
LIVE_MODEL = MODELS / "live_d10.pt"
HINDSIGHT_MODEL = MODELS / "spotter.pt"
SENDING_OFF = ("Red card", "Yellow->red card")
MARKET_EVENTS = {"Goal": "goal", "Sending off": "red_card"}
SCORECARD_CLASSES = ("Goal", "Sending off", "Yellow card", "Penalty")
REPORT_LABELS = {"goal": "Goal", "red_card": "Sending off", "goal_disallowed": "Goal disallowed"}
BROWSER_CODECS = {"h264", "vp9", "av1"}
MARKET_GRID_S = 1.0
REGULATION_MINUTE = 90.0
ALERT_MIN_SCORE = 0.05


def display_label(label: str) -> str:
    return "Sending off" if label in SENDING_OFF else label


# ---- video ----------------------------------------------------------------


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def probe(path: Path) -> tuple[float, str]:
    """(duration in seconds, video codec) from ffmpeg's header dump."""
    err = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    d = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    c = re.search(r"Video: (\w+)", err)
    if not d or not c:
        raise ValueError(f"cannot read {path}: {err[-300:]}")
    h, m, s = d.groups()
    return int(h) * 3600 + int(m) * 60 + float(s), c.group(1)


def make_video(videos: list[Path], out: Path) -> list[float]:
    """Concatenate ``videos`` into one faststart MP4, re-encoding only if the browser can't
    play the codec. Returns each input's start offset (s) in the output."""
    info = [probe(v) for v in videos]
    offsets = [float(x) for x in np.cumsum([0.0] + [d for d, _ in info[:-1]])]
    if all(c in BROWSER_CODECS for _, c in info):
        vcodec = ["-c:v", "copy"]
    else:
        vcodec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]
    with tempfile.TemporaryDirectory() as tmp:
        if len(videos) == 1:
            inputs = ["-i", str(videos[0])]
        else:
            lst = Path(tmp) / "list.txt"
            lst.write_text("".join("file '{}'\n".format(str(v.resolve()).replace("'", "'\\''")) for v in videos))
            inputs = ["-f", "concat", "-safe", "0", "-i", str(lst)]
        cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", *inputs, *vcodec,
               "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(out)]
        subprocess.run(cmd, check=True)
    return offsets


def make_clip_video(source: Path, clips: pd.DataFrame, out: Path, pre_s: float, post_s: float) -> list[dict]:
    """One real-time segment per anchored clip of ``source``: the clip's first frame frozen for
    ``pre_s``, the clip, then its last frame frozen for ``post_s``, so the market keeps running
    around each event. ``clips`` columns: clip_start_s, clip_end_s, event_s, event_utc, minute,
    label, team, note."""
    segments, parts, t = [], [], 0.0
    with tempfile.TemporaryDirectory() as tmp:
        for i, c in enumerate(clips.itertuples()):
            dur = c.clip_end_s - c.clip_start_s
            part = Path(tmp) / f"{i}.mp4"
            cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", str(c.clip_start_s), "-t", str(dur), "-i", str(source),
                   "-vf", f"tpad=start_mode=clone:start_duration={pre_s}:stop_mode=clone:stop_duration={post_s},fps=25",
                   "-af", f"adelay={int(pre_s * 1000)}:all=1,apad=whole_dur={pre_s + dur + post_s}",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-b:a", "96k", str(part)]
            subprocess.run(cmd, check=True)
            total, _ = probe(part)
            segments.append({
                "start_s": t, "end_s": t + total, "footage": [t + pre_s, t + pre_s + dur],
                "anchor_video_s": t + pre_s + (c.event_s - c.clip_start_s), "anchor_utc": pd.Timestamp(c.event_utc),
                "minute": float(c.minute), "label": c.label, "team": c.team, "note": c.note,
            })
            parts.append(part)
            t += total
        lst = Path(tmp) / "list.txt"
        lst.write_text("".join(f"file '{p}'\n" for p in parts))
        subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-c", "copy", "-movflags", "+faststart", str(out)], check=True)
    return segments


def load_features(videos: list[Path], features: Path | None, out_dir: Path, device: str) -> list[np.ndarray]:
    if features is None:
        from ..spotting.features import FeatureExtractor

        features = out_dir / "features"
        FeatureExtractor(device=device).extract_match(videos, features)
    if features.is_file():
        return [np.load(features)]
    return [np.load(features / FEATURE_FILE.format(half=h)) for h in range(1, len(videos) + 1)]


# ---- model ----------------------------------------------------------------


def run_model(path: Path, feats: list[np.ndarray], offsets: list[float], n_frames: int,
              device: str) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict]:
    """Event-aligned scores per display class over the whole video, plus alerts (peaks)
    with ``time_s`` (the incident) and ``fire_s`` (when a live system would know)."""
    from ..spotting.infer import event_scores, frame_scores, load_spotter, peak_pick

    model, ckpt = load_spotter(path, device)
    classes, delay = ckpt["classes"], ckpt["delay"]
    window = ckpt.get("config", {}).get("window", 120)
    full = np.zeros((n_frames, len(classes)), np.float32)
    alerts = []
    for x, off in zip(feats, offsets):
        s = event_scores(frame_scores(model, x, window), delay)
        i = int(round(off * FPS))
        n = min(len(s), n_frames - i)
        full[i : i + n] = s[:n]
        ev = peak_pick(s, classes, min_score=ALERT_MIN_SCORE, calibrators=ckpt.get("calibrators"))
        ev["time_s"] += off
        ev["fire_s"] = ev["time_s"] + delay / FPS
        alerts.append(ev)
    scores: dict[str, np.ndarray] = {}
    for k, c in enumerate(classes):
        d = display_label(c)
        scores[d] = np.maximum(scores[d], full[:, k]) if d in scores else full[:, k]
    alerts = pd.concat(alerts, ignore_index=True)
    alerts["label"] = alerts["label"].map(display_label)
    meta = {"path": str(path), "causal": bool(getattr(model, "causal", False)), "delay_s": delay / FPS,
            "classes": list(scores)}
    return scores, alerts.sort_values("fire_s").reset_index(drop=True), meta


def score_model(path: Path, feats: list[np.ndarray], offsets: list[float], n_frames: int, video: Path,
                device: str) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict]:
    """Feature spotter (``run_model``) or visual goal detector (runs on the session video)."""
    from ..goals.stream import is_goal_vision, run_goal_vision

    if is_goal_vision(path):
        return run_goal_vision(path, video, n_frames, FPS, device)
    return run_model(path, feats, offsets, n_frames, device)


def model_label(meta: dict) -> str:
    name = Path(meta["path"]).stem
    if meta.get("kind") == "vision":
        return f"Visual goal detector, goals only ({name})"
    if not meta["causal"]:
        return f"Feature spotter, not causal ({name})"
    return f"Feature spotter, {meta['delay_s']:g}s look-ahead ({name})"


# ---- reference events -------------------------------------------------------


def soccernet_truth(labels: Path, offsets: list[float]) -> tuple[pd.DataFrame, dict]:
    a = json.loads(Path(labels).read_text())
    rows = []
    for x in a["annotations"]:
        half = int(x["gameTime"].split("-")[0])
        if half > len(offsets):
            continue
        rows.append({
            "video_s": offsets[half - 1] + int(x["position"]) / 1000,
            "label": display_label(x["label"]),
            "team": x["team"] if x.get("team") in ("home", "away") else None,
            "visible": x.get("visibility") == "visible",
        })
    # Some games annotate the same goal twice, a fraction of a second apart.
    truth = pd.DataFrame(rows).sort_values(["label", "video_s"])
    dup = (truth["label"] == truth["label"].shift()) & (truth["video_s"].diff() < 2.0)
    truth = truth[~dup].sort_values("video_s").reset_index(drop=True)
    return truth, {"home": a.get("gameHomeTeam"), "away": a.get("gameAwayTeam")}


def report_truth(path: Path, video_kickoff_s: float, video_second_half_s: float | None) -> pd.DataFrame:
    """A ``varflash backtest --events`` CSV (event, team, video_s or game_time)."""
    from ..backtest import parse_game_time

    ev = pd.read_csv(path)
    if "video_s" not in ev:
        def to_video(gt: str) -> float:
            half, secs = parse_game_time(gt)
            start = video_kickoff_s if half == 1 else video_second_half_s
            return math.nan if start is None else start + secs

        ev["video_s"] = ev["game_time"].map(to_video)
    return pd.DataFrame({"video_s": ev["video_s"], "label": ev["event"].map(REPORT_LABELS).fillna(ev["event"]),
                         "team": ev.get("team"), "visible": True})


def attach_teams(alerts: pd.DataFrame, truth: pd.DataFrame, tol_s: float) -> pd.DataFrame:
    """Team (and matched reference time) for each alert, from the nearest same-type reference event."""
    alerts = alerts.copy()
    alerts["team"], alerts["ref_s"] = None, math.nan
    if truth.empty:
        return alerts
    ref = truth[truth["team"].notna()]
    for i, a in alerts.iterrows():
        cand = ref[(ref["label"] == a["label"]) & ((ref["video_s"] - a["time_s"]).abs() <= tol_s)]
        if len(cand):
            j = (cand["video_s"] - a["time_s"]).abs().idxmin()
            alerts.at[i, "team"], alerts.at[i, "ref_s"] = cand.at[j, "team"], cand.at[j, "video_s"]
    return alerts


def scorecard(alerts: pd.DataFrame, truth: pd.DataFrame, min_conf: float, tol_s: float) -> list[dict]:
    """Per class: how many reference events the live alerts caught, how late, and how many false alarms."""
    rows = []
    for c in SCORECARD_CLASSES:
        t = truth[truth["label"] == c]["video_s"].to_numpy() if len(truth) else np.array([])
        a = alerts[(alerts["label"] == c) & (alerts["confidence"] >= min_conf)]
        if not len(t) and not len(a):
            continue
        lat = []
        for ts in t:
            hit = a[(a["time_s"] - ts).abs() <= tol_s]
            if len(hit):
                lat.append(float(hit["fire_s"].min() - ts))
        false = sum(not np.any(np.abs(t - x) <= tol_s) for x in a["time_s"]) if len(t) else len(a)
        rows.append({"label": c, "events": len(t), "caught": len(lat), "alerts": len(a), "false_alarms": int(false),
                     "median_latency_s": float(np.median(lat)) if lat else None})
    return rows


# ---- market and strategy ----------------------------------------------------


def market_section(m, clock, book, duration_s: float) -> dict:
    from ..market import implied_probs

    roles = m.outcome_roles
    grid = np.arange(0.0, duration_s + MARKET_GRID_S, MARKET_GRID_S)
    utc = [clock.video_to_utc(t) for t in grid]
    probs = implied_probs(m, "trades", freq=f"{MARKET_GRID_S:g}s")
    out = {"prob": {}, "bid": {}, "ask": {}}
    for r in roles:
        ticker = m.ticker_for(r)
        out["prob"][r] = [math.nan if u is None else probs[r].asof(u) for u in utc]
        quotes = []
        for u in utc:
            try:
                quotes.append((math.nan, math.nan) if u is None else book.at(ticker, u))
            except LookupError:
                quotes.append((math.nan, math.nan))
        out["bid"][r], out["ask"][r] = [q[0] for q in quotes], [q[1] for q in quotes]

    known = [u for u in utc if u is not None]
    t = m.trades.assign(role=m.trades["ticker"].map(m.roles))
    t = t[t["role"].isin(roles) & (t["ts"] >= min(known)) & (t["ts"] <= max(known))]
    v = np.array([clock.utc_to_video(x) for x in t["ts"]], dtype=float)
    keep = np.isfinite(v)
    t, v = t[keep], v[keep]
    names = {m.roles[x["ticker"]]: x.get("yes_sub_title") or m.roles[x["ticker"]] for x in m.markets
             if m.roles.get(x["ticker"]) in roles}
    return {
        "event_ticker": m.event_ticker, "title": m.title, "roles": roles, "names": names, "result": m.result,
        "kickoff_utc": m.kickoff, "kickoff_is_estimate": m.kickoff_is_estimate, "grid_s": MARKET_GRID_S, **out,
        "tape": {"v": v.round(2), "role": t["role"].tolist(), "price": t["yes_price"].round(3),
                 "count": t["count"], "side": t["taker_side"].tolist()},
    }


def trade_section(m, clock, book, alerts: pd.DataFrame, cfg) -> tuple[list, list, dict]:
    from ..backtest import generate_orders, markouts, run_orders, summarize, video_events_to_utc
    from ..market import implied_probs

    tradable = alerts[alerts["label"].isin(list(MARKET_EVENTS)) & alerts["team"].notna() & alerts["trade_eligible"]]
    if tradable.empty:
        return [], [], summarize(pd.DataFrame())
    ev = pd.DataFrame({
        "video_s": tradable["fire_s"], "incident_video_s": tradable["time_s"],
        "event": tradable["label"].map(MARKET_EVENTS), "team": tradable["team"],
        "confidence": tradable["confidence"], "alert_id": tradable["id"],
    })
    ev = video_events_to_utc(ev, clock)
    orders, dec = generate_orders(m, ev, clock, cfg, book=book)
    dec["alert_id"] = ev["alert_id"].to_numpy()
    dec = markouts(dec, implied_probs(m, "trades"))
    fills = run_orders(m, orders, book)

    decisions = []
    for d in dec.to_dict("records"):
        decisions.append({
            "alert_id": d["alert_id"], "video_s": clock.utc_to_video(d["ts"]), "minute": d["minute"],
            "event": d["event"], "team": d["team"], "confidence": d["confidence"], "role": d["role"],
            "side": d["side"], "price": d["price"], "edge": d["edge"], "contracts": d["contracts"],
            "fair": {r: d[f"fair_{r}"] for r in m.outcome_roles},
            "markouts": {k.removeprefix("markout_"): v for k, v in d.items() if k.startswith("markout_")},
        })
    traded = [d for d in decisions if d["contracts"] > 0]
    role_of = {m.ticker_for(r): r for r in m.outcome_roles}
    out_fills = []
    for f, d in zip(fills.to_dict("records"), traded):
        out_fills.append({
            "alert_id": d["alert_id"], "entry_s": clock.utc_to_video(f["entry_ts"]),
            "exit_s": clock.utc_to_video(f["exit_ts"]), "role": role_of[f["ticker"]], "ticker": f["ticker"],
            "side": f["side"], "contracts": f["contracts"], "entry_px": f["entry_px"], "exit_px": f["exit_px"],
            "fees": f["fees"], "pnl": f["pnl"], "reason": f["reason"],
        })
    return decisions, out_fills, summarize(fills)


# ---- session ----------------------------------------------------------------


def _clean(x):
    """JSON-safe: numpy -> python, NaN -> null, timestamps -> ISO strings."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        return [_clean(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if not math.isfinite(x) else round(float(x), 4)
    if isinstance(x, pd.Timestamp):
        return None if pd.isna(x) else x.isoformat()
    if x is pd.NaT:
        return None
    return x


def _team_names(title: str) -> dict:
    """'Argentina vs Switzerland: Regulation Time Moneyline' -> {'home': 'Argentina', 'away': 'Switzerland'}."""
    m = re.match(r"\s*(.+?)\s+vs\.?\s+(.+?)(?::|$)", title)
    return {"home": m.group(1), "away": m.group(2)} if m else {}


def _scores_json(scores: dict[str, np.ndarray]) -> dict[str, list[int]]:
    """Scores as integers in thousandths to keep session.json small."""
    return {c: np.round(s * 1000).astype(int).tolist() for c, s in scores.items()}


def build_session(
    name: str,
    videos: list[Path],
    features: Path | None = None,
    labels: Path | None = None,
    report: Path | None = None,
    live_model: Path = LIVE_MODEL,
    extra_models: list[Path] = (),
    hindsight_model: Path | None = HINDSIGHT_MODEL,
    event: str | None = None,
    kickoff: str | None = None,
    video_kickoff_s: float | None = None,
    video_second_half_s: float | None = None,
    strategy=None,
    trade_min_confidence: float = 0.3,
    match_tol_s: float = 60.0,
    note: str = "",
    clips: Path | None = None,
    clip_pad_s: tuple[float, float] = (45.0, 150.0),
    device: str | None = None,
    root: Path = SESSIONS,
) -> Path:
    """``clips``: build from a highlights video instead, one real-time segment per anchored clip
    (see ``make_clip_video``); ``clip_pad_s`` is the frozen lead-in / run-out around each clip.
    ``extra_models``: more live models the GUI can switch to; each gets its own alerts and trades."""
    import torch

    from ..backtest import MatchClock, StrategyConfig
    from ..backtest.clock import SegmentClock

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    videos = [Path(v) for v in videos]

    video = out / "video.mp4"
    segments = None
    if clips:
        if not event:
            raise ValueError("clips are anchored to wall-clock time; give the Kalshi event too")
        segments = make_clip_video(videos[0], pd.read_csv(clips), video, *clip_pad_s)
        offsets, videos, features = [0.0], [video], None  # features must come from the stitched video
    else:
        offsets = make_video(videos, video)
    duration_s, _ = probe(video)
    log.info("video: %d part(s), %.1f min -> %s", len(segments or videos), duration_s / 60, video)

    feats = load_features(videos, features, out, device)
    n_frames = int(math.ceil(duration_s * FPS))
    raw = [score_model(Path(p), feats, offsets, n_frames, video, device) for p in [live_model, *extra_models]]
    hindsight = None
    if hindsight_model:
        h_scores, h_alerts, h_meta = run_model(hindsight_model, feats, offsets, n_frames, device)
        hindsight = {"model": h_meta, "scores": _scores_json(h_scores),
                     "alerts": h_alerts[["time_s", "label", "confidence"]].to_dict("records")}

    teams: dict = {}
    truth = pd.DataFrame(columns=["video_s", "label", "team", "visible"])
    if labels:
        truth, teams = soccernet_truth(labels, offsets)
    if segments:
        truth = pd.DataFrame([{"video_s": s["anchor_video_s"], "label": s["label"], "team": s["team"], "visible": True}
                              for s in segments])

    m = None
    if event:
        from ..backtest.engine import QuoteBook
        from ..kalshi.store import find_event_dir, game_key, load_match
        from ..spotting.kickoffs import lookup

        m = load_match(find_event_dir(event), kickoff=kickoff)
        known = lookup(game_key(m.event_ticker))
        video_kickoff_s = video_kickoff_s if video_kickoff_s is not None else known[0]
        video_second_half_s = video_second_half_s if video_second_half_s is not None else known[1]
    if video_kickoff_s is None:
        video_kickoff_s = offsets[0]
    if video_second_half_s is None and len(offsets) > 1:
        video_second_half_s = offsets[1]
    if report:
        truth = pd.concat([truth, report_truth(report, video_kickoff_s, video_second_half_s)], ignore_index=True)

    clock = None
    if m is not None:
        clock = MatchClock(m.kickoff, m.second_half, video_kickoff_s, video_second_half_s)
        if segments:
            clock = SegmentClock(clock, segments)
        teams = teams or _team_names(m.title)

    book = QuoteBook(m) if m is not None else None
    cfg = strategy or StrategyConfig()

    def model_entry(scores: dict, alerts: pd.DataFrame, meta: dict) -> dict:
        alerts = alerts.sort_values("fire_s").reset_index(drop=True)
        alerts.insert(0, "id", range(len(alerts)))
        alerts = attach_teams(alerts, truth, match_tol_s)
        alerts["no_trade"] = None
        if segments:
            footage = [s["footage"] for s in segments]
            frozen = ~alerts["time_s"].map(lambda t: any(a <= t <= b for a, b in footage))
            minute = alerts["time_s"].map(lambda t: clock.match_minute(clock.video_to_utc(t)))
            alerts.loc[minute > REGULATION_MINUTE, "no_trade"] = "extra time: in-play model covers regulation only"
            alerts.loc[frozen, "no_trade"] = "frozen frame, no footage"
        alerts["trade_eligible"] = (alerts["confidence"] >= trade_min_confidence) & alerts["no_trade"].isna()
        entry = {"key": Path(meta["path"]).stem, "label": model_label(meta),
                 "live": {"model": meta, "scores": _scores_json(scores)},
                 "alerts": alerts.drop(columns=["frame"]).to_dict("records"),
                 "scorecard": scorecard(alerts, truth, trade_min_confidence, match_tol_s),
                 "decisions": [], "fills": [], "summary": None}
        if m is not None:
            entry["decisions"], entry["fills"], entry["summary"] = trade_section(m, clock, book, alerts, cfg)
        return entry

    models = [model_entry(*r) for r in raw]
    session = {
        "version": 2, "name": name, "note": note,
        "title": m.title if m else (f"{teams['home']} vs {teams['away']}" if teams else name),
        "teams": teams, "video": "video.mp4", "duration_s": duration_s,
        "segments": [{"start_s": o, "source": str(v)} for o, v in zip(offsets, videos)],
        "clock": {"video_kickoff_s": video_kickoff_s, "video_second_half_s": video_second_half_s},
        "fps": FPS, "hindsight": hindsight,
        "truth": truth.sort_values("video_s").to_dict("records"),
        "trade_min_confidence": trade_min_confidence,
        "market": None,
        # The first model is the default; the GUI copies the chosen entry's fields to the top level.
        **{k: models[0][k] for k in ("live", "alerts", "scorecard", "decisions", "fills", "summary")},
        "models": [{"key": e["key"], "label": e["label"]} for e in models],
    }
    if segments:
        session["clock"]["segments"] = segments
    if m is not None:
        session["clock"].update(kickoff_utc=clock.kickoff_utc, second_half_utc=clock.second_half_utc)
        session["market"] = market_section(m, clock, book, duration_s)
        session["strategy"] = {k: str(v) if isinstance(v, pd.Timedelta) else v for k, v in vars(cfg).items()}
    for e in models[1:]:
        (out / f"model_{e['key']}.json").write_text(json.dumps(_clean(e), separators=(",", ":")))

    (out / "session.json").write_text(json.dumps(_clean(session), separators=(",", ":")))
    log.info("session -> %s", out)
    return out


def soccernet_game_inputs(game_dir: Path) -> dict:
    """Videos, features and labels of one downloaded SoccerNet game (both halves)."""
    videos = [game_dir / f"{h}_224p.mkv" for h in (1, 2)]
    missing = [v for v in videos if not v.exists()]
    if missing:
        raise FileNotFoundError(f"missing {missing}; download with `varflash spotting download --files 1_224p.mkv 2_224p.mkv`")
    has_feats = all((game_dir / FEATURE_FILE.format(half=h)).exists() for h in (1, 2))
    labels = game_dir / LABEL_FILE
    return {"videos": videos, "features": game_dir if has_feats else None, "labels": labels if labels.exists() else None}
