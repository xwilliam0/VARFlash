"""Command-line entry point: ``python -m varflash <group> <command> ...``."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

import pandas as pd

from .paths import KALSHI_DATA, MODELS, OUTPUTS

log = logging.getLogger("varflash")

SOCCER_PATTERN = r"soccer|world cup|fifa|premier|epl|champions league|uefa|la ?liga|serie a|bundesliga|ligue|mls|eredivisie|brasileiro"


# ---- kalshi ---------------------------------------------------------------


def cmd_kalshi_series(a):
    from .kalshi import KalshiClient

    series = pd.DataFrame(KalshiClient().list_series(a.category))[["ticker", "title"]]
    series = series[series["title"].str.contains(a.match, case=False, regex=True)]
    if a.games_only:
        series = series[series["ticker"].str.endswith("GAME")]
    print(series.sort_values("ticker").to_string(index=False))


def cmd_kalshi_events(a):
    from .kalshi import KalshiClient

    events = pd.DataFrame(KalshiClient().list_events(a.series, status=a.status))
    print(events[["event_ticker", "title"]].to_string(index=False) if len(events) else "no events")


def cmd_kalshi_pull_event(a):
    from .kalshi import KalshiClient, pull_event

    client = KalshiClient()
    for ev in a.event:
        print(pull_event(client, ev, with_trades=not a.no_trades, refresh=a.refresh))


def cmd_kalshi_pull_series(a):
    from .kalshi import KalshiClient, pull_series

    client = KalshiClient()
    for s in a.series:
        paths = pull_series(client, s, with_trades=not a.no_trades, refresh=a.refresh, limit=a.limit)
        print(f"{s}: {len(paths)} events saved")


# ---- market ---------------------------------------------------------------


def _analyze(event_dir: Path, a) -> pd.DataFrame:
    from .kalshi.store import load_match
    from .market import detect_shocks, implied_probs
    from .market.plot import plot_match

    m = load_match(event_dir, kickoff=getattr(a, "kickoff", None))
    probs = implied_probs(m, a.source)
    in_play = probs[probs.index >= m.kickoff - pd.Timedelta(minutes=10)]
    shocks = detect_shocks(in_play, window=a.window, threshold=a.threshold)
    shocks.insert(0, "event_ticker", m.event_ticker)
    shocks["minutes_after_kickoff"] = (shocks["peak"] - m.kickoff).dt.total_seconds() / 60
    shocks["team"] = shocks["role"].map(
        {r: m.ticker_for(r).split("-")[-1] for r in ("home", "away", "draw")}
    )
    if a.plot:
        plot_match(m, probs, shocks, out=OUTPUTS / f"{m.event_ticker}_market.png")
    return shocks


def cmd_market_analyze(a):
    from .kalshi.store import find_event_dir

    shocks = _analyze(find_event_dir(a.event), a)
    cols = ["peak", "minutes_after_kickoff", "team", "p_before", "p_after", "delta"]
    print(shocks[cols].to_string(index=False, float_format="%.3f"))
    if a.plot:
        print(f"plot -> {OUTPUTS / (a.event + '_market.png')}")


def cmd_market_scan(a):
    """Shock table for every downloaded event in a series (the cross-match dataset)."""
    rows = []
    dirs = sorted((KALSHI_DATA / a.series).glob("*/event.json"))
    for p in dirs:
        try:
            rows.append(_analyze(p.parent, a))
        except Exception as exc:
            log.warning("skip %s: %s", p.parent.name, exc)
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out = OUTPUTS / f"{a.series}_shocks.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{len(dirs)} events, {len(df)} shocks -> {out}")
    if len(df):
        print(df.groupby("event_ticker").size().describe().round(2).to_string())


# ---- vision ---------------------------------------------------------------


def _read_frame(a):
    import cv2

    if a.image:
        frame = cv2.imread(a.image)
    else:
        cap = cv2.VideoCapture(a.video)
        cap.set(cv2.CAP_PROP_POS_MSEC, a.time * 1000)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            frame = None
    if frame is None:
        raise SystemExit("could not read frame")
    return frame


def cmd_vision_calibrate(a):
    from .vision.homography import calibrate_interactive

    h = calibrate_interactive(_read_frame(a), a.out)
    print(f"saved {a.out}: {h.n_points} points, mean reprojection error {h.reprojection_error_m:.2f} m")


def cmd_vision_review(a):
    import cv2

    from .vision.detect import detect_image
    from .vision.homography import load_calibration
    from .vision.var_review import review_frame

    frame = _read_frame(a)
    dets = detect_image(frame, a.weights, conf=a.conf)
    receiver = tuple(float(v) for v in a.receiver.split(",")) if a.receiver else None
    r = review_frame(
        frame, dets, load_calibration(a.calib), attack_right=a.attack == "right",
        attacking_color=a.attacking_color, receiver_xy=receiver, position_sigma_m=a.sigma,
    )
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "review_overlay.jpg"), r.overlay)
    cv2.imwrite(str(out / "review_minimap.png"), r.minimap)
    r.players.to_csv(out / "review_players.csv", index=False)
    call = dataclasses.asdict(r.call)
    (out / "review_call.json").write_text(json.dumps(call, indent=2))
    print(json.dumps(call, indent=2))


def cmd_vision_track(a):
    from .vision.detect import track_video

    df = track_video(a.video, a.weights, conf=a.conf, vid_stride=a.stride, max_frames=a.max_frames)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(f"{len(df)} detections, {df['track_id'].nunique()} tracks -> {a.out}")


# ---- spotting -------------------------------------------------------------


def cmd_spotting_download(a):
    from .spotting.data import download

    download(a.splits, a.max_games, a.workers)


def cmd_spotting_train(a):
    from .spotting.train import TrainConfig, train

    cfg = TrainConfig(epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, dim=a.dim,
                      max_train_games=a.max_games, num_threads=a.threads)
    train(cfg, Path(a.out))


def cmd_spotting_eval(a):
    from .spotting.infer import load_spotter
    from .spotting.train import evaluate

    model, _ = load_spotter(a.model)
    report = evaluate(model, a.split, max_games=a.max_games)
    out = OUTPUTS / f"spotting_eval_{a.split}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)
    print(report.round(3).to_string(index=False))
    ap_cols = [c for c in report.columns if c.startswith("AP@")]
    print("\nmAP (all classes):", report[ap_cols].mean().round(3).to_dict())
    print("mAP (market-moving):", report[report["market_moving"]][ap_cols].mean().round(3).to_dict())


# ---- backtest -------------------------------------------------------------


def cmd_backtest(a):
    from .backtest import (MatchClock, StrategyConfig, generate_orders, markouts, run_orders,
                           summarize, video_events_to_utc)
    from .backtest.engine import QuoteBook
    from .kalshi.store import find_event_dir, load_match
    from .market import detect_shocks, implied_probs
    from .market.plot import plot_match

    m = load_match(find_event_dir(a.event), kickoff=a.kickoff)
    clock = MatchClock(
        m.kickoff, a.second_half or m.second_half, a.video_kickoff_s, a.video_second_half_s
    )
    raw_events = pd.read_csv(a.events)
    events = video_events_to_utc(raw_events, clock)
    cfg = StrategyConfig(
        broadcast_delay_s=a.delay, processing_latency_s=a.latency, min_edge=a.min_edge,
        bankroll=a.bankroll, hold=None if a.to_settlement else pd.Timedelta(minutes=a.hold_min),
        anchor=a.anchor,
    )
    book = QuoteBook(m)
    orders, decisions = generate_orders(m, events, clock, cfg, book=book)
    fills = run_orders(m, orders, book)
    probs = implied_probs(m, "trades")
    marks = markouts(decisions, probs) if len(decisions) else decisions

    pd.set_option("display.width", 220)
    print(f"{m.title}  kickoff {m.kickoff} ({'estimated' if m.kickoff_is_estimate else 'known'})\n")
    print("decisions:")
    show = ["minute", "event", "team", "confidence", "role", "side", "price", "edge", "contracts"]
    print(marks[show + [c for c in marks.columns if c.startswith("markout")]].round(3).to_string(index=False))
    print("\nfills:")
    print(fills.drop(columns=["exit_ts"]).to_string(index=False, float_format="%.3f") if len(fills) else "none")
    print("\nsummary:", summarize(fills))

    out = OUTPUTS / f"{m.event_ticker}_backtest"
    out.mkdir(parents=True, exist_ok=True)
    marks.to_csv(out / "decisions.csv", index=False)
    fills.to_csv(out / "fills.csv", index=False)
    shocks = detect_shocks(probs[probs.index >= m.kickoff - pd.Timedelta(minutes=10)])
    plot_match(m, probs, shocks, signals=decisions, out=out / "chart.png")
    print(f"\noutputs -> {out}")
    if "video_s" not in raw_events and "utc" not in raw_events:
        print(
            "\nNOTE: event times came from match-report game times (minute resolution, and the "
            "second-half kickoff may be estimated). Signals can land up to a minute early, which "
            "looks like edge but is look-ahead. Use video timestamps for real conclusions."
        )


# ---- parser ---------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="varflash", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    groups = p.add_subparsers(dest="group", required=True)

    k = groups.add_parser("kalshi", help="download Kalshi market data").add_subparsers(dest="cmd", required=True)
    s = k.add_parser("series", help="list soccer series")
    s.add_argument("--category", default="Sports")
    s.add_argument("--match", default=SOCCER_PATTERN, help="regex on series title")
    s.add_argument("--games-only", action="store_true", help="only per-match moneyline series")
    s.set_defaults(fn=cmd_kalshi_series)
    s = k.add_parser("events", help="list events in a series")
    s.add_argument("series")
    s.add_argument("--status", choices=["open", "closed", "settled"])
    s.set_defaults(fn=cmd_kalshi_events)
    s = k.add_parser("pull-event", help="download one or more events")
    s.add_argument("event", nargs="+")
    s.add_argument("--no-trades", action="store_true")
    s.add_argument("--refresh", action="store_true")
    s.set_defaults(fn=cmd_kalshi_pull_event)
    s = k.add_parser("pull-series", help="download every settled event in one or more series")
    s.add_argument("series", nargs="+")
    s.add_argument("--limit", type=int)
    s.add_argument("--no-trades", action="store_true")
    s.add_argument("--refresh", action="store_true")
    s.set_defaults(fn=cmd_kalshi_pull_series)

    mk = groups.add_parser("market", help="analyse price data").add_subparsers(dest="cmd", required=True)
    for name, fn in (("analyze", cmd_market_analyze), ("scan", cmd_market_scan)):
        s = mk.add_parser(name)
        if name == "analyze":
            s.add_argument("event")
            s.add_argument("--kickoff", help="UTC kickoff override, e.g. 2026-07-12T01:00:00Z")
        else:
            s.add_argument("series")
        s.add_argument("--source", choices=["candles", "trades"], default="trades")
        s.add_argument("--window", default="2min")
        s.add_argument("--threshold", type=float, default=0.08)
        s.add_argument("--no-plot", dest="plot", action="store_false")
        s.set_defaults(fn=fn)

    v = groups.add_parser("vision", help="computer-vision tools").add_subparsers(dest="cmd", required=True)
    for name, fn in (("calibrate", cmd_vision_calibrate), ("review", cmd_vision_review)):
        s = v.add_parser(name)
        src = s.add_mutually_exclusive_group(required=True)
        src.add_argument("--image")
        src.add_argument("--video")
        s.add_argument("--time", type=float, default=0.0, help="seconds into --video")
        if name == "calibrate":
            s.add_argument("--out", required=True, help="calibration JSON path")
        else:
            s.add_argument("--calib", required=True)
            s.add_argument("--attack", choices=["left", "right"], required=True,
                           help="direction the attacking team is shooting in the image")
            s.add_argument("--attacking-color", help="hex jersey colour of the attacking team")
            s.add_argument("--receiver", help="u,v pixel near the attacker receiving the pass")
            s.add_argument("--weights", default="yolo11m.pt")
            s.add_argument("--conf", type=float, default=0.2)
            s.add_argument("--sigma", type=float, default=0.3, help="position noise (m)")
            s.add_argument("--out", default=str(OUTPUTS / "review"))
        s.set_defaults(fn=fn)
    s = v.add_parser("track", help="detect + track players and ball through a video")
    s.add_argument("--video", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--weights", default="yolo11m.pt")
    s.add_argument("--conf", type=float, default=0.2)
    s.add_argument("--stride", type=int, default=1)
    s.add_argument("--max-frames", type=int)
    s.set_defaults(fn=cmd_vision_track)

    sp = groups.add_parser("spotting", help="event-spotting model").add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("download")
    s.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    s.add_argument("--max-games", type=int, help="per split")
    s.add_argument("--workers", type=int, default=8)
    s.set_defaults(fn=cmd_spotting_download)
    s = sp.add_parser("train")
    s.add_argument("--epochs", type=int, default=15)
    s.add_argument("--batch-size", type=int, default=32)
    s.add_argument("--lr", type=float, default=1e-3)
    s.add_argument("--dim", type=int, default=128)
    s.add_argument("--max-games", type=int)
    s.add_argument("--threads", type=int)
    s.add_argument("--out", default=str(MODELS / "spotter.pt"))
    s.set_defaults(fn=cmd_spotting_train)
    s = sp.add_parser("eval")
    s.add_argument("--model", default=str(MODELS / "spotter.pt"))
    s.add_argument("--split", default="test")
    s.add_argument("--max-games", type=int)
    s.set_defaults(fn=cmd_spotting_eval)

    b = groups.add_parser("backtest", help="replay video events against Kalshi prices")
    b.add_argument("--event", required=True, help="Kalshi event ticker")
    b.add_argument("--events", required=True, help="CSV of match events (game_time|video_s|utc, event, team, confidence)")
    b.add_argument("--kickoff")
    b.add_argument("--second-half")
    b.add_argument("--video-kickoff-s", type=float, default=0.0)
    b.add_argument("--video-second-half-s", type=float)
    b.add_argument("--delay", type=float, default=0.0, help="broadcast delay of your feed (s)")
    b.add_argument("--latency", type=float, default=2.0, help="model + order latency (s)")
    b.add_argument("--min-edge", type=float, default=0.03)
    b.add_argument("--bankroll", type=float, default=1000.0)
    b.add_argument("--hold-min", type=float, default=5.0)
    b.add_argument("--to-settlement", action="store_true")
    b.add_argument("--anchor", choices=["market", "model"], default="market",
                   help="fair value = market + model jump (default) or raw model")
    b.set_defaults(fn=cmd_backtest)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.fn(args)


if __name__ == "__main__":
    main()
