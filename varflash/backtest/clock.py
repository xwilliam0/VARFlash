"""Map between video time, match time and wall-clock (UTC) time.

A recording is anchored by noting the video timestamp of each half's kickoff
and the real UTC kickoff times. Everything the vision models emit is in video
seconds; markets are in UTC; the in-play model wants match minutes.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

HALF_MINUTES = 45.0
DEFAULT_HALFTIME_GAP = pd.Timedelta(minutes=45 + 3 + 15)  # 45' + stoppage + break


@dataclass
class MatchClock:
    kickoff_utc: pd.Timestamp
    second_half_utc: pd.Timestamp | None = None
    video_kickoff_s: float = 0.0
    video_second_half_s: float | None = None

    def __post_init__(self):
        self.kickoff_utc = pd.Timestamp(self.kickoff_utc)
        if self.second_half_utc is None:
            self.second_half_utc = self.kickoff_utc + DEFAULT_HALFTIME_GAP
        self.second_half_utc = pd.Timestamp(self.second_half_utc)

    def video_to_utc(self, t: float) -> pd.Timestamp:
        if self.video_second_half_s is not None and t >= self.video_second_half_s:
            return self.second_half_utc + pd.Timedelta(seconds=t - self.video_second_half_s)
        return self.kickoff_utc + pd.Timedelta(seconds=t - self.video_kickoff_s)

    def utc_to_video(self, ts: pd.Timestamp) -> float | None:
        """Inverse of ``video_to_utc``; None for wall-clock moments the video cut out (half-time)."""
        ts = pd.Timestamp(ts)
        if self.video_second_half_s is not None and ts >= self.second_half_utc:
            return self.video_second_half_s + (ts - self.second_half_utc).total_seconds()
        t = self.video_kickoff_s + (ts - self.kickoff_utc).total_seconds()
        if self.video_second_half_s is not None and t >= self.video_second_half_s:
            return None
        return t

    def game_time_to_utc(self, half: int, seconds_into_half: float) -> pd.Timestamp:
        """SoccerNet-style game time ("2 - 13:20" -> half=2, 800 s)."""
        start = self.kickoff_utc if half == 1 else self.second_half_utc
        return start + pd.Timedelta(seconds=seconds_into_half)

    def match_minute(self, ts: pd.Timestamp) -> float:
        """Minutes of regulation elapsed, with first-half stoppage clamped to 45'."""
        if ts >= self.second_half_utc:
            return HALF_MINUTES + (ts - self.second_half_utc).total_seconds() / 60
        return min((ts - self.kickoff_utc).total_seconds() / 60, HALF_MINUTES)


class SegmentClock(MatchClock):
    """Clock for a video stitched from separate real-time segments (e.g. highlight clips).

    Each segment is a dict with ``start_s``/``end_s`` (video), and one anchor moment
    ``anchor_video_s`` whose wall-clock time ``anchor_utc`` and match ``minute`` are known.
    Time runs at 1x inside a segment; between segments it jumps.
    """

    def __init__(self, base: MatchClock, segments: list[dict]):
        super().__init__(base.kickoff_utc, base.second_half_utc, base.video_kickoff_s, base.video_second_half_s)
        self.segments = [{**s, "anchor_utc": pd.Timestamp(s["anchor_utc"])} for s in segments]

    def _video_offset(self, s: dict, ts: pd.Timestamp) -> float:
        return s["anchor_video_s"] + (pd.Timestamp(ts) - s["anchor_utc"]).total_seconds()

    def video_to_utc(self, t: float) -> pd.Timestamp | None:
        for s in self.segments:
            if s["start_s"] <= t <= s["end_s"]:
                # ms precision: float noise would otherwise give ns timestamps that pandas
                # refuses to compare against the market's coarser-unit indexes
                return s["anchor_utc"] + pd.Timedelta(milliseconds=round((t - s["anchor_video_s"]) * 1000))
        return None

    def utc_to_video(self, ts: pd.Timestamp) -> float | None:
        for s in self.segments:
            t = self._video_offset(s, ts)
            if s["start_s"] <= t <= s["end_s"]:
                return t
        return None

    def match_minute(self, ts: pd.Timestamp) -> float:
        for s in self.segments:
            t = self._video_offset(s, ts)
            if s["start_s"] <= t <= s["end_s"]:
                return s["minute"] + (t - s["anchor_video_s"]) / 60
        return super().match_minute(ts)


def parse_game_time(s: str) -> tuple[int, float]:
    half, clock = (x.strip() for x in s.split("-"))
    m, sec = clock.split(":")
    return int(half), int(m) * 60 + float(sec)
