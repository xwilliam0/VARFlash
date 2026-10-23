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

    def game_time_to_utc(self, half: int, seconds_into_half: float) -> pd.Timestamp:
        """SoccerNet-style game time ("2 - 13:20" -> half=2, 800 s)."""
        start = self.kickoff_utc if half == 1 else self.second_half_utc
        return start + pd.Timedelta(seconds=seconds_into_half)

    def match_minute(self, ts: pd.Timestamp) -> float:
        """Minutes of regulation elapsed, with first-half stoppage clamped to 45'."""
        if ts >= self.second_half_utc:
            return HALF_MINUTES + (ts - self.second_half_utc).total_seconds() / 60
        return min((ts - self.kickoff_utc).total_seconds() / 60, HALF_MINUTES)


def parse_game_time(s: str) -> tuple[int, float]:
    half, clock = (x.strip() for x in s.split("-"))
    m, sec = clock.split(":")
    return int(half), int(m) * 60 + float(sec)
