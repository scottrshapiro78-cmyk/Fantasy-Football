"""nflverse loaders with a local parquet cache, so each season downloads once."""
from __future__ import annotations

import polars as pl
import nflreadpy as nfl

from .config import DATA

FIRST_SEASON = 2012  # 10+ seasons to fit aging, stability and injury priors
CURRENT_SEASON = 2026
PLAYER_IDS_RAW = "https://raw.githubusercontent.com/dynastyprocess/data/master/files/db_playerids.csv"


def _cached(name: str, seasons: list[int], loader, refresh_current: bool = False) -> pl.DataFrame:
    out = []
    for s in seasons:
        path = DATA / "raw" / name / f"{s}.parquet"
        if path.exists() and not (refresh_current and s == CURRENT_SEASON):
            out.append(pl.read_parquet(path))
            continue
        df = loader([s])
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)
        out.append(df)
    return pl.concat(out, how="diagonal_relaxed")


def seasons(first: int = FIRST_SEASON, last: int = CURRENT_SEASON) -> list[int]:
    return list(range(first, last + 1))


def player_weeks(ss: list[int] | None = None, refresh_current: bool = False) -> pl.DataFrame:
    """One row per player per game (regular season and playoffs)."""
    return _cached("player_stats", ss or seasons(), nfl.load_player_stats, refresh_current)


def schedules(ss: list[int] | None = None) -> pl.DataFrame:
    return nfl.load_schedules(ss or seasons())


def injuries(ss: list[int] | None = None, refresh_current: bool = False) -> pl.DataFrame:
    return _cached("injuries", ss or seasons(), nfl.load_injuries, refresh_current)


def player_ids() -> pl.DataFrame:
    """ffverse ID crosswalk: gsis_id (nflverse) <-> yahoo_id, sleeper_id, pfr_id, etc."""
    path = DATA / "raw" / "ff_playerids.parquet"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            df = nfl.load_ff_playerids()
        except Exception:
            # github.com/<repo>/raw/ can be blocked by proxies; raw.githubusercontent.com is the same file
            df = pl.read_csv(PLAYER_IDS_RAW, infer_schema_length=0)
        df.write_parquet(path)
    return pl.read_parquet(path)


def opportunity(ss: list[int] | None = None, refresh_current: bool = False) -> pl.DataFrame:
    """ffverse expected-points model: expected vs actual yards, TDs and fantasy points per player-week."""
    return _cached("ff_opportunity", ss or seasons(), lambda s: nfl.load_ff_opportunity(seasons=s),
                   refresh_current)


def rosters_weekly(ss: list[int] | None = None, refresh_current: bool = False) -> pl.DataFrame:
    """Weekly roster status (ACT, RES = injured reserve/PUP, INA, CUT, ...) per player."""
    return _cached("rosters_weekly", ss or seasons(), nfl.load_rosters_weekly, refresh_current)
