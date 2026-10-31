"""Rookie projections from draft capital: per position, PPG (rookies with 4+ games) and share
of games played (all drafted rookies) regressed on log(draft pick), 2012 onward."""
from __future__ import annotations

import numpy as np
import polars as pl

from . import data
from .config import DATA

POS = ["QB", "RB", "WR", "TE"]


def _picks() -> pl.DataFrame:
    """nflverse draft picks: overall pick number per player (the ID crosswalk's pick column is
    not reliable for this)."""
    return (data.draft_picks()
                .select(pl.col("gsis_id"), pl.col("pfr_player_name").alias("name"), "position",
                        "team", pl.col("season").cast(pl.Int32).alias("draft_year"),
                        pl.col("pick").cast(pl.Float64).alias("draft_pick"))
                .filter(pl.col("gsis_id").is_not_null()))


def _rookie_history(before: int) -> pl.DataFrame:
    xw = _picks().filter(pl.col("position").is_in(POS), pl.col("draft_year").is_between(2012, before - 1))
    s = pl.read_parquet(DATA / "processed" / "player_seasons.parquet")
    r = xw.join(s.select("player_id", "season", "games", "ppg_burbs", "ppg_cbs"),
                left_on=["gsis_id", "draft_year"], right_on=["player_id", "season"], how="left")
    return r.with_columns(pl.col("games").fill_null(0))


BUCKETS = [0, 10, 32, 64, 100, 160]


def _bucket(pick: pl.Expr) -> pl.Expr:
    return pick.cut(BUCKETS[1:-1], labels=[str(i) for i in range(len(BUCKETS) - 1)], left_closed=False)


def project(target: int, leagues=("burbs", "cbs")) -> pl.DataFrame:
    """Average rookie season of past rookies at the same position and draft range
    (picks 1-10, 11-32, 33-64, 65-100, 101-160). A log-linear fit over-projected top picks
    (Jeremiyah Love 20.8 PPG for 2026; he is at 10.2 after 8 weeks)."""
    hist = _rookie_history(target).filter(pl.col("draft_pick") <= 160).with_columns(
        _bucket(pl.col("draft_pick")).alias("b"))
    new = _picks().filter(pl.col("draft_year") == target, pl.col("position").is_in(POS),
                          pl.col("draft_pick") <= 160).with_columns(_bucket(pl.col("draft_pick")).alias("b"))
    if new.height == 0:
        return pl.DataFrame()
    played = hist.filter(pl.col("games") >= 4)
    rates = (played.group_by("position", "b").agg(*[pl.col(f"ppg_{lg}").mean().alias(f"pred_ppg_{lg}") for lg in leagues])
                   .join(hist.group_by("position", "b").agg((pl.col("games") / 17).clip(0, 1).mean().alias("exp_frac")),
                         on=["position", "b"]))
    return (new.join(rates, on=["position", "b"], how="left")
               .select(pl.col("gsis_id").alias("player_id"), pl.col("name").alias("player_display_name"),
                       "position", "team", "exp_frac", *[f"pred_ppg_{lg}" for lg in leagues])
               .with_columns(pl.lit(True).alias("rookie")))
