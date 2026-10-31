"""End-of-season opponent adjustment.

For each season and position, every player-game is modelled as

    fpts = player level + defense-vs-position effect + noise

and the two are fitted together by alternating shrunken means (a ridge regression solved by
backfitting). A defense's effect is how far players beat or missed *their own* averages
against it, so a defense that faced elite offenses is judged against those offenses, not the
league average. Shrinkage (k) is in player-games and is tuned by the backtest.
"""
from __future__ import annotations

import polars as pl

POSITIONS = ["QB", "RB", "WR", "TE"]


def defense_effects(weeks: pl.DataFrame, col: str, k: float = 30.0, iters: int = 6) -> pl.DataFrame:
    """Return [season, position, opponent_team, def_<col>] = points allowed above expectation per player-game."""
    w = weeks.filter(pl.col("position").is_in(POSITIONS)).select(
        "season", "position", "player_id", "opponent_team", pl.col(col).alias("y"))
    w = w.with_columns(pl.lit(0.0).alias("d"))
    for _ in range(iters):
        w = w.with_columns((pl.col("y") - pl.col("d")).mean()
                           .over("season", "player_id").alias("p"))
        eff = (w.group_by("season", "position", "opponent_team")
                .agg(((pl.col("y") - pl.col("p")).sum() / (pl.len() + k)).alias("d_new")))
        w = w.drop("d").join(eff, on=["season", "position", "opponent_team"], how="left") \
             .rename({"d_new": "d"})
    return eff.rename({"d_new": f"def_{col}"})


def adjust(weeks: pl.DataFrame, leagues: list[str], k: float = 30.0) -> pl.DataFrame:
    """Add adj_fpts_<league> = fpts minus the opponent's effect (positive effect = soft defense)."""
    out = weeks
    for lg in leagues:
        col = f"fpts_{lg}"
        eff = defense_effects(weeks, col, k)
        out = (out.join(eff, on=["season", "position", "opponent_team"], how="left")
                  .with_columns((pl.col(col) - pl.col(f"def_{col}").fill_null(0)).alias(f"adj_{col}"))
                  .drop(f"def_{col}"))
    return out


def adjusted_seasons(weeks: pl.DataFrame, seasons: pl.DataFrame, leagues: list[str]) -> pl.DataFrame:
    agg = (weeks.group_by("player_id", "season")
                .agg([pl.col(f"adj_fpts_{lg}").sum() for lg in leagues]))
    return seasons.join(agg, on=["player_id", "season"], how="left")
