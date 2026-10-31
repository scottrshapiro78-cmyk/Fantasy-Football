"""Performance against tough defenses.

Each season, defenses are ranked against each position by their end-of-season
opponent-adjusted effect (opponent.defense_effects). The hardest N (default 8 of 32) are
"tough". For every player-season we measure:

  ppg_tough   points per game vs tough defenses
  ppg_easy    points per game vs everyone else
  tough_idx   ppg_tough / ppg_easy   (1.00 = no drop-off against good defenses)
"""
from __future__ import annotations

import polars as pl

from .opponent import defense_effects


def tag_tough(weeks: pl.DataFrame, league: str, n_tough: int = 8, k: float = 30.0) -> pl.DataFrame:
    col = f"fpts_{league}"
    eff = defense_effects(weeks, col, k).with_columns(
        pl.col(f"def_{col}").rank("ordinal").over("season", "position").alias("def_rank"))
    return (weeks.join(eff, on=["season", "position", "opponent_team"], how="left")
                 .with_columns((pl.col("def_rank") <= n_tough).fill_null(False).alias("tough")))


def splits(tagged: pl.DataFrame, league: str) -> pl.DataFrame:
    col = f"fpts_{league}"
    return (tagged.group_by("player_id", "player_display_name", "position", "season")
                  .agg(pl.len().alias("games"),
                       pl.col("tough").sum().alias("g_tough"),
                       pl.col(col).filter(pl.col("tough")).mean().alias("ppg_tough"),
                       pl.col(col).filter(~pl.col("tough")).mean().alias("ppg_easy"),
                       pl.col(col).mean().alias("ppg"))
                  .with_columns((pl.col("ppg_tough") / pl.col("ppg_easy")).alias("tough_idx")))
