"""Fill-in bumps: how a player scores when his team's lead player at his position sits.

Event = a team game the lead QB / RB / WR / TE missed (lead = most pass attempts (QB), carries +
targets (RB) or targets (WR / TE) at that position on that team that season, 4+ games). For every
other player at the position who played that game: delta = his points that game minus his average
in that season's games the lead played (2+ such games needed).

The league prior is the average delta for the next man up (QB2 / RB2 / TE2 by season usage; for WRs,
every other WR). A player's own bump = (his deltas summed + K * prior) / (his events + K), so a
backup with a long record of filling in (good or bad) moves away from the league average.
"""
from __future__ import annotations

import polars as pl

from .config import DATA

POS = ["QB", "RB", "WR", "TE"]
# shrinkage, in events. Backtest 2018-2025 (predict each fill-in game from earlier ones only), MAE vs the
# flat league prior: QB 6.34 -> 5.99, RB 6.75 -> 6.55, WR 3.98 -> 3.91, TE 4.42 -> 4.40 (no real gain).
K = {"QB": 2.0, "RB": 3.0, "WR": 5.0, "TE": 15.0}


def _usage(w: pl.DataFrame) -> pl.Expr:
    return (pl.when(pl.col("position") == "QB").then(pl.col("attempts").fill_null(0))
              .when(pl.col("position") == "RB").then(pl.col("carries").fill_null(0) + pl.col("targets").fill_null(0))
              .otherwise(pl.col("targets").fill_null(0)))


def events(league: str, seasons=range(2012, 2027), thru: tuple[int, int] | None = None) -> pl.DataFrame:
    """One row per (fill-in player, game the lead missed). thru=(season, week) cuts off later games."""
    w = pl.read_parquet(DATA / "processed" / "player_weeks.parquet").filter(
        pl.col("season").is_in(list(seasons)), pl.col("week") <= 18, pl.col("position").is_in(POS))
    if thru:
        w = w.filter((pl.col("season") < thru[0]) | ((pl.col("season") == thru[0]) & (pl.col("week") <= thru[1])))
    w = w.with_columns(_usage(w).alias("use"))
    tw = w.select("season", "team", "week").unique()
    tot = (w.group_by("season", "team", "position", "player_id")
             .agg(pl.col("use").sum(), pl.len().alias("g"), pl.col("week").alias("wks")))
    lead = (tot.filter(pl.col("g") >= 4).sort("use", descending=True)
               .group_by("season", "team", "position").first()
               .select("season", "team", "position", pl.col("player_id").alias("lead_id"), pl.col("wks").alias("lead_wks")))
    nxt = (tot.join(lead, on=["season", "team", "position"]).filter(pl.col("player_id") != pl.col("lead_id"))
              .sort("use", descending=True).group_by("season", "team", "position").first()
              .select("season", "team", "position", pl.col("player_id").alias("next_id")))
    # games the lead missed
    lw = lead.explode("lead_wks").rename({"lead_wks": "week"}).with_columns(pl.lit(True).alias("lead_played"))
    g = (w.join(lead.drop("lead_wks"), on=["season", "team", "position"])
           .filter(pl.col("player_id") != pl.col("lead_id"))
           .join(lw.select("season", "team", "position", "week", "lead_played"), on=["season", "team", "position", "week"], how="left")
           .with_columns(pl.col("lead_played").fill_null(False)))
    base = (g.filter(pl.col("lead_played")).group_by("season", "player_id")
              .agg(pl.col(f"fpts_{league}").mean().alias("usual"), pl.len().alias("n_with")))
    ev = (g.filter(~pl.col("lead_played")).join(base, on=["season", "player_id"]).filter(pl.col("n_with") >= 2)
            .join(nxt, on=["season", "team", "position"], how="left")
            .with_columns((pl.col(f"fpts_{league}") - pl.col("usual")).alias("delta"),
                          (pl.col("player_id") == pl.col("next_id")).fill_null(False).alias("is_next")))
    return ev.select("season", "week", "team", "position", "player_id", "player_display_name", "is_next",
                     pl.col(f"fpts_{league}").alias("pts"), "usual", "delta")


def priors(ev: pl.DataFrame) -> dict:
    """League prior per position: next man up (QB/RB/TE) or any other WR."""
    use = ev.filter(pl.col("is_next") | (pl.col("position") == "WR"))
    return {p: float(v) for p, v in use.group_by("position").agg(pl.col("delta").mean()).iter_rows()}


def bumps(league: str, season: int, thru: int, k: dict = K) -> tuple[pl.DataFrame, dict]:
    """Each player's fill-in bump from everything through (season, thru), shrunk to the prior."""
    ev = events(league, thru=(season, thru))
    pr = priors(ev)
    own = ev.group_by("player_id").agg(pl.col("position").last(), pl.col("delta").sum().alias("dsum"), pl.len().alias("n_ev"),
                                       pl.col("delta").mean().alias("own_avg"))
    kk = pl.col("position").replace_strict(k, return_dtype=pl.Float64)
    own = own.with_columns(((pl.col("dsum") + kk * pl.col("position").replace_strict(pr, return_dtype=pl.Float64))
                            / (pl.col("n_ev") + kk)).alias("bump"))
    return own.select("player_id", "n_ev", "own_avg", "bump"), pr
