"""Player-season predictor table: everything known after season t, target = PPG in season t+1."""
from __future__ import annotations

import polars as pl

from . import data, ids
from .config import DATA
from .tough import tag_tough, splits

LEAGUES = ["burbs", "cbs"]


def _opportunity() -> pl.DataFrame:
    o = data.opportunity().with_columns(pl.col("season").cast(pl.Int32),
                                        pl.col("week").cast(pl.Int32, strict=False))
    o = o.filter(pl.col("week") <= 18)
    num = lambda c: pl.col(c).cast(pl.Float64, strict=False).fill_null(0)
    return (o.group_by("player_id", "season")
             .agg(num("total_fantasy_points_exp").mean().alias("xfp_pg"),
                  num("total_fantasy_points_diff").mean().alias("fpoe_pg"),
                  num("total_touchdown_diff").mean().alias("td_luck_pg"),
                  num("total_yards_gained_diff").mean().alias("yds_oe_pg"),
                  (num("rec_air_yards")).mean().alias("air_yds_pg")))


def _player_season_stats(raw: pl.DataFrame) -> pl.DataFrame:
    r = raw.filter(pl.col("season_type") == "REG")
    f = lambda c: pl.col(c).cast(pl.Float64, strict=False).fill_null(0)
    return (r.group_by("player_id", "season")
             .agg(f("targets").mean().alias("tgt_pg"), f("carries").mean().alias("car_pg"),
                  f("attempts").mean().alias("att_pg"), f("receptions").mean().alias("rec_pg"),
                  f("target_share").mean().alias("tgt_share"), f("air_yards_share").mean().alias("ay_share"),
                  f("wopr").mean().alias("wopr"),
                  (f("receiving_yards").sum() / f("targets").sum().clip(1)).alias("yds_per_tgt"),
                  (f("rushing_yards").sum() / f("carries").sum().clip(1)).alias("yds_per_car"),
                  (f("receptions").sum() / f("targets").sum().clip(1)).alias("catch_rate"),
                  (f("passing_epa").sum() / f("attempts").sum().clip(1)).alias("pass_epa_per_att"),
                  f("passing_cpoe").mean().alias("cpoe"),
                  f("rushing_yards").mean().alias("rush_yds_pg"),
                  f("receiving_epa").mean().alias("rec_epa_pg"),
                  f("rushing_epa").mean().alias("rush_epa_pg"),
                  (f("punt_return_yards") + f("kickoff_return_yards")).mean().alias("ret_yds_pg")))


def _team_context(weeks: pl.DataFrame, raw: pl.DataFrame) -> pl.DataFrame:
    """Team offense quality and the team's QB quality, per team-season."""
    team = (weeks.group_by("team", "season", "game_id").agg(pl.col("fpts_burbs").sum().alias("tf"))
                 .group_by("team", "season").agg(pl.col("tf").mean().alias("team_fp_pg")))
    r = raw.filter(pl.col("season_type") == "REG")
    qb = (r.filter(pl.col("position") == "QB")
           .group_by("team", "season")
           .agg((pl.col("passing_epa").sum() / pl.col("attempts").sum().clip(1)).alias("team_qb_epa"),
                (pl.col("attempts").sum() / pl.col("game_id").n_unique()).alias("team_pass_att_pg")))
    return team.join(qb, on=["team", "season"], how="left")


def _opening_teams() -> pl.DataFrame:
    """Team each player is on at the start of each season (first roster week), keyed to the
    season before, i.e. the team known on draft night for the season being projected."""
    r = data.rosters_weekly().filter(~pl.col("status").is_in(["CUT", "RET"]), pl.col("gsis_id").is_not_null())
    first = (r.sort("week").group_by("gsis_id", "season")
              .agg(pl.col("team").first().alias("opening_team_next")))
    return first.select(pl.col("gsis_id").alias("player_id"), (pl.col("season") - 1).alias("season"),
                        "opening_team_next")


def _role(seasons: pl.DataFrame, df: pl.DataFrame) -> pl.DataFrame:
    """Opportunity open to each player on his team for next season.

    Opportunity = carries + targets (RB), targets (WR/TE), pass attempts (QB).
    open_share = 1 - (last season's opportunity of everyone else at his position who is on his
    draft-night team) / (that team's total opportunity at the position last season).
    role_gain  = open_share - his own share last season: work vacated by departures, or a bigger
    role on a new team (e.g. a lead back leaving).
    Incoming rookies are not counted as competition (no NFL history yet).
    """
    opp = seasons.select("player_id", "season", "team", "position",
                         pl.when(pl.col("position") == "RB").then(pl.col("carries") + pl.col("targets"))
                           .when(pl.col("position") == "QB").then(pl.col("attempts"))
                           .otherwise(pl.col("targets")).fill_null(0).cast(pl.Float64).alias("opp"))
    tot = opp.group_by("team", "season", "position").agg(pl.col("opp").sum().alias("team_opp"))
    d = (df.select("player_id", "season", "position", "team", "team_next")
           .join(opp.select("player_id", "season", "opp"), on=["player_id", "season"], how="left"))
    incoming = d.group_by("team_next", "season", "position").agg(pl.col("opp").sum().alias("incoming"))
    d = (d.join(incoming, on=["team_next", "season", "position"], how="left")
          .join(tot.rename({"team": "team_next", "team_opp": "next_team_opp"}),
                on=["team_next", "season", "position"], how="left")
          .join(tot, on=["team", "season", "position"], how="left"))
    return d.select(
        "player_id", "season",
        (1 - (pl.col("incoming") - pl.col("opp")) / pl.col("next_team_opp")).clip(0, 1).alias("open_share"),
        (pl.col("opp") / pl.col("team_opp")).clip(0, 1).alias("own_share"),
    ).with_columns((pl.col("open_share") - pl.col("own_share")).alias("role_gain")).join(
        df.select("player_id", "season", "position"), on=["player_id", "season"]).with_columns(
        # QBs: one starter per team, so share-of-attempts says little (and hurt the backtest)
        [pl.when(pl.col("position") == "QB").then(0.0).otherwise(pl.col(c)).alias(c)
         for c in ("open_share", "own_share", "role_gain")]).drop("position")


def build() -> pl.DataFrame:
    raw = data.player_weeks()
    weeks = pl.read_parquet(DATA / "processed" / "player_weeks.parquet")
    seasons = pl.read_parquet(DATA / "processed" / "player_seasons.parquet")
    weeks = weeks.filter(pl.col("position").is_in(["QB", "RB", "WR", "TE"]))
    seasons = seasons.filter(pl.col("position").is_in(["QB", "RB", "WR", "TE"]))

    tough = splits(tag_tough(weeks, "burbs"), "burbs").select(
        "player_id", "season", "ppg_tough", "ppg_easy", "tough_idx", "g_tough")
    df = (seasons.select("player_id", "player_display_name", "position", "season", "team", "games",
                         *[f"ppg_{l}" for l in LEAGUES], "birthdate", "draft_year", "draft_round",
                         "draft_pick")
                 .join(_player_season_stats(raw), on=["player_id", "season"], how="left")
                 .join(_opportunity(), on=["player_id", "season"], how="left")
                 .join(tough, on=["player_id", "season"], how="left")
                 .join(_team_context(weeks, raw), on=["team", "season"], how="left"))

    first = seasons.group_by("player_id").agg(pl.col("season").min().alias("first_season"))
    df = df.join(first, on="player_id").with_columns(
        (pl.col("season") - pl.col("first_season") + 1).alias("nfl_year"),
        ((pl.date(pl.col("season") + 1, 9, 1)
          - pl.col("birthdate").str.to_date(strict=False)).dt.total_days() / 365.25).alias("age_next"),
        pl.col("draft_round").cast(pl.Float64, strict=False).fill_null(8).alias("draft_round"),
        pl.col("draft_pick").cast(pl.Float64, strict=False).fill_null(260).alias("draft_pick"),
        (pl.col("ppg_burbs") * pl.col("games") / pl.col("team_fp_pg") / 17).alias("share_team_fp"),
    )

    prev = df.select("player_id", (pl.col("season") + 1).alias("season"),
                     pl.col("ppg_burbs").alias("ppg_burbs_lag1"), pl.col("ppg_cbs").alias("ppg_cbs_lag1"),
                     pl.col("games").alias("games_lag1"))
    nxt = seasons.select("player_id", (pl.col("season") - 1).alias("season"),
                         pl.col("ppg_burbs").alias("next_ppg_burbs"), pl.col("ppg_cbs").alias("next_ppg_cbs"),
                         pl.col("games").alias("next_games"), pl.col("team").alias("next_team"))
    df = (df.join(prev, on=["player_id", "season"], how="left")
            .join(nxt, on=["player_id", "season"], how="left")
            .join(_opening_teams(), on=["player_id", "season"], how="left")
            .with_columns(pl.coalesce("opening_team_next", "next_team", "team").alias("team_next"))
            .with_columns((pl.col("team_next") != pl.col("team")).cast(pl.Int8).alias("changed_team")))
    df = df.join(_role(seasons, df), on=["player_id", "season"], how="left")

    # 3-season recency-weighted PPG through season t (3/2/1 by season, weighted by games)
    s = seasons.select("player_id", "season", "games", *[f"fpts_{l}" for l in LEAGUES])
    b = df.select("player_id", "season")
    for lag in (0, 1, 2):
        b = b.join(s.with_columns(pl.col("season") + lag).rename(
            {"games": f"g{lag}", **{f"fpts_{l}": f"{l}{lag}" for l in LEAGUES}}),
            on=["player_id", "season"], how="left")
    z = lambda c: pl.col(c).fill_null(0)
    den = 3 * z("g0") + 2 * z("g1") + z("g2")
    b = b.select("player_id", "season", *[
        ((3 * z(f"{l}0") + 2 * z(f"{l}1") + z(f"{l}2")) / den).alias(f"base_{l}") for l in LEAGUES])
    return df.join(b, on=["player_id", "season"], how="left")


# ppg / ppg_lag1 are swapped for the league being modelled (see projection.league_features)
FEATURES = ["ppg_burbs", "ppg_burbs_lag1", "games", "games_lag1", "age_next", "nfl_year",
            "draft_round", "draft_pick", "tgt_pg", "car_pg", "att_pg", "rec_pg", "tgt_share",
            "ay_share", "wopr", "yds_per_tgt", "yds_per_car", "catch_rate", "pass_epa_per_att",
            "cpoe", "rush_yds_pg", "rec_epa_pg", "rush_epa_pg", "ret_yds_pg", "xfp_pg", "fpoe_pg",
            "td_luck_pg", "yds_oe_pg", "air_yds_pg", "ppg_tough", "ppg_easy", "tough_idx",
            "team_fp_pg", "team_qb_epa", "team_pass_att_pg", "share_team_fp", "changed_team",
            "open_share", "role_gain"]
