"""Team defense / special teams: weekly fantasy scoring for each league, and the predictors
used to rank defenses for next season (prior-year averages, head-coach change, returning
defensive snap share).

Points allowed = the opponent's final score (assumption; check one Yahoo DST total).
Defensive coordinator changes are not in nflverse; they need a hand-entered list for 2027.
"""
from __future__ import annotations

import polars as pl

from . import data
from .config import load_leagues


def _pa_points(pa: pl.Expr, tiers: dict) -> pl.Expr:
    expr = None
    for upper, pts in sorted(((int(k), v) for k, v in tiers.items()), reverse=True):
        expr = pl.when(pa <= upper).then(pts) if expr is None else pl.when(pa <= upper).then(pts).otherwise(expr)
    return expr


def dst_weeks() -> pl.DataFrame:
    t = data.team_weeks().filter(pl.col("season_type") == "REG")
    sch = data.schedules().filter(pl.col("game_type") == "REG")
    pa = pl.concat([
        sch.select("season", "week", pl.col("home_team").alias("team"), pl.col("away_score").alias("pa"),
                   pl.col("home_coach").alias("coach")),
        sch.select("season", "week", pl.col("away_team").alias("team"), pl.col("home_score").alias("pa"),
                   pl.col("away_coach").alias("coach"))])
    z = lambda c: pl.col(c).cast(pl.Float64, strict=False).fill_null(0)
    d = (t.with_columns(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32))
          .join(pa.with_columns(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32)),
                on=["season", "week", "team"], how="inner")
          .with_columns(z("def_sacks").alias("sacks"), z("def_interceptions").alias("ints"),
                        z("fumble_recovery_opp").alias("fum_rec"),
                        (z("def_tds") + z("special_teams_tds")).alias("tds"),
                        z("def_safeties").alias("safeties"),
                        (z("def_punt_blocks") + z("def_fg_blocks") + z("def_pat_blocks")).alias("blocks")))
    for key in ["burbs", "cbs"]:
        L = load_leagues()[key]
        sc = L["scoring_dst"] if "scoring_dst" in L else {"sacks": 1, "ints": 2, "fum_rec": 2, "tds": 6,
                                                          "safeties": 2, "blocks": 2}
        d = d.with_columns((sum(pl.col(c) * v for c, v in sc.items())
                            + _pa_points(pl.col("pa"), L["dst_points_allowed"])).alias(f"fpts_{key}"))
    return d


def team_seasons() -> pl.DataFrame:
    w = dst_weeks()
    s = (w.group_by("team", "season")
          .agg(pl.len().alias("games"), pl.col("coach").mode().first().alias("coach"),
               *[pl.col(f"fpts_{k}").mean().alias(f"ppg_{k}") for k in ["burbs", "cbs"]],
               pl.col("sacks").mean().alias("sacks_pg"), (pl.col("ints") + pl.col("fum_rec")).mean().alias("takeaways_pg"),
               pl.col("pa").mean().alias("pa_pg"), pl.col("tds").mean().alias("tds_pg")))
    # returning defensive snap share: share of this season's defensive snaps played by players
    # who are on the same team next season
    sn = data.snap_counts().filter(pl.col("game_type") == "REG").with_columns(pl.col("season").cast(pl.Int32))
    ps = (sn.group_by("pfr_player_id", "team", "season").agg(pl.col("defense_snaps").sum().alias("snaps")))
    nxt_team = (sn.group_by("pfr_player_id", "season").agg(pl.col("team").mode().first().alias("nteam"))
                  .with_columns((pl.col("season") - 1).alias("season")))
    ret = (ps.join(nxt_team, on=["pfr_player_id", "season"], how="left")
             .group_by("team", "season")
             .agg(((pl.col("snaps") * (pl.col("nteam") == pl.col("team")).fill_null(False)).sum()
                   / pl.col("snaps").sum()).alias("returning_snaps")))
    s = s.join(ret, on=["team", "season"], how="left")
    nxt = s.select("team", (pl.col("season") - 1).alias("season"),
                   *[pl.col(f"ppg_{k}").alias(f"next_ppg_{k}") for k in ["burbs", "cbs"]],
                   pl.col("coach").alias("next_coach"))
    lag = s.select("team", (pl.col("season") + 1).alias("season"),
                   *[pl.col(f"ppg_{k}").alias(f"ppg_{k}_lag1") for k in ["burbs", "cbs"]])
    return (s.join(nxt, on=["team", "season"], how="left").join(lag, on=["team", "season"], how="left")
             .with_columns((pl.col("next_coach") != pl.col("coach")).cast(pl.Int8).alias("hc_change")))


DST_FEATURES = ["ppg_{lg}", "ppg_{lg}_lag1", "sacks_pg", "takeaways_pg", "pa_pg", "tds_pg",
                "returning_snaps", "hc_change"]


def project(seasons: pl.DataFrame, target: int, league: str, hc_change: dict | None = None) -> pl.DataFrame:
    """Ridge on prior-season team defense stats. hc_change: {team: 1} for new head coaches
    (known after January firings); defaults to 0 for everyone."""
    import numpy as np
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = [c.format(lg=league) for c in DST_FEATURES]
    d = seasons.to_pandas()
    d[f"ppg_{league}_lag1"] = d[f"ppg_{league}_lag1"].fillna(d[f"ppg_{league}"])
    d["returning_snaps"] = d["returning_snaps"].fillna(d["returning_snaps"].median())
    tr = d[(d.season + 1 < target) & d[f"next_ppg_{league}"].notna() & (d.season >= 2013)]
    te = d[d.season == target - 1].copy()
    te["hc_change"] = te["team"].map(hc_change or {}).fillna(0)
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))).fit(tr[X], tr[f"next_ppg_{league}"])
    te[f"pred_ppg_{league}"] = m.predict(te[X])
    return pl.from_pandas(te[["team", f"pred_ppg_{league}", "returning_snaps", "hc_change"]]).sort(
        f"pred_ppg_{league}", descending=True)
