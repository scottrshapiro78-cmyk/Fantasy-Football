"""Phase 4: expected games played next season.

Target: share of next season's regular-season weeks a player is NOT out injured, for players
who were fantasy-relevant this year (8+ PPG in The Burbs, 4+ games). "Out injured" = on
reserve (IR/PUP, weekly roster status RES) or listed Out/Doubtful on the injury report.
Being benched, cut, traded or rested in week 18 is not an injury and is not counted, which
matters most for QBs (an earlier version counted lost starting jobs as missed games).

Model: binomial GLM (logit link) on position, age, games-missed history, injury-report
history (weeks listed Out/Doubtful, soft-tissue and knee listings) and workload.
Injury cost in fantasy points is applied later against replacement level (Phase 5), so a
missed game is never charged twice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
import statsmodels.formula.api as smf

from . import data
from .config import DATA

SOFT = ["Hamstring", "Groin", "Calf", "Quadricep", "Quadriceps"]


def team_games(season: int) -> int:
    return 17 if season >= 2021 else 16


def build() -> pl.DataFrame:
    s = pl.read_parquet(DATA / "processed" / "player_seasons.parquet").filter(
        pl.col("position").is_in(["QB", "RB", "WR", "TE"]))
    inj = data.injuries().with_columns(pl.col("season").cast(pl.Int32)).filter(pl.col("game_type") == "REG")
    rep = (inj.group_by(pl.col("gsis_id").alias("player_id"), "season")
              .agg(pl.col("report_status").is_in(["Out", "Doubtful"]).sum().alias("wk_out"),
                   (pl.col("report_primary_injury").is_in(SOFT)
                    & pl.col("report_status").is_in(["Out", "Doubtful", "Questionable"])).sum().alias("wk_soft"),
                   (pl.col("report_primary_injury") == "Knee").sum().alias("wk_knee")))
    ros = data.rosters_weekly().with_columns(pl.col("season").cast(pl.Int32))
    ir = (ros.filter(pl.col("game_type") == "REG", pl.col("status") == "RES")
             .group_by(pl.col("gsis_id").alias("player_id"), "season")
             .agg(pl.col("week").n_unique().alias("wk_ir")))
    s = (s.join(rep, on=["player_id", "season"], how="left")
          .join(ir, on=["player_id", "season"], how="left")
          .with_columns(pl.col("wk_ir").fill_null(0))
          .with_columns(pl.col(["wk_out", "wk_soft", "wk_knee"]).fill_null(0),
                        (1 - ((pl.col("wk_out").fill_null(0) + pl.col("wk_ir"))
                              / pl.col("season").map_elements(team_games, return_dtype=pl.Int64)).clip(0, 1)
                         ).alias("frac"),
                        ((pl.col("c_rec") + pl.col("carries")) / pl.col("games")).alias("touches_pg")))

    # lags and target
    lag = lambda k, cols: s.select("player_id", (pl.col("season") + k).alias("season"),
                                   *[pl.col(c).alias(f"{c}_l{k}") for c in cols])
    df = (s.join(lag(1, ["frac", "wk_out", "wk_soft"]), on=["player_id", "season"], how="left")
           .join(lag(2, ["frac"]), on=["player_id", "season"], how="left"))
    nxt = s.select("player_id", (pl.col("season") - 1).alias("season"), pl.col("frac").alias("next_frac"))
    # a whole season on reserve has no stats row: take it from the roster IR weeks instead
    ir_only = ir.with_columns((1 - pl.col("wk_ir") / pl.col("season").map_elements(
        team_games, return_dtype=pl.Int64)).clip(0, 1).alias("ir_frac")).select(
        "player_id", (pl.col("season") - 1).alias("season"), "ir_frac")
    feats = pl.read_parquet(DATA / "processed" / "features.parquet").select(
        "player_id", "season", "age_next", "nfl_year", "ppg_burbs")
    df = (df.drop("ppg_burbs", strict=False).join(feats, on=["player_id", "season"], how="left")
            .join(nxt, on=["player_id", "season"], how="left")
            .join(ir_only, on=["player_id", "season"], how="left")
            .with_columns(pl.coalesce("next_frac", "ir_frac").alias("next_frac")).drop("ir_frac"))
    return df


FORMULA = ("next_frac ~ C(position) + age_next + I(age_next**2) + frac + frac_l1 + frac_l2"
           " + wk_out + wk_out_l1 + wk_soft + wk_soft_l1 + wk_knee + touches_pg + C(position):touches_pg")


def _prep(df: pl.DataFrame) -> pd.DataFrame:
    d = df.filter(pl.col("ppg_burbs") >= 8, pl.col("games") >= 4).to_pandas()
    for c in ["frac_l1", "frac_l2"]:
        d[c] = d[c].fillna(d["frac"])  # no prior season: assume like this one
    for c in ["wk_out_l1", "wk_soft_l1"]:
        d[c] = d[c].fillna(0)
    d["age_next"] = d["age_next"].fillna(d.groupby("position")["age_next"].transform("median"))
    d["touches_pg"] = d["touches_pg"].fillna(0)
    return d


def fit(df: pl.DataFrame):
    d = _prep(df).dropna(subset=["next_frac"])
    return smf.glm(FORMULA, d, family=sm.families.Binomial()).fit()


def predict(model, df: pl.DataFrame, season_games: int = 17) -> pd.DataFrame:
    d = _prep(df)
    d["exp_frac"] = model.predict(d)
    d["exp_games"] = d["exp_frac"] * season_games
    d["p_miss4"] = 1 - _p_at_least(d["exp_frac"], season_games, season_games - 3)
    d["risk_score"] = (100 * d["p_miss4"]).round(0)
    return d


def _p_at_least(p, n, k):
    """P(games >= k) under a beta-binomial with modest overdispersion (injuries cluster)."""
    from scipy.stats import betabinom
    rho = 4.0  # concentration; lower = more all-or-nothing seasons
    a, b = np.asarray(p) * rho, (1 - np.asarray(p)) * rho
    return 1 - betabinom.cdf(k - 1, n, a, b)
