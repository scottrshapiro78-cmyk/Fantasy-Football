"""Rest-of-season projections.

ROS PPG = (k x preseason projection + n x this season's PPG so far) / (k + n), where n = games
played so far and k (a prior worth k games) is fitted per position on past seasons. For skill
players "PPG so far" is luck-adjusted: u x (expected - actual fantasy points per game, from the
ffverse usage model) is added back, with u also fitted.

Expected games left = team games left x the chance he plays each one, from his roster status now
(active / on IR, PUP or suspended / ruled Out this week), measured on past seasons.
"""
from __future__ import annotations

import itertools

import numpy as np
import polars as pl

from . import data
from .board import projections
from .config import DATA
from .dst import dst_weeks

LEAGUES = ("burbs", "cbs")
POS = ["QB", "RB", "WR", "TE", "K", "DEF"]
K_GRID = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30]
U_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def _weekly(season: int) -> pl.DataFrame:
    """Player-week fantasy points (both leagues) incl. kickers and defenses, plus usage luck."""
    w = pl.read_parquet(DATA / "processed" / "player_weeks.parquet").filter(pl.col("season") == season)
    o = data.opportunity([season]).filter(pl.col("player_id").is_not_null()).select(
        "player_id", pl.col("week").cast(pl.Int32),
        (pl.col("total_fantasy_points_exp").cast(pl.Float64) - pl.col("total_fantasy_points").cast(pl.Float64))
        .alias("luck_gap"))
    w = (w.select("player_id", "player_display_name", "position", "team", pl.col("week").cast(pl.Int32),
                  "fpts_burbs", "fpts_cbs")
          .join(o.group_by("player_id", "week").agg(pl.col("luck_gap").sum()), on=["player_id", "week"], how="left"))
    d = (dst_weeks().filter(pl.col("season") == season)
         .select(pl.col("team").alias("player_id"), (pl.col("team") + " DEF").alias("player_display_name"),
                 pl.lit("DEF").alias("position"), "team", pl.col("week").cast(pl.Int32), "fpts_burbs", "fpts_cbs",
                 pl.lit(None, dtype=pl.Float64).alias("luck_gap")))
    return pl.concat([w, d], how="vertical_relaxed")


def to_date(season: int, thru: int) -> pl.DataFrame:
    w = _weekly(season).filter(pl.col("week") <= thru)
    return w.group_by("player_id").agg(
        pl.len().alias("n"), pl.col("team").sort_by("week").last().alias("team_now"),
        *[pl.col(f"fpts_{lg}").mean().alias(f"ppg_{lg}_sofar") for lg in LEAGUES],
        pl.col("luck_gap").mean().fill_null(0).alias("luck_pg"))


def after(season: int, start: int, end: int = 17) -> pl.DataFrame:
    w = _weekly(season).filter(pl.col("week").is_between(start, end))
    return w.group_by("player_id").agg(pl.len().alias("n_after"),
                                       *[pl.col(f"fpts_{lg}").mean().alias(f"ppg_{lg}_after") for lg in LEAGUES])


def status(season: int, week: int) -> pl.DataFrame:
    """Roster status as of `week`: out_long (IR / PUP / suspended), out_week (ruled Out or
    Doubtful on that week's injury report), active, or gone (cut / retired)."""
    r = data.rosters_weekly([season]).filter(pl.col("week") <= week, pl.col("gsis_id").is_not_null())
    r = r.sort("week").group_by("gsis_id").agg(pl.all().last())
    inj = (data.injuries([season]).filter(pl.col("week") == week, pl.col("gsis_id").is_not_null())
           .group_by("gsis_id").agg(pl.col("report_status").last()))
    r = r.join(inj, on="gsis_id", how="left")
    return r.select(
        pl.col("gsis_id").alias("player_id"), pl.col("team").alias("roster_team"),
        pl.when(pl.col("status").is_in(["CUT", "RET"])).then(pl.lit("gone"))
          .when(pl.col("status").is_in(["RES", "EXE", "SUS"])).then(pl.lit("out_long"))
          .when(pl.col("report_status").is_in(["Out", "Doubtful"])).then(pl.lit("out_week"))
          .otherwise(pl.lit("active")).alias("status"),
        pl.col("status_description_abbr").alias("status_code"))


def team_games(season: int, start: int, end: int) -> dict:
    s = data.schedules([season]).filter(pl.col("game_type") == "REG", pl.col("week").is_between(start, end))
    t = pl.concat([s.select(pl.col("home_team").alias("t")), s.select(pl.col("away_team").alias("t"))])
    return dict(t.group_by("t").len().iter_rows())


def _history(seasons=range(2021, 2026), thru: int = 8) -> pl.DataFrame:
    rows = []
    for s in seasons:
        pri = projections(s, "burbs").select("player_id", "position", "pred_ppg_burbs").join(
            projections(s, "cbs").select("player_id", "pred_ppg_cbs"), on="player_id", how="left")
        st = status(s, thru)
        tg = team_games(s, thru + 1, 17)
        h = (pri.join(to_date(s, thru), on="player_id", how="left")
                .join(after(s, thru + 1), on="player_id", how="left")
                .join(st, on="player_id", how="left")
                .with_columns(pl.lit(s).alias("season"), pl.col("n").fill_null(0)))
        h = h.with_columns(pl.coalesce("roster_team", "team_now").replace_strict(tg, default=None).alias("tg_left"))
        rows.append(h)
    return pl.concat(rows, how="diagonal_relaxed")


def fit(thru: int = 8) -> dict:
    """Fit k (prior weight, in games) and u (usage-luck weight) per position and league, and
    the chance of playing each remaining game by status. Saved to data/processed/ros_fit.json."""
    cache = DATA / "processed" / f"ros_history_wk{thru}.parquet"
    h = pl.read_parquet(cache) if cache.exists() else _history(thru=thru)
    h.write_parquet(cache)
    out = {"k": {}, "u": {}, "play": {}, "thru": thru}
    for lg in LEAGUES:
        out["k"][lg], out["u"][lg] = {}, {}
        for p in POS:
            d = h.filter(pl.col("position") == p, pl.col("n_after") >= 3, pl.col("n") >= 1,
                         pl.col(f"pred_ppg_{lg}").is_not_null())
            if p in ("QB", "RB", "WR", "TE"):  # fantasy-relevant only
                d = d.filter(pl.col(f"pred_ppg_{lg}") >= {"QB": 12, "RB": 6, "WR": 6, "TE": 5}[p])
            pri, n = d[f"pred_ppg_{lg}"].to_numpy(), d["n"].to_numpy()
            so, gap, y = d[f"ppg_{lg}_sofar"].to_numpy(), d["luck_pg"].fill_null(0).to_numpy(), d[f"ppg_{lg}_after"].to_numpy()
            best = None
            for k, u in itertools.product(K_GRID, U_GRID if p not in ("K", "DEF") else [0.0]):
                pred = (k * pri + n * (so + u * gap)) / (k + n)
                mae = float(np.mean(np.abs(pred - y)))
                if best is None or mae < best[0]:
                    best = (mae, k, u)
            base = float(np.mean(np.abs(pri - y)))
            sofar = float(np.mean(np.abs(so - y)))
            out["k"][lg][p], out["u"][lg][p] = best[1], best[2]
            out.setdefault("check", {}).setdefault(lg, {})[p] = {
                "n": len(y), "mae_preseason": round(base, 2), "mae_sofar": round(sofar, 2), "mae_ros": round(best[0], 2)}
    pl_ = h.filter(pl.col("tg_left") > 0, pl.col("status").is_not_null(), pl.col("n") >= 2,
                   pl.col("position").is_in(["QB", "RB", "WR", "TE"]),
                   pl.col("pred_ppg_burbs") >= 8)  # fantasy-relevant players
    pl_ = pl_.with_columns((pl.col("n_after").fill_null(0) / pl.col("tg_left")).clip(0, 1).alias("rate"))
    out["play"] = {s: round(float(v), 3) for s, v in pl_.group_by("status").agg(pl.col("rate").mean()).iter_rows()}
    import json
    (DATA / "processed" / "ros_fit.json").write_text(json.dumps(out, indent=1))
    return out


def project(season: int, thru: int, league: str, fitp: dict | None = None) -> pl.DataFrame:
    import json
    fitp = fitp or json.loads((DATA / "processed" / "ros_fit.json").read_text())
    k, u, play = fitp["k"][league], fitp["u"][league], fitp["play"]
    col = f"pred_ppg_{league}"
    pri = projections(season, league).select("player_id", "player_display_name", "position", "team",
                                             pl.col(col).alias("prior"), "rookie")
    so = to_date(season, thru)
    names = (_weekly(season).sort("week").group_by("player_id")
             .agg(pl.col("player_display_name").last(), pl.col("position").last()))
    # players with no preseason projection (undrafted, late call-ups): a low prior
    low = pri.group_by("position").agg(pl.col("prior").quantile(0.2).alias("low"))
    d = (so.join(names, on="player_id", how="left")
           .join(pri.drop("player_display_name", "position"), on="player_id", how="full", coalesce=True))
    d = d.join(pri.select("player_id", pl.col("player_display_name").alias("pn"), pl.col("position").alias("pp")),
               on="player_id", how="left").with_columns(
        pl.coalesce("player_display_name", "pn").alias("player_display_name"),
        pl.coalesce("pp", "position").alias("position")).drop("pn", "pp")
    d = d.filter(pl.col("position").is_in(POS)).join(low, on="position", how="left").with_columns(
        pl.col("prior").fill_null(pl.col("low")), pl.col("n").fill_null(0),
        pl.col("rookie").fill_null(False), pl.col("luck_pg").fill_null(0))
    kk = pl.col("position").replace_strict(k, return_dtype=pl.Float64)
    uu = pl.col("position").replace_strict(u, return_dtype=pl.Float64)
    obs = pl.col(f"ppg_{league}_sofar").fill_null(0) + uu * pl.col("luck_pg")
    d = d.with_columns(((kk * pl.col("prior") + pl.col("n") * obs) / (kk + pl.col("n"))).alias(col))

    st = status(season, thru)
    d = d.join(st, on="player_id", how="left").with_columns(
        pl.when(pl.col("position") == "DEF").then(pl.col("player_id"))
          .otherwise(pl.coalesce("roster_team", "team_now", "team")).alias("team"),
        pl.when(pl.col("position") == "DEF").then(pl.lit("active"))
          .otherwise(pl.col("status").fill_null("active")).alias("status"))
    d = d.filter(pl.col("status") != "gone")
    reg, po = team_games(season, thru + 1, 14), team_games(season, 15, 17)
    rate = (pl.when(pl.col("position").is_in(["K", "DEF"])).then(
                pl.when(pl.col("status") == "active").then(0.97).otherwise(play["out_long"]))
            .otherwise(pl.col("status").replace_strict(play, default=play["active"], return_dtype=pl.Float64)))
    d = d.with_columns(
        pl.col("team").replace_strict(reg, default=0, return_dtype=pl.Int64).alias("games_left_reg"),
        pl.col("team").replace_strict(po, default=0, return_dtype=pl.Int64).alias("games_left_po"),
        rate.alias("play_rate"))
    return d.with_columns(
        (pl.col("games_left_reg") * pl.col("play_rate")).alias("exp_reg"),
        (pl.col("games_left_po") * pl.col("play_rate")).alias("exp_po"))
