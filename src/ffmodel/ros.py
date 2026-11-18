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
R_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
# Rookies (fitted on 2021-25 at week 8, small samples: 23 QB, 43 RB, 88 WR, 37 TE): the draft-slot
# preseason projection deserves less weight for QB/WR, and recent usage more weight for RB/TE.
ROOKIE_K_MULT = {"QB": 0.25, "WR": 0.25}
ROOKIE_R = {"QB": 0.75, "RB": 1.0, "TE": 0.75}
S_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]


def _weekly(season: int) -> pl.DataFrame:
    """Player-week fantasy points (both leagues) incl. kickers and defenses, plus usage luck."""
    w = pl.read_parquet(DATA / "processed" / "player_weeks.parquet").filter(pl.col("season") == season)
    o_all = data.opportunity([season])
    # team-weeks whose usage data is published; the newest week arrives a day or two after the box scores,
    # and until then a missing row means "not published yet", not "no touches"
    have = o_all.select(pl.col("posteam").alias("team"), pl.col("week").cast(pl.Int32)).unique().with_columns(pl.lit(True).alias("_pub"))
    o = o_all.filter(pl.col("player_id").is_not_null()).select(
        "player_id", pl.col("week").cast(pl.Int32),
        (pl.col("total_fantasy_points_exp").cast(pl.Float64) - pl.col("total_fantasy_points").cast(pl.Float64))
        .alias("luck_gap"), pl.col("total_fantasy_points_exp").cast(pl.Float64).alias("xfp"))
    w = (w.select("player_id", "player_display_name", "position", "team", pl.col("week").cast(pl.Int32),
                  "fpts_burbs", "fpts_cbs")
          .join(o.group_by("player_id", "week").agg(pl.col("luck_gap").sum(), pl.col("xfp").sum()),
                on=["player_id", "week"], how="left")
          .join(have, on=["team", "week"], how="left")
          # a game played with no touches is zero usage, not missing (only once that game's usage data is out)
          .with_columns(pl.when(pl.col("position").is_in(["QB", "RB", "WR", "TE"]) & pl.col("_pub").fill_null(False))
                          .then(pl.col("xfp").fill_null(0)).otherwise(pl.col("xfp")).alias("xfp")).drop("_pub"))
    d = (dst_weeks().filter(pl.col("season") == season)
         .select(pl.col("team").alias("player_id"), (pl.col("team") + " DEF").alias("player_display_name"),
                 pl.lit("DEF").alias("position"), "team", pl.col("week").cast(pl.Int32), "fpts_burbs", "fpts_cbs",
                 pl.lit(None, dtype=pl.Float64).alias("luck_gap"), pl.lit(None, dtype=pl.Float64).alias("xfp")))
    return pl.concat([w, d], how="vertical_relaxed")


def _healthy_dnp(season: int, thru: int, w: pl.DataFrame) -> pl.DataFrame:
    """Games a player sat out while healthy: on his team's active roster, not on the injury report at all,
    team played, but no stats. Counted as zero usage for the role trend (e.g. a backup who stops
    playing once the starters are back)."""
    r = (data.rosters_weekly([season]).filter(pl.col("week") <= thru, pl.col("game_type") == "REG", pl.col("status") == "ACT",
                                               pl.col("position").is_in(HEALTHY_DNP_POS), pl.col("gsis_id").is_not_null())
         .select(pl.col("gsis_id").alias("player_id"), pl.col("week").cast(pl.Int32), "team"))
    inj = (data.injuries([season]).filter(pl.col("week") <= thru, pl.col("report_status").is_not_null())
           .select(pl.col("gsis_id").alias("player_id"), pl.col("week").cast(pl.Int32)))
    played = w.select("team", "week").unique()
    first = w.group_by("player_id").agg(pl.col("week").min().alias("first"))
    return (r.join(played, on=["team", "week"]).join(w.select("player_id", "week"), on=["player_id", "week"], how="anti")
             .join(inj, on=["player_id", "week"], how="anti").join(first, on="player_id")
             .filter(pl.col("week") > pl.col("first"))
             .select("player_id", "week", pl.lit(0.0).alias("xfp")))


HEALTHY_DNP = True  # backtest 2021-25 (weeks 6/9/12): RBs affected (n=28/22) MAE 3.13 -> 2.87 burbs, 3.37 -> 2.97 cbs;
HEALTHY_DNP_POS = ["RB"]  # TEs mixed (small n), QB/WR have no role weight, so RBs only


def to_date(season: int, thru: int) -> pl.DataFrame:
    w = _weekly(season).filter(pl.col("week") <= thru)
    # QB relief / mop-up appearances (under 10 passes) aren't representative games
    pw = pl.read_parquet(DATA / "processed" / "player_weeks.parquet").filter(
        pl.col("season") == season, pl.col("position") == "QB", pl.col("attempts").fill_null(0) < 10)
    w = w.join(pw.select("player_id", pl.col("week").cast(pl.Int32)), on=["player_id", "week"], how="anti")
    so = _to_date(w)
    if not HEALTHY_DNP:
        return so
    z = pl.concat([w.filter(pl.col("xfp").is_not_null()).select("player_id", "week", "xfp"), _healthy_dnp(season, thru, w)], how="vertical_relaxed")
    tr = z.group_by("player_id").agg(
        pl.col("xfp").sort_by("week").tail(3).mean().alias("xfp_last3"),
        pl.col("xfp").sort_by("week").tail(3).median().alias("xfp_med_last3"),
        pl.col("xfp").sort_by("week").head(pl.len() - 3).median().alias("xfp_med_before"))
    return (so.drop("xfp_last3", "xfp_med_last3", "xfp_med_before", "role_trend").join(tr, on="player_id", how="left")
              .with_columns((pl.col("xfp_last3") - pl.col("xfp_pg")).fill_null(0).alias("role_trend")))


def _to_date(w: pl.DataFrame) -> pl.DataFrame:
    return (w.group_by("player_id").agg(
        pl.len().alias("n"), pl.col("team").sort_by("week").last().alias("team_now"),
        *[pl.col(f"fpts_{lg}").mean().alias(f"ppg_{lg}_sofar") for lg in LEAGUES],
        pl.col("luck_gap").mean().fill_null(0).alias("luck_pg"),
        # role trend: expected fantasy points from usage (targets, carries, where on the field) in his
        # last 3 games vs his season average; > 0 = his role is growing (e.g. Tuten taking over JAX)
        pl.col("xfp").mean().alias("xfp_pg"),
        pl.col("xfp").sort_by("week").tail(3).mean().alias("xfp_last3"),
        # for the displayed Role flag: medians, so one huge game (e.g. a 24-target day) can't flip it
        pl.col("xfp").sort_by("week").tail(3).median().alias("xfp_med_last3"),
        pl.col("xfp").sort_by("week").head(pl.len() - 3).median().alias("xfp_med_before"))
    .with_columns((pl.col("xfp_last3") - pl.col("xfp_pg")).fill_null(0).alias("role_trend")))


def after(season: int, start: int, end: int = 17) -> pl.DataFrame:
    w = _weekly(season).filter(pl.col("week").is_between(start, end))
    return w.group_by("player_id").agg(pl.len().alias("n_after"),
                                       *[pl.col(f"fpts_{lg}").mean().alias(f"ppg_{lg}_after") for lg in LEAGUES])


def status(season: int, week: int, roster_week: int | None = None) -> pl.DataFrame:
    """Roster status as of `week`: out_long (IR / PUP / suspended), out_week (ruled Out or
    Doubtful on that week's injury report), active, or gone (cut / retired)."""
    # roster_week: take the team from a later roster (e.g. next week's, after a trade); default = week
    r = data.rosters_weekly([season]).filter(pl.col("week") <= (roster_week or week), pl.col("gsis_id").is_not_null())
    r = r.sort("week").group_by("gsis_id").agg(pl.all().last())
    inj = (data.injuries([season]).filter(pl.col("week") == (roster_week or week), pl.col("gsis_id").is_not_null())
           .group_by("gsis_id").agg(pl.col("report_status").last()))
    r = r.join(inj, on="gsis_id", how="left")
    return r.select(
        pl.col("gsis_id").alias("player_id"), pl.col("team").alias("roster_team"),
        pl.when(pl.col("status").is_in(["CUT", "RET"])).then(pl.lit("gone"))
          .when(pl.col("status").is_in(["RES", "EXE", "SUS"])).then(pl.lit("out_long"))
          .when((pl.col("status") == "DEV") | pl.col("status_description_abbr").str.starts_with("P").fill_null(False))
          .then(pl.lit("practice"))
          .when(pl.col("report_status").is_in(["Out", "Doubtful"])).then(pl.lit("out_week"))
          .otherwise(pl.lit("active")).alias("status"),
        pl.col("status_description_abbr").alias("status_code"))


def team_games(season: int, start: int, end: int, unplayed_before: bool = False) -> dict:
    s = data.schedules([season]).filter(pl.col("game_type") == "REG")
    s = s.filter(pl.col("week").is_between(start, end)
                 | (pl.lit(unplayed_before) & (pl.col("week") < start) & pl.col("home_score").is_null()))
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


def _hist_full(thru: int) -> pl.DataFrame:
    """History rows for one cut-off week, with schedule columns, cached."""
    cache = DATA / "processed" / f"ros_history_wk{thru}.parquet"
    if cache.exists():
        return pl.read_parquet(cache)
    h = _history(thru=thru)
    parts = []
    for s_ in h["season"].unique().to_list():
        a = sos(s_, thru, "burbs").rename({"sos_faced": "sos_faced_burbs", "sos_left": "sos_left_burbs"})
        b = sos(s_, thru, "cbs").rename({"sos_faced": "sos_faced_cbs", "sos_left": "sos_left_cbs"})
        parts.append(a.join(b, on="player_id", how="full", coalesce=True)
                      .with_columns(pl.lit(s_).cast(h["season"].dtype).alias("season")))
    h = h.join(pl.concat(parts), on=["player_id", "season"], how="left")
    h.write_parquet(cache)
    return h


def fit(thru: int = 8) -> dict:
    """Fit k (prior weight, in games) and u (usage-luck weight) per position and league, and
    the chance of playing each remaining game by status. Saved to data/processed/ros_fit.json."""
    # pooled over the weeks either side of `thru`: a single week's fit moved the role and schedule weights
    # around from one week to the next (TE role 0.5 at week 8, 0 at week 9), which is noise, not signal
    h = pl.concat([_hist_full(t).with_columns(pl.lit(t).alias("thru")) for t in (thru - 1, thru, thru + 1)],
                  how="diagonal_relaxed")
    out = {"k": {}, "u": {}, "r": {}, "play": {}, "thru": thru}
    for lg in LEAGUES:
        out["k"][lg], out["u"][lg], out["r"][lg] = {}, {}, {}
        for p in POS:
            d = h.filter(pl.col("position") == p, pl.col("n_after") >= 3, pl.col("n") >= 1,
                         pl.col(f"pred_ppg_{lg}").is_not_null())
            if p in ("QB", "RB", "WR", "TE"):  # fantasy-relevant only
                d = d.filter(pl.col(f"pred_ppg_{lg}") >= {"QB": 12, "RB": 6, "WR": 6, "TE": 5}[p])
            pri, n = d[f"pred_ppg_{lg}"].to_numpy(), d["n"].to_numpy()
            so, gap, y = d[f"ppg_{lg}_sofar"].to_numpy(), d["luck_pg"].fill_null(0).to_numpy(), d[f"ppg_{lg}_after"].to_numpy()
            rt = d["role_trend"].fill_null(0).to_numpy()
            skill = p not in ("K", "DEF")
            best, no_role = None, None
            for k, u, r in itertools.product(K_GRID, U_GRID if skill else [0.0], R_GRID if skill else [0.0]):
                pred = (k * pri + n * (so + u * gap + r * rt)) / (k + n)
                mae = float(np.mean(np.abs(pred - y)))
                if best is None or mae < best[0]:
                    best = (mae, k, u, r)
                if r == 0 and (no_role is None or mae < no_role):
                    no_role = mae
            base = float(np.mean(np.abs(pri - y)))
            sofar = float(np.mean(np.abs(so - y)))
            out["k"][lg][p], out["u"][lg][p], out["r"][lg][p] = best[1], best[2], best[3]
            # schedule: take the faced schedule out of results so far, add the remaining schedule
            sf, sl = d[f"sos_faced_{lg}"].fill_null(0).to_numpy(), d[f"sos_left_{lg}"].fill_null(0).to_numpy()
            k_, u_, r_ = best[1], best[2], best[3]
            sbest = None
            for sw in S_GRID:
                pred = (k_ * pri + n * (so + u_ * gap + r_ * rt - sw * sf)) / (k_ + n) + sw * sl
                mae = float(np.mean(np.abs(pred - y)))
                if sbest is None or mae < sbest[0] - 1e-9:
                    sbest = (mae, sw)
            out.setdefault("s", {}).setdefault(lg, {})[p] = sbest[1]
            no_sched = best[0]
            best = (sbest[0],) + best[1:]
            out.setdefault("check", {}).setdefault(lg, {})[p] = {
                "n": len(y), "mae_preseason": round(base, 2), "mae_sofar": round(sofar, 2), "mae_ros": round(best[0], 2), "mae_ros_no_role": round(no_role, 2),
                "mae_ros_no_schedule": round(no_sched, 2)}
    pl_ = h.filter(pl.col("tg_left") > 0, pl.col("status").is_not_null(), pl.col("n") >= 2,
                   pl.col("position").is_in(["QB", "RB", "WR", "TE"]),
                   pl.col("pred_ppg_burbs") >= 8)  # fantasy-relevant players
    pl_ = pl_.with_columns((pl.col("n_after").fill_null(0) / pl.col("tg_left")).clip(0, 1).alias("rate"))
    out["play"] = {s: round(float(v), 3) for s, v in pl_.group_by("status").agg(pl.col("rate").mean()).iter_rows()}
    # practice squad / no games yet (healthy but not playing): rarely play, and score less when they do
    q = h.filter(pl.col("tg_left") > 0, pl.col("position").is_in(["QB", "RB", "WR", "TE"]),
                 pl.col("pred_ppg_burbs") >= 6).with_columns(
        (pl.col("n_after").fill_null(0) / pl.col("tg_left")).clip(0, 1).alias("rate"),
        pl.when(pl.col("status_code").str.starts_with("P").fill_null(False)).then(pl.lit("practice"))
          .when((pl.col("n") == 0) & (pl.col("status") == "active")).then(pl.lit("no_games"))
          .otherwise(None).alias("grp")).filter(pl.col("grp").is_not_null())
    for g_, rate, ratio in q.group_by("grp").agg(
            pl.col("rate").mean(),
            (pl.col("ppg_burbs_after").mean() / pl.col("pred_ppg_burbs").mean())).iter_rows():
        out["play"][g_] = round(float(rate), 3)
        out.setdefault("ppg_ratio", {})[g_] = round(float(ratio), 3)
    import json
    (DATA / "processed" / "ros_fit.json").write_text(json.dumps(out, indent=1))
    return out


def project(season: int, thru: int, league: str, fitp: dict | None = None) -> pl.DataFrame:
    import json
    fitp = fitp or json.loads((DATA / "processed" / "ros_fit.json").read_text())
    k, u, play = fitp["k"][league], fitp["u"][league], fitp["play"]
    rr = fitp.get("r", {}).get(league, {p: 0.0 for p in POS})
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
        pl.col("rookie").fill_null(False), pl.col("luck_pg").fill_null(0), pl.col("role_trend").fill_null(0))
    kk = pl.col("position").replace_strict(k, return_dtype=pl.Float64)
    uu = pl.col("position").replace_strict(u, return_dtype=pl.Float64)
    rt = pl.col("position").replace_strict(rr, return_dtype=pl.Float64)
    rk = pl.col("rookie").fill_null(False)
    kk = pl.when(rk).then(kk * pl.col("position").replace_strict(ROOKIE_K_MULT, default=1.0, return_dtype=pl.Float64)).otherwise(kk)
    rt = pl.when(rk).then(pl.max_horizontal(rt, pl.col("position").replace_strict(ROOKIE_R, default=0.0, return_dtype=pl.Float64))).otherwise(rt)
    obs = pl.col(f"ppg_{league}_sofar").fill_null(0) + uu * pl.col("luck_pg") + rt * pl.col("role_trend").fill_null(0)
    ss = pl.col("position").replace_strict(fitp.get("s", {}).get(league, {p: 0.0 for p in POS}), return_dtype=pl.Float64)
    d = d.join(sos(season, thru, league), on="player_id", how="left").with_columns(
        pl.col("sos_faced").fill_null(0), pl.col("sos_left").fill_null(0))
    obs = obs - ss * pl.col("sos_faced")
    d = d.with_columns(((kk * pl.col("prior") + pl.col("n") * obs) / (kk + pl.col("n"))
                        + ss * pl.col("sos_left")).alias(col), (ss * pl.col("sos_left")).alias("sched_adj"))

    st = status(season, thru, roster_week=thru + 1)  # current team, incl. trades since last week
    d = d.join(st, on="player_id", how="left").with_columns(
        pl.when(pl.col("position") == "DEF").then(pl.col("player_id"))
          .otherwise(pl.coalesce("roster_team", "team_now", "team")).alias("team"),
        pl.when(pl.col("position") == "DEF").then(pl.lit("active"))
          .otherwise(pl.col("status").fill_null("active")).alias("status"))
    # traded since his last game: the schedule left is his new team's, not his old team's
    tl = sos_left_by_team(season, thru, league).rename({"sos_left": "sos_left_now"})
    d = d.join(tl, on=["position", "team"], how="left").with_columns(
        pl.when((pl.col("position") != "DEF") & pl.col("team_now").is_not_null() & (pl.col("team") != pl.col("team_now")))
          .then(pl.col("sos_left_now").fill_null(0)).otherwise(pl.col("sos_left")).alias("sos_left_new")).drop("sos_left_now")
    d = d.with_columns((pl.col(col) + ss * (pl.col("sos_left_new") - pl.col("sos_left"))).alias(col),
                       (ss * pl.col("sos_left_new")).alias("sched_adj"),
                       pl.col("sos_left_new").alias("sos_left")).drop("sos_left_new")
    # not on any NFL roster this season and no games played (released / unsigned, e.g. Tyreek Hill 2026): his
    # preseason team is stale, so drop him instead of showing a projection for a team he isn't on (Nov 18 2026)
    d = d.with_columns(pl.when(pl.col("roster_team").is_null() & (pl.col("n").fill_null(0) == 0) & (pl.col("position") != "DEF"))
                         .then(pl.lit("gone")).otherwise(pl.col("status")).alias("status"))
    d = d.with_columns(pl.when((pl.col("status") == "active") & (pl.col("n") == 0) & (pl.col("position") != "DEF"))
                         .then(pl.lit("no_games")).otherwise(pl.col("status")).alias("status"))
    manual = DATA / "input" / f"status_{season}.csv"  # news the data doesn't have yet: Player, Status
    if manual.exists():
        m = pl.read_csv(manual).select(pl.col("Player").alias("player_display_name"), pl.col("Status").alias("manual"))
        d = d.join(m, on="player_display_name", how="left").with_columns(
            pl.coalesce("manual", "status").alias("status")).drop("manual")
    d = d.with_columns(pl.lit(0).alias("missed_reg"), pl.lit(0).alias("missed_po"), pl.lit("").alias("news"))
    d = apply_news(d, season, thru)
    d = apply_depth(d, season, col)
    d = fill_in_starters(d, season, col)
    d = wr_depth_moves(d, season, thru, col)
    ratio = fitp.get("ppg_ratio", {})
    d = d.with_columns((pl.col(col) * pl.col("status").replace_strict(ratio, default=1.0, return_dtype=pl.Float64)).alias(col))
    d = d.filter(pl.col("status") != "gone")
    d = qb_change(d, season, thru, league, col)
    # work left by a teammate who just went down long-term (fresh absence, so recent roles don't show it yet)
    outs = d.filter(pl.col("status").is_in(["out_long", "season", "gone"]) | (pl.col("missed_reg") >= 3))["player_id"].to_list()
    v = vacated(season, thru, league, out_ids=outs)
    d = d.join(v, on="player_id", how="left").with_columns(
        (pl.col("inherit").fill_null(0) * pl.col("position").replace_strict(VAC_BETA, default=0.0, return_dtype=pl.Float64))
        .alias("vac_adj")).with_columns((pl.col(col) + pl.col("vac_adj")).alias(col))
    reg, po = team_games(season, thru + 1, 14, unplayed_before=True), team_games(season, 15, 17)
    play = {**play, "dated": play["active"], "season": 0.0, "backup": 0.1}  # backup QB: rough, not fitted
    rate = (pl.when(pl.col("position").is_in(["K", "DEF"]) & ~pl.col("status").is_in(["season", "dated"])).then(
                pl.when(pl.col("status") == "active").then(0.97).otherwise(play["out_long"]))
            .otherwise(pl.col("status").replace_strict(play, default=play["active"], return_dtype=pl.Float64)))
    d = d.with_columns(
        pl.col("team").replace_strict(reg, default=0, return_dtype=pl.Int64).alias("games_left_reg"),
        pl.col("team").replace_strict(po, default=0, return_dtype=pl.Int64).alias("games_left_po"),
        rate.alias("play_rate"))
    d = d.with_columns(
        ((pl.col("games_left_reg") - pl.col("missed_reg")).clip(0) * pl.col("play_rate")).alias("exp_reg"),
        ((pl.col("games_left_po") - pl.col("missed_po")).clip(0) * pl.col("play_rate")).alias("exp_po"))
    d = short_fill_ins(d, season, thru, league, col)
    # fill-in starter: only the games the regular starter misses, then the backup rate
    fi = pl.col("fill_reg").is_not_null()
    return d.with_columns(
        pl.when(fi).then(pl.min_horizontal("exp_reg", pl.col("fill_reg") * pl.col("play_rate")
                                           + (pl.col("games_left_reg") - pl.col("fill_reg")).clip(0) * play["backup"]))
          .otherwise(pl.col("exp_reg")).alias("exp_reg"),
        pl.when(fi).then(pl.min_horizontal("exp_po", pl.col("fill_po") * pl.col("play_rate")
                                           + (pl.col("games_left_po") - pl.col("fill_po")).clip(0) * play["backup"]))
          .otherwise(pl.col("exp_po")).alias("exp_po"))


def _remaining_opps(season: int, thru: int, end: int = 17) -> pl.DataFrame:
    s = data.schedules([season]).filter(pl.col("game_type") == "REG")
    s = s.filter(pl.col("week").is_between(thru + 1, end) | ((pl.col("week") <= thru) & pl.col("home_score").is_null()))
    return pl.concat([s.select(pl.col("home_team").alias("team"), pl.col("away_team").alias("opp")),
                      s.select(pl.col("away_team").alias("team"), pl.col("home_team").alias("opp"))])


def sos_left_by_team(season: int, thru: int, league: str) -> pl.DataFrame:
    """Remaining-schedule effect per (position, team) for skill positions: position, team, sos_left."""
    from .opponent import defense_effects
    col = f"fpts_{league}"
    w = pl.read_parquet(DATA / "processed" / "player_weeks.parquet").filter(
        pl.col("season") == season, pl.col("week") <= thru)
    eff = defense_effects(w, col).select("position", pl.col("opponent_team").alias("opp"), pl.col(f"def_{col}").alias("eff"))
    rem = _remaining_opps(season, thru)
    return (rem.join(eff, on="opp", how="inner").group_by("position", "team")
               .agg(pl.col("eff").fill_null(0).mean().alias("sos_left")))


def sos(season: int, thru: int, league: str, k_def: float = 4.0) -> pl.DataFrame:
    """Strength of schedule from this season's games through `thru`.

    Skill players: each defense's points allowed to the position above what those same players
    usually score (opponent.defense_effects: run defense shows up at RB, pass defense at QB/WR/TE).
    Fantasy defenses: each offense's points given up to opposing defenses vs average, shrunk.
    Returns per player: sos_faced (avg of opponents already played) and sos_left (avg of
    opponents still to play), in fantasy points per game.
    """
    from .opponent import defense_effects
    col = f"fpts_{league}"
    w = pl.read_parquet(DATA / "processed" / "player_weeks.parquet").filter(
        pl.col("season") == season, pl.col("week") <= thru)
    eff = defense_effects(w, col).select("position", pl.col("opponent_team").alias("opp"), pl.col(f"def_{col}").alias("eff"))
    faced = (w.select("player_id", "position", pl.col("opponent_team").alias("opp"))
               .join(eff, on=["position", "opp"], how="left")
               .group_by("player_id").agg(pl.col("eff").fill_null(0).mean().alias("sos_faced")))
    pos = (w.sort("week").group_by("player_id").agg(pl.col("position").last(), pl.col("team").last()))
    rem = _remaining_opps(season, thru)
    left = (pos.join(rem, on="team", how="left").join(eff, on=["position", "opp"], how="left")
               .group_by("player_id").agg(pl.col("eff").fill_null(0).mean().fill_null(0).alias("sos_left")))
    skill = faced.join(left, on="player_id", how="full", coalesce=True)

    dw = dst_weeks().filter(pl.col("season") == season, pl.col("week") <= thru)
    sch = data.schedules([season]).filter(pl.col("game_type") == "REG", pl.col("week") <= thru)
    opp_of = pl.concat([sch.select("week", pl.col("home_team").alias("team"), pl.col("away_team").alias("opp")),
                        sch.select("week", pl.col("away_team").alias("team"), pl.col("home_team").alias("opp"))])
    dw = dw.join(opp_of.with_columns(pl.col("week").cast(dw["week"].dtype)), on=["week", "team"], how="left")
    mu = dw[col].mean()
    off = dw.group_by("opp").agg(((pl.col(col) - mu).sum() / (pl.len() + k_def)).alias("eff"))
    dfaced = dw.join(off, on="opp", how="left").group_by("team").agg(pl.col("eff").fill_null(0).mean().alias("sos_faced"))
    dleft = rem.join(off, on="opp", how="left").group_by("team").agg(pl.col("eff").fill_null(0).mean().alias("sos_left"))
    dst_ = dfaced.join(dleft, on="team", how="full", coalesce=True).rename({"team": "player_id"})
    return pl.concat([skill, dst_], how="vertical_relaxed").with_columns(
        pl.col("sos_faced").fill_null(0), pl.col("sos_left").fill_null(0))


def role_flags(season: int, thru: int) -> pl.DataFrame:
    """Displayed Role flag, from playing time and share of the team's work (not points):
    snap share, and share of team carries + targets (RB) or targets (WR/TE), last 3 games vs
    earlier games, compared on medians so one odd game can't flip it.
    Up/down if snap share moves 10+ points or work share 8+ points (QBs: snaps 20+ points only)."""
    ps = pl.read_parquet(DATA / "processed" / "player_seasons.parquet").filter(pl.col("season") == season)
    pfr = ps.select("player_id", "pfr_id").drop_nulls()
    sn = (data.snap_counts([season]).filter(pl.col("week") <= thru, pl.col("game_type") == "REG")
          .join(pfr, left_on="pfr_player_id", right_on="pfr_id")
          .select("player_id", pl.col("week").cast(pl.Int32), pl.col("offense_pct").cast(pl.Float64).alias("snap")))
    w = pl.read_parquet(DATA / "processed" / "player_weeks.parquet").filter(
        pl.col("season") == season, pl.col("week") <= thru)
    w = w.with_columns(pl.col("week").cast(pl.Int32),
                       pl.col("carries").fill_null(0).alias("car"), pl.col("targets").fill_null(0).alias("tgt"))
    team = w.group_by("team", "week").agg(pl.col("car").sum().alias("t_car"), pl.col("tgt").sum().alias("t_tgt"))
    w = (w.join(team, on=["team", "week"])
          .with_columns(pl.when(pl.col("position") == "RB")
                          .then((pl.col("car") + pl.col("tgt")) / (pl.col("t_car") + pl.col("t_tgt")))
                          .otherwise(pl.col("tgt") / pl.col("t_tgt")).alias("share"))
          .join(sn, on=["player_id", "week"], how="left"))
    if HEALTHY_DNP:  # sat out while healthy = no snaps, no work
        z = _healthy_dnp(season, thru, _weekly(season).filter(pl.col("week") <= thru)).join(
            w.group_by("player_id").agg(pl.col("position").last()), on="player_id")
        w = pl.concat([w, z.select("player_id", "week", "position", pl.lit(0.0).alias("share"), pl.lit(0.0).alias("snap"))],
                      how="diagonal_relaxed")

    def split(c):
        ok = pl.col(c).is_not_null()  # skip weeks whose snap counts are not published yet
        v = pl.col(c).filter(ok).sort_by(pl.col("week").filter(ok))
        return [v.tail(3).median().alias(f"{c}_l3"), v.head(v.len() - 3).median().alias(f"{c}_pre")]
    g = (w.group_by("player_id").agg(pl.col("position").last(), pl.len().alias("g"), *split("snap"), *split("share"))
          .filter(pl.col("g") >= 4)
          .with_columns((pl.col("snap_l3") - pl.col("snap_pre")).alias("d_snap"),
                        (pl.col("share_l3") - pl.col("share_pre")).alias("d_share")))
    qb = pl.col("position") == "QB"
    up = pl.when(qb).then(pl.col("d_snap") >= 0.20).otherwise((pl.col("d_snap") >= 0.10) | (pl.col("d_share") >= 0.08))
    dn = pl.when(qb).then(pl.col("d_snap") <= -0.20).otherwise((pl.col("d_snap") <= -0.10) | (pl.col("d_share") <= -0.08))
    return g.select("player_id",
                    pl.when(up.fill_null(False) & ~dn.fill_null(False)).then(pl.lit("Role up"))
                      .when(dn.fill_null(False) & ~up.fill_null(False)).then(pl.lit("Role down"))
                      .otherwise(pl.lit("")).alias("role_flag"),
                    (pl.col("snap_l3") * 100).round(0).alias("snap_l3"), (pl.col("snap_pre") * 100).round(0).alias("snap_pre"),
                    (pl.col("share_l3") * 100).round(0).alias("share_l3"), (pl.col("share_pre") * 100).round(0).alias("share_pre"))


# Share of the vacated PPG a teammate actually picks up over the rest of the season. Backtest 2021-25 (after
# weeks 6/9/12, fresh long-term absences): RB 0.28-0.30 (MAE 3.67 -> 3.40, n=33), TE 0.25-0.5 (3.93 -> 3.2, n=13),
# WR ~0.07 and adding it made WR projections worse (n=82), so WRs get none: the targets spread out thin.
VAC_BETA = {"RB": 0.3, "TE": 0.4, "WR": 0.0}


def vacated(season: int, thru: int, league: str, recent_min: int = 2, out_ids: list | None = None) -> pl.DataFrame:
    """Work freed up by injured / released teammates, handed to whoever is getting the work now.

    For each team and position: points per game so far of teammates now on IR / PUP / suspended
    or released (2+ games played), times each active teammate's share of the work in his last
    3 games over the active teammates' total share. inherit = expected extra PPG from that.
    """
    w = pl.read_parquet(DATA / "processed" / "player_weeks.parquet").filter(
        pl.col("season") == season, pl.col("week") <= thru, pl.col("position").is_in(["RB", "WR", "TE"]))
    w = w.with_columns(pl.col("carries").fill_null(0).alias("car"), pl.col("targets").fill_null(0).alias("tgt"))
    team = w.group_by("team", "week").agg(pl.col("car").sum().alias("t_car"), pl.col("tgt").sum().alias("t_tgt"))
    w = w.join(team, on=["team", "week"]).with_columns(
        pl.when(pl.col("position") == "RB").then((pl.col("car") + pl.col("tgt")) / (pl.col("t_car") + pl.col("t_tgt")))
          .otherwise(pl.col("tgt") / pl.col("t_tgt")).alias("share"))
    p = (w.sort("week").group_by("player_id").agg(
            pl.col("team").last(), pl.col("position").last(), pl.len().alias("g"),
            pl.col(f"fpts_{league}").mean().alias("ppg"), pl.col("share").tail(3).mean().alias("share_l3"))
         .join(status(season, thru).select("player_id", "status"), on="player_id", how="left")
         .with_columns(pl.col("status").fill_null("active")))
    if out_ids is not None:  # current statuses (after injury news): long-term out = in out_ids
        p = p.with_columns(pl.when(pl.col("player_id").is_in(out_ids)).then(pl.lit("out_long"))
                             .when(pl.col("status").is_in(["out_long", "gone"])).then(pl.lit("active"))
                             .otherwise(pl.col("status")).alias("status"))
    # only fresh absences: he played in at least 2 of his team's last 3 games, so his teammates' recent
    # roles (role trend) don't already reflect the missing work
    tl3 = w.select("team", "week").unique().sort("week").group_by("team").agg(pl.col("week").tail(3).alias("l3"))
    recent = (w.join(tl3, on="team").filter(pl.col("week").is_in(pl.col("l3"))).group_by("player_id").len()
                .rename({"len": "g_l3"}))
    p = p.join(recent, on="player_id", how="left").with_columns(pl.col("g_l3").fill_null(0))
    gone = (p.filter(pl.col("status").is_in(["out_long", "gone"]), pl.col("g") >= 2, pl.col("g_l3") >= recent_min)
              .group_by("team", "position").agg(pl.col("ppg").sum().alias("vac_ppg")))
    act = p.filter(pl.col("status").is_in(["active", "out_week"]))
    tot = act.group_by("team", "position").agg(pl.col("share_l3").sum().alias("tot_share"))
    return (act.join(gone, on=["team", "position"]).join(tot, on=["team", "position"])
               .select("player_id", (pl.col("vac_ppg") * pl.col("share_l3") / pl.col("tot_share")).alias("inherit")))


def _remaining_games(season: int, thru: int) -> pl.DataFrame:
    s = data.schedules([season]).filter(pl.col("game_type") == "REG")
    s = s.filter((pl.col("week") > thru) | pl.col("home_score").is_null())
    return pl.concat([s.select("week", "gameday", pl.col(t).alias("team")) for t in ("home_team", "away_team")]
                     ).with_columns(pl.col("gameday").str.to_date())


def apply_news(d: pl.DataFrame, season: int, thru: int) -> pl.DataFrame:
    """Apply a pasted injury report (data/input/news_<season>.tsv: Player, Pos, Return, Status, Note;
    e.g. Rotowire's injury page). It is treated as the current, complete list for skill players
    and kickers:
      Questionable -> plays (normal risk); Doubtful -> likely out this week;
      Out / IR with a return date -> misses every game before that date, then normal risk;
      return date in Jan/Feb -> out for the fantasy season;
      IR with no real date yet (return shown as the next game) -> the usual IR rate;
      not on the list -> this week's official injury report, else his NFL roster status (IR stays IR)."""
    import datetime as dt
    from .market import _norm
    path = DATA / "input" / f"news_{season}.tsv"
    live = sleeper_injuries()
    if live is not None:  # live feed first; a pasted report (with return dates) overrides it below
        d = d.join(live, on="player_id", how="left").with_columns(
            # a player on no NFL roster stays gone whatever Sleeper lists (Tyreek Hill, Nov 18 2026)
            pl.when(pl.col("status") == "gone").then(pl.lit("gone"))
              .when(pl.col("sl_status").is_in(["IR", "PUP", "Sus", "NFI"])).then(pl.lit("out_long"))
              .when(pl.col("sl_status").is_in(["Out", "Doubtful"])).then(pl.lit("out_week"))
              .when(pl.col("sl_status") == "Questionable").then(pl.lit("active"))
              .otherwise(pl.col("status")).alias("status"),
            pl.when(pl.col("sl_status").is_not_null()).then(pl.concat_str([pl.lit("Sleeper: "), pl.col("sl_status")]))
              .otherwise(pl.col("news")).alias("news")).drop("sl_status")
    if not path.exists():
        return d
    news = pl.read_csv(path, separator="\t", quote_char=None).with_columns(pl.col("Note").fill_null(""))
    # dated injury feed (return dates for the whole league). It replaces swept notes for the same player;
    # only what Scott told us directly (note mentions Scott) outranks it.
    feed, _ = injury_feed(season)
    if feed is not None:
        feed = feed.filter(pl.col("Pos").is_in(["QB", "RB", "WR", "TE", "K", "PK"]),
                           ~pl.col("Note").str.contains("coach's decision|emergency third"),
                           # practice-squad injured list: keep the model's practice-squad status (Jawhar Jordan, Nov 2026)
                           ~pl.col("Note").str.to_lowercase().str.contains("practice squad injured"))
        fk = set(feed["Player"].map_elements(_norm, return_dtype=pl.Utf8).to_list())
        keep = news.filter(pl.col("Note").str.contains("Scott") | ~pl.col("Player").map_elements(_norm, return_dtype=pl.Utf8).is_in(list(fk)))
        kk = set(keep["Player"].map_elements(_norm, return_dtype=pl.Utf8).to_list())
        add = feed.filter(~pl.col("Player").map_elements(_norm, return_dtype=pl.Utf8).is_in(list(kk)))
        news = pl.concat([keep.select("Player", "Pos", "Return", "Status", "Note"), add.select("Player", "Pos", "Return", "Status", "Note")])
        news = news.unique("Player", keep="first", maintain_order=True)
    months = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}

    def ret(x):
        mon, day = x.split()
        y = season + 1 if months[mon] <= 3 else season
        return dt.date(y, months[mon], int(day))
    news = news.with_columns(pl.col("Player").map_elements(_norm, return_dtype=pl.Utf8).alias("key"),
                             pl.col("Return").map_elements(ret, return_dtype=pl.Date).alias("ret"))
    games = _remaining_games(season, thru)
    nxt = games.group_by("team").agg(pl.col("gameday").min().alias("next_game"))
    d = d.with_columns(pl.col("player_display_name").map_elements(_norm, return_dtype=pl.Utf8).alias("key"))
    d = (d.join(news.select("key", "Status", "ret", "Note", "Return"), on="key", how="left")
          .join(nxt, on="team", how="left"))
    # this week's official injury report (game statuses). It is used directly, so nothing depends on the notes
    # file being hand-updated:
    #   - listed on the report and not in the notes (or only a short-term note): the report's status
    #   - a short-term note (Q / D / Out for this week's game) for a player whose team's final report is in
    #     and who is not on it: cleared
    #   - the notes win when they are more severe (ruled out / inactive after the report) or long-term
    off = (data.injuries([season]).filter(pl.col("week") == thru + 1, pl.col("report_status").is_not_null(),
                                          pl.col("gsis_id").is_not_null())
           .group_by("gsis_id").agg(pl.col("report_status").last().alias("o_status"), pl.col("team").last().alias("o_team"),
                                    pl.col("report_primary_injury").last().alias("o_inj"),
                                    pl.col("practice_status").last().alias("o_prac")))
    final_teams = off["o_team"].unique().to_list()
    d = d.join(off.rename({"gsis_id": "player_id"}), on="player_id", how="left")
    sev = {"Questionable": 1, "Doubtful": 2, "Out": 3}
    t_sev = pl.col("Status").replace_strict(sev, default=0, return_dtype=pl.Int64)
    o_sev = pl.col("o_status").replace_strict(sev, default=0, return_dtype=pl.Int64)
    short = pl.col("Status").is_in(list(sev)) & (pl.col("ret") <= pl.col("next_game") + pl.duration(days=2))
    use_off = pl.col("o_status").is_not_null() & (pl.col("Status").is_null() | (short & (o_sev >= t_sev)))
    drop_note = short & pl.col("o_status").is_null() & pl.col("team").is_in(final_teams) & (t_sev < 3)
    o_ret = pl.when(pl.col("o_status") == "Questionable").then(pl.col("next_game")).otherwise(pl.col("next_game") + pl.duration(days=2))
    o_note = pl.concat_str([pl.col("o_inj").fill_null("").str.to_lowercase(), pl.lit(", official report"),
                            pl.when(pl.col("o_prac").fill_null("").str.starts_with("Did Not")).then(pl.lit(", DNP last practice"))
                              .when(pl.col("o_prac").fill_null("").str.starts_with("Limited")).then(pl.lit(", limited last practice"))
                              .otherwise(pl.lit(""))])
    d = d.with_columns(
        pl.when(use_off).then(pl.col("o_status")).when(drop_note).then(None).otherwise(pl.col("Status")).alias("Status"),
        pl.when(use_off).then(o_ret).otherwise(pl.col("ret")).alias("ret"),
        pl.when(use_off).then(o_note).otherwise(pl.col("Note")).alias("Note"),
    ).with_columns(pl.when(pl.col("ret").is_not_null()).then(pl.col("ret").dt.strftime("%b %-d")).otherwise(pl.col("Return")).alias("Return")
                   ).drop("o_status", "o_team", "o_inj", "o_prac")
    # games missed before the return date, split regular season (wk <= 14) / playoffs
    miss = (d.filter(pl.col("ret").is_not_null()).select("player_id", "team", "ret")
              .join(games, on="team").filter(pl.col("gameday") < pl.col("ret"))
              .group_by("player_id").agg((pl.col("week") <= 14).sum().alias("m_reg"),
                                         (pl.col("week") > 14).sum().alias("m_po")))
    d = d.join(miss, on="player_id", how="left")
    skill = pl.col("position").is_in(["QB", "RB", "WR", "TE", "K"])
    # a stale "Injured Reserve" note with no real return date, for a player the NFL roster now shows active: cleared
    stale_ir = (pl.col("Status") == "Injured Reserve") & (pl.col("status") == "active") & (pl.col("ret") <= pl.col("next_game"))
    d = d.with_columns(pl.when(stale_ir).then(None).otherwise(pl.col("Status")).alias("Status"))
    # fresh injured reserve with no note: out at least 4 games from when he went on it
    rw = data.rosters_weekly([season]).filter(pl.col("gsis_id").is_not_null(), pl.col("game_type") == "REG")
    lastw = int(rw["week"].max())
    res = rw.filter(pl.col("status") == "RES").group_by("gsis_id").agg(pl.col("week").alias("res_wks"))
    stint = []
    for pid, wks in res.iter_rows():
        wks = set(wks)
        if lastw not in wks:
            continue
        w0 = lastw
        while (w0 - 1) in wks:
            w0 -= 1
        stint.append((pid, w0 + 4))
    stint = pl.DataFrame(stint, schema=["player_id", "elig_wk"], orient="row") if stint else pl.DataFrame(
        schema={"player_id": pl.Utf8, "elig_wk": pl.Int64})
    irm = (d.filter(pl.col("status") == "out_long", pl.col("Status").is_null()).select("player_id", "team")
             .join(stint, on="player_id").join(games, on="team").filter(pl.col("week") < pl.col("elig_wk"))
             .group_by("player_id").agg((pl.col("week") <= 14).sum().alias("ir_reg"), (pl.col("week") > 14).sum().alias("ir_po"),
                                        pl.col("elig_wk").first()))
    back = (d.select("player_id", "team").join(stint, on="player_id").join(games, on="team")
              .filter(pl.col("week") >= pl.col("elig_wk")).group_by("player_id").agg(pl.col("gameday").min().alias("ir_back")))
    irm = irm.join(back, on="player_id", how="left")
    d = d.join(irm, on="player_id", how="left").with_columns(
        pl.when(pl.col("ir_reg").is_not_null()).then(pl.col("ir_reg")).otherwise(pl.col("missed_reg")).alias("missed_reg"),
        pl.when(pl.col("ir_po").is_not_null()).then(pl.col("ir_po")).otherwise(pl.col("missed_po")).alias("missed_po"),
        pl.when(pl.col("ir_reg").is_not_null() & skill)
          .then(pl.concat_str([pl.lit("Injured Reserve, est. return "), pl.col("ir_back").dt.strftime("%b %-d").fill_null("Feb 15"),
                               pl.lit(" (NFL roster: must miss 4 games, earliest week "), pl.col("elig_wk").cast(pl.Utf8), pl.lit(")")]))
          .when((pl.col("status") == "out_long") & pl.col("Status").is_null() & skill)
          .then(pl.lit("Injured Reserve, no return date (NFL roster)"))
          .otherwise(pl.col("news")).alias("news")).drop("ir_reg", "ir_po", "elig_wk", "ir_back")
    out = pl.col("Status").is_in(["Out", "Injured Reserve"])
    season_over = out & (pl.col("ret") >= dt.date(season + 1, 1, 1))
    dated = out & ~season_over & (pl.col("ret") > pl.col("next_game"))
    d = d.with_columns(
        pl.when(~skill).then(pl.col("status"))
          # not listed anywhere: the NFL roster status stands (injured reserve stays injured reserve)
          .when(pl.col("Status").is_null()).then(pl.col("status"))
          .when(pl.col("Status") == "Questionable").then(pl.lit("active"))
          .when(pl.col("Status") == "Doubtful").then(pl.lit("out_week"))
          .when(season_over).then(pl.lit("season"))
          .when(dated).then(pl.lit("dated"))
          .when(out & (pl.col("Status") == "Injured Reserve")).then(pl.lit("out_long"))
          .when(out).then(pl.lit("out_week"))
          .otherwise(pl.col("status")).alias("status"),
        pl.when(dated).then(pl.col("m_reg").fill_null(0)).otherwise(pl.col("missed_reg")).alias("missed_reg"),
        pl.when(dated).then(pl.col("m_po").fill_null(0)).otherwise(pl.col("missed_po")).alias("missed_po"),
        pl.when(pl.col("Status").is_not_null() & skill)
          .then(pl.concat_str([pl.col("Status"), pl.lit(", est. return "), pl.col("Return"),
                               pl.when(pl.col("Note") != "").then(pl.concat_str([pl.lit(" ("), pl.col("Note"), pl.lit(")")]))
                                 .otherwise(pl.lit(""))]))
          .otherwise(pl.col("news")).alias("news"))
    return d.drop("key", "Status", "ret", "Note", "Return", "next_game", "m_reg", "m_po")


ESPN_INJ_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
FEED_MAX_AGE_DAYS = 4


ESPN_SITE = "https://site.api.espn.com/apis"
FEED_POS = ("QB", "RB", "WR", "TE", "PK")


def espn_injury_rows() -> list[dict]:
    """Every QB/RB/WR/TE/PK with an ESPN expected return date: Team, Player, Pos, Return ("Nov 29"), Status, Note.
    ESPN's league injuries endpoint only lists each team's 25 most recent injury updates, so long-term IR players
    (e.g. a back on IR since October) drop off it. So: take that list, then read all 32 team rosters for injured
    skill players it left out and fetch each one's own page for the return date."""
    import datetime as dt
    import json
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor
    mon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    def get(url, tries=3):
        for k in range(tries):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30) as r:
                    return json.load(r)
            except Exception:
                if k == tries - 1:
                    raise

    def soft(url):   # one team's roster or one player's page failing shouldn't sink the whole feed
        try:
            return get(url)
        except Exception:
            return {}

    def row(team, a, i):
        rd = (i.get("details") or {}).get("returnDate")
        pos = (a.get("position") or {}).get("abbreviation")
        if not rd or pos not in FEED_POS:
            return None
        d_ = dt.date.fromisoformat(rd[:10])
        note = (i.get("shortComment") or "")[:150].replace("\t", " ").replace("\n", " ")
        return {"Team": team, "Player": a.get("displayName"), "Pos": pos, "Return": f"{mon[d_.month - 1]} {d_.day}",
                "Status": i.get("status") or "Out", "Note": note}

    js = get(ESPN_INJ_URL)
    out, seen, teams = [], set(), {}
    for t in js.get("injuries", []):
        teams[t["id"]] = t["displayName"]
        for i in t.get("injuries", []):
            a = i.get("athlete", {})
            seen.add(a.get("displayName"))
            r = row(t["displayName"], a, i)
            if r:
                out.append(r)
    with ThreadPoolExecutor(8) as ex:
        rosters = list(ex.map(lambda tid: (tid, soft(f"{ESPN_SITE}/site/v2/sports/football/nfl/teams/{tid}/roster")), teams))
        items = [(tid, a, g.get("position")) for tid, r in rosters for g in r.get("athletes", []) for a in g.get("items", [])]
        # practice-squad players (incl. the practice-squad injured list) keep the model's practice-squad status
        ps = {a["displayName"] for _, a, grp in items
              if grp == "practiceSquad" or "practice" in str((a.get("status") or {}).get("name", "")).lower()}
        todo = [(teams[tid], a["id"]) for tid, a, _ in items
                if a.get("injuries") and a["displayName"] not in seen and a["displayName"] not in ps
                and (a.get("position") or {}).get("abbreviation") in FEED_POS]
        pages = list(ex.map(lambda x: (x[0], soft(f"{ESPN_SITE}/common/v3/sports/football/nfl/athletes/{x[1]}")), todo))
    out = [r for r in out if r["Player"] not in ps]
    for team, pg in pages:
        a = pg.get("athlete", {})
        if a.get("injuries"):
            r = row(team, a, a["injuries"][0])
            if r:
                out.append(r)
    return out


def injury_feed(season: int) -> tuple[pl.DataFrame | None, str]:
    """League-wide injury list WITH expected return dates (ESPN's injuries page): Player, Pos, Return ("Nov 29"),
    Status, Note. The official report says who is out this week but not for how long; this does.
    Tried in order: ESPN's API directly (espn_injury_rows; needs site.api.espn.com allowed in the environment's network settings),
    then data/input/injury_feed_<season>.tsv, which the scheduled news sweep refreshes by reading the same page.
    Returns (rows or None, description of the source and its age)."""
    import datetime as dt
    try:
        rows = [{k: v for k, v in r.items() if k != "Team"} | {"Note": r["Note"] + " (ESPN)"} for r in espn_injury_rows()]
        if rows:
            return pl.DataFrame(rows), "ESPN live feed"
    except Exception:
        pass
    path = DATA / "input" / f"injury_feed_{season}.tsv"
    if not path.exists():
        return None, "no injury feed (ESPN blocked and no saved copy)"
    f = pl.read_csv(path, separator="\t", quote_char=None)
    age = (dt.date.today() - dt.date.fromisoformat(f["Fetched"].max())).days
    teams = f["Team"].n_unique()
    desc = f"ESPN injuries page, saved copy {age} day(s) old, {teams} of 32 teams"
    if age > FEED_MAX_AGE_DAYS:
        return None, desc + " (too old, ignored)"
    return f.select("Player", "Pos", "Return", "Status", pl.col("Note").fill_null("") + " (ESPN)"), desc


SLEEPER_URL = "https://api.sleeper.app/v1/players/nfl"


def sleeper_injuries() -> pl.DataFrame | None:
    """Live injury status from Sleeper (no key needed). Returns player_id, sl_status
    (IR / PUP / Sus / Out / Doubtful / Questionable ...) or None if the host is unreachable
    (it must be allowed in the environment's network settings)."""
    import json
    import urllib.request
    try:
        with urllib.request.urlopen(SLEEPER_URL, timeout=30) as r:
            players = json.load(r)
    except Exception:
        return None
    rows = [{"sleeper_id": str(k), "sl_status": v.get("injury_status")}
            for k, v in players.items() if v.get("injury_status")]
    if not rows:
        return None
    ids = (pl.read_parquet(DATA / "processed" / "player_seasons.parquet")
             .select("player_id", pl.col("sleeper_id").cast(pl.Utf8)).drop_nulls().unique("sleeper_id"))
    return pl.DataFrame(rows).join(ids, on="sleeper_id").select("player_id", "sl_status")



# QB change: per point of the starting QB's preseason projection, a pass catcher / RB gains this share of his
# own PPG. 2019-2025, players who had 2+ starting QBs in a season (QB quality = preseason projection, so the
# game itself can't leak in): WR 0.0069 (+-0.0022), TE 0.0072 (+-0.0038), RB 0.0045 (+-0.0026). E.g. a 12-PPG
# WR whose QB goes from a 20 to a 14 projected-PPG starter: -0.5 PPG. Likely understated (projections are noisy).
QB_CHANGE = {"RB": 0.0045, "WR": 0.0069, "TE": 0.0072}


def qb_quality(season: int, league: str) -> pl.DataFrame:
    p = projections(season, league).select("player_id", pl.col(f"pred_ppg_{league}").alias("q"))
    return p


def qb_change(d: pl.DataFrame, season: int, thru: int, league: str, col: str) -> pl.DataFrame:
    """Adjust RB / WR / TE for a different starting QB from here on (trade, injury, benching) vs the QBs
    they played with so far. Current starter = the team's active QB with the best ROS PPG."""
    q = qb_quality(season, league)
    low = float(q["q"].quantile(0.1))
    w = pl.read_parquet(DATA / "processed" / "player_weeks.parquet").filter(pl.col("season") == season, pl.col("week") <= thru)
    st = (w.filter(pl.col("position") == "QB").sort("attempts", descending=True, nulls_last=True)
           .group_by("team", "week").first().select("team", "week", pl.col("player_id").alias("qb")))
    st = st.join(q, left_on="qb", right_on="player_id", how="left").with_columns(pl.col("q").fill_null(low))
    had = (w.filter(pl.col("position").is_in(list(QB_CHANGE))).join(st, on=["team", "week"])
             .group_by("player_id").agg(pl.col("q").mean().alias("q_had")))
    # starter from here on: best two healthy-enough QBs by ROS PPG; the top one's share of the remaining games
    # is what he's expected to be available for (a dated injury return splits the games)
    left = team_games(season, thru + 1, 17, unplayed_before=True)
    qbs = (d.filter(pl.col("position") == "QB", ~pl.col("status").is_in(["out_long", "season", "gone", "practice"]))
             .join(q, on="player_id", how="left").with_columns(pl.col("q").fill_null(low))
             .sort([pl.col("status") == "backup", pl.col("dc_rank").fill_null(99), pl.col(col)], descending=[False, False, True]))
    top = qbs.group_by("team").agg(pl.col("q").first().alias("q1"), pl.col("q").get(1, null_on_oob=True).alias("q2"),
                                   pl.col("player_display_name").first().alias("qb_now"),
                                   (pl.col("missed_reg").first() + pl.col("missed_po").first()).alias("miss1"))
    top = top.with_columns(pl.col("team").replace_strict(left, default=1, return_dtype=pl.Int64).alias("gl"))
    sh = (1 - pl.col("miss1") / pl.col("gl").clip(1)).clip(0, 1)
    now = top.select("team", "qb_now", (sh * pl.col("q1") + (1 - sh) * pl.col("q2").fill_null(low)).alias("q_now"))
    c = pl.col("position").replace_strict(QB_CHANGE, default=0.0, return_dtype=pl.Float64)
    d = d.join(had, on="player_id", how="left").join(now, on="team", how="left")
    adj = (c * pl.col(col) * (pl.col("q_now") - pl.col("q_had"))).fill_null(0)
    return d.with_columns(adj.alias("qb_adj")).with_columns((pl.col(col) + pl.col("qb_adj")).alias(col))


def depth(season: int) -> pl.DataFrame:
    """Latest ESPN depth chart (daily snapshots): player_id, depth team, position, rank (1 = starter), as-of."""
    d = data.depth_charts([season])[0]
    d = d.filter(pl.col("dt") == d["dt"].max(), pl.col("pos_abb").is_in(["QB", "RB", "TE"]), pl.col("gsis_id").is_not_null())
    return (d.group_by("gsis_id").agg(pl.col("team").first().alias("dc_team"), pl.col("pos_abb").first().alias("dc_pos"),
                                      pl.col("pos_rank").min().alias("dc_rank"), pl.col("dt").first().alias("dc_asof"))
             .rename({"gsis_id": "player_id"}))


OUT_ST = ["out_long", "out_week", "season", "gone", "practice", "dated"]


# WR depth-chart moves (scripts/study_new_starter.py, 2021-25, cut-off weeks 8-10, burbs):
#  - newly a starter (top 3 on the depth chart now, a starter in at most a third of the weeks so far), n=119:
#    the model ran 0.9 pts low on average. They split: a quarter finished under 4.3 PPG, a quarter over 10.7, so
#    a floor (e.g. "starters score 8") made the average miss worse (3.05 -> 3.40). A small flat bump removes most
#    of the bias without hurting the miss (3.03 at +0.5).
#  - lost the job (a starter in two thirds of the weeks, not now, healthy), n=47: the model ran 1.3 high;
#    x0.85 cut the miss from 3.16 to about 2.9, better in 4 of 5 seasons.
WR_NEW_STARTER_BUMP = 0.75
WR_DEMOTED_MULT = 0.85
# RBs, same study: new lead backs (n=52) needed nothing: model 10.1 vs 10.0 actual, the role trend already
# catches them, so they get a note only. RBs who lost the lead job while healthy (n=21, small): model ran 1.2
# high; x0.85 took the miss from 2.24 to 1.88 and removed the bias.
RB_DEMOTED_MULT = 0.85


def wr_depth_moves(d: pl.DataFrame, season: int, thru: int, col: str) -> pl.DataFrame:
    dc = data.depth_charts([season])[0]
    if "dt" not in dc.columns:
        return d.with_columns(pl.lit("").alias("depth_note"))
    nst = pl.col("pos_abb").replace_strict({"WR": 3, "RB": 1}, return_dtype=pl.Int32)
    dc = dc.filter(pl.col("pos_abb").is_in(["WR", "RB"]), pl.col("gsis_id").is_not_null()).with_columns(pl.col("dt").str.slice(0, 10).alias("day"))
    sch = data.schedules([season]).filter(pl.col("game_type") == "REG", pl.col("week") <= thru).group_by("week").agg(pl.col("gameday").min())
    days = sorted(dc["day"].unique().to_list())
    import datetime as dt
    wk = []
    for w, gd in sch.iter_rows():
        prev = [x for x in days if x <= (dt.date.fromisoformat(gd) - dt.timedelta(days=2)).isoformat()]
        if prev:
            s_ = dc.filter(pl.col("day") == prev[-1])
            wk.append(s_.filter(pl.col("dt") == s_["dt"].max(), pl.col("pos_rank") <= nst).select(pl.lit(w).alias("week"), "gsis_id").unique())
    if not wk:
        return d.with_columns(pl.lit("").alias("depth_note"))
    share = pl.concat(wk).group_by("gsis_id").agg((pl.col("week").n_unique() / thru).alias("st_share"))
    now = dc.filter(pl.col("dt") == dc["dt"].max(), pl.col("pos_rank") <= nst).select("gsis_id", pl.col("team").alias("st_team")).unique("gsis_id")
    d = (d.join(share, left_on="player_id", right_on="gsis_id", how="left").join(now, left_on="player_id", right_on="gsis_id", how="left")
          .with_columns(pl.col("st_share").fill_null(0.0)))
    wr, ok = pl.col("position") == "WR", pl.col("status").is_in(["active", "no_games"])
    rb = pl.col("position") == "RB"
    new = wr & ok & pl.col("st_team").is_not_null() & (pl.col("st_share") <= 1 / 3)
    dem = wr & ok & pl.col("st_team").is_null() & (pl.col("st_share") > 2 / 3)
    rnew = rb & ok & pl.col("st_team").is_not_null() & (pl.col("st_share") <= 1 / 3)
    rdem = rb & ok & pl.col("st_team").is_null() & (pl.col("st_share") > 2 / 3)
    return d.with_columns(
        pl.when(new).then(pl.col(col) + WR_NEW_STARTER_BUMP).when(dem).then(pl.col(col) * WR_DEMOTED_MULT)
          .when(rdem).then(pl.col(col) * RB_DEMOTED_MULT).otherwise(pl.col(col)).alias(col),
        pl.when(new).then(pl.lit("New starter on depth chart")).when(dem).then(pl.lit("Dropped from starters on depth chart"))
          .when(rnew).then(pl.lit("New lead back on depth chart")).when(rdem).then(pl.lit("No longer lead back on depth chart"))
          .otherwise(pl.lit("")).alias("depth_note")).drop("st_share", "st_team")


def short_fill_ins(d: pl.DataFrame, season: int, thru: int, league: str, col: str) -> pl.DataFrame:
    """Lead RB / TE out for a week or two (dated return, fewer than 3 games; longer absences are handled by
    vacated work): the next man on the depth chart gets his own fill-in bump (fillin.bumps: his history of
    games the lead missed, shrunk to the league average, RB about +5.7) for those games only. ROS PPG is a
    per-game average, so the bump is spread over his remaining games; `fill_adj` holds the amount so the
    weekly file, which applies the full bump in the week itself, can take it back out."""
    from . import fillin
    fb, pr = fillin.bumps(league, season, thru)
    left = pl.col("games_left_reg") + pl.col("games_left_po")
    outs = []
    for p_ in ("RB", "TE"):
        ps = d.filter(pl.col("position") == p_, pl.col("status") != "gone")
        lead = (ps.sort([pl.col("dc_rank").fill_null(99), pl.col(col)], descending=[False, True])
                  .group_by("team", maintain_order=True).first()
                  .filter(pl.col("status").is_in(["dated", "out_week"]),
                          (pl.col("missed_reg") + pl.col("missed_po")).is_between(1, 2))
                  .select("team", pl.col("player_id").alias("lead_id"), pl.col("player_display_name").alias("lead_name"),
                          (pl.col("missed_reg") + pl.col("missed_po")).alias("lead_miss"), pl.col("n").alias("lead_n"),
                          pl.col("news").str.extract(r"est\. return (\w{3} \d+)").alias("lead_ret")))
        nxt = (ps.join(lead, on="team").filter(pl.col("player_id") != pl.col("lead_id"), pl.col("status").is_in(["active", "no_games"]))
                 .sort([pl.col("dc_rank").fill_null(99), pl.col(col)], descending=[False, True])
                 .group_by("team", maintain_order=True).first()
                 .join(fb.select("player_id", "bump"), on="player_id", how="left")
                 .with_columns(pl.col("bump").fill_null(pr.get(p_, 0.0)))
                 .with_columns((pl.col("bump") * pl.col("lead_miss") / left.clip(1)).alias("fill_adj"))
                 .select("player_id", "fill_adj", "lead_name", "lead_ret"))
        outs.append(nxt)
    d = d.join(pl.concat(outs), on="player_id", how="left").with_columns(pl.col("fill_adj").fill_null(0.0))
    return d.with_columns(
        (pl.col(col) + pl.col("fill_adj")).alias(col),
        pl.when(pl.col("lead_name").is_not_null())
          .then(pl.concat_str([pl.col("depth_note"), pl.lit(" Filling in while "), pl.col("lead_name"), pl.lit(" is out (back "),
                               pl.col("lead_ret").fill_null("soon"), pl.lit(")")]).str.strip_chars())
          .otherwise(pl.col("depth_note")).alias("depth_note")).drop("lead_name", "lead_ret")


def fill_in_starters(d: pl.DataFrame, season: int, col: str) -> pl.DataFrame:
    """A QB who is starting only while the regular starter is hurt keeps the job just until that return.
    Regular starter = the QB listed first on the team's depth chart on the most days this season. If he is out
    with a return date and is still on the team, the current
    starter gets fill_reg / fill_po (games he covers) and a note. A benching (regular starter healthy) or a
    season-ending injury is left alone."""
    dc = data.depth_charts([season])[0].filter(pl.col("pos_abb") == "QB", pl.col("pos_rank") == 1, pl.col("gsis_id").is_not_null())
    inc = (dc.group_by("team", "gsis_id").agg(pl.col("dt").n_unique().alias("days"))
             .sort("days", descending=True).group_by("team", maintain_order=True).first()
             .select("team", pl.col("gsis_id").alias("inc_id")))
    q = d.filter(pl.col("position") == "QB")
    incq = (q.join(inc, left_on=["team", "player_id"], right_on=["team", "inc_id"])
             .filter(pl.col("status").is_in(["dated", "out_week"]), pl.col("missed_reg") + pl.col("missed_po") > 0)
             .select("team", pl.col("player_display_name").alias("inc_name"), pl.col(col).alias("inc_ppg"),
                     pl.col("missed_reg").alias("fill_reg"), pl.col("missed_po").alias("fill_po"),
                     pl.col("news").str.extract(r"est\. return (\w{3} \d+)").alias("inc_ret")))
    cur = (q.filter(pl.col("status") == "active").join(incq, on="team")
            .select("player_id", "inc_name", "fill_reg", "fill_po", "inc_ret"))
    d = d.join(cur, on="player_id", how="left")
    return d.with_columns(
        pl.when(pl.col("inc_name").is_not_null())
          .then(pl.concat_str([pl.lit("Fill-in starter, until "), pl.col("inc_name"), pl.lit(" returns, est. return "),
                               pl.col("inc_ret").fill_null("unknown")]))
          .otherwise(pl.col("news")).alias("news")).drop("inc_name", "inc_ret")


def apply_depth(d: pl.DataFrame, season: int, col: str | None = None) -> pl.DataFrame:
    """QBs: the depth-chart starter plays (even with no games yet, e.g. a rookie just named starter); other
    QBs are backups (status 'backup', ~10% of games) unless every QB ahead of them is out. A depth-chart
    starter the model has never seen (no projection, no games) is added at a low-end projection."""
    dc = depth(season)
    if col:
        nm = data.depth_charts([season])[0]
        nm = nm.filter(pl.col("dt") == nm["dt"].max()).group_by("gsis_id").agg(pl.col("player_name").first())
        miss = (dc.filter(pl.col("dc_rank") == 1, pl.col("dc_pos") == "QB").join(d.select("player_id"), on="player_id", how="anti")
                  .join(nm, left_on="player_id", right_on="gsis_id", how="left"))
        if miss.height:
            low = float(d.filter(pl.col("position") == "QB", pl.col("status") == "active", pl.col("n") >= 4)[col].quantile(0.1))
            add = miss.select("player_id", pl.col("player_name").alias("player_display_name"), pl.lit("QB").alias("position"),
                              pl.col("dc_team").alias("team"), pl.lit(low).alias(col), pl.lit("active").alias("status"),
                              pl.lit(0).cast(pl.UInt32).alias("n"), pl.lit(0).alias("missed_reg"), pl.lit(0).alias("missed_po"),
                              pl.lit(0.0).alias("sched_adj"), pl.lit(low).alias("prior"), pl.lit(0.0).alias("role_trend"),
                              pl.lit(0.0).alias("luck_pg"), pl.lit(False).alias("rookie"),
                              pl.lit("New starter per depth chart: no stats or projection yet (low-end QB estimate)").alias("news"))
            d = pl.concat([d, add], how="diagonal_relaxed")
    d = d.join(dc.select("player_id", "dc_rank"), on="player_id", how="left")
    qb = d.filter(pl.col("position") == "QB", pl.col("dc_rank").is_not_null())
    healthy_top = (qb.filter(~pl.col("status").is_in(OUT_ST)).sort("dc_rank").group_by("team").first()
                     .select("team", pl.col("player_id").alias("qb_start_id")))
    d = d.join(healthy_top, on="team", how="left")
    isqb = pl.col("position") == "QB"
    return d.with_columns(
        pl.when(isqb & (pl.col("player_id") == pl.col("qb_start_id")) & pl.col("status").is_in(["no_games", "active"]))
          .then(pl.lit("active"))
          .when(isqb & pl.col("qb_start_id").is_not_null() & (pl.col("player_id") != pl.col("qb_start_id"))
                & ~pl.col("status").is_in(OUT_ST))
          .then(pl.lit("backup"))
          .otherwise(pl.col("status")).alias("status")).drop("qb_start_id")
