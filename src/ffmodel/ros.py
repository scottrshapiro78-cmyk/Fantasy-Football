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
    o = data.opportunity([season]).filter(pl.col("player_id").is_not_null()).select(
        "player_id", pl.col("week").cast(pl.Int32),
        (pl.col("total_fantasy_points_exp").cast(pl.Float64) - pl.col("total_fantasy_points").cast(pl.Float64))
        .alias("luck_gap"), pl.col("total_fantasy_points_exp").cast(pl.Float64).alias("xfp"))
    w = (w.select("player_id", "player_display_name", "position", "team", pl.col("week").cast(pl.Int32),
                  "fpts_burbs", "fpts_cbs")
          .join(o.group_by("player_id", "week").agg(pl.col("luck_gap").sum(), pl.col("xfp").sum()),
                on=["player_id", "week"], how="left")
          # a game played with no touches is zero usage, not missing
          .with_columns(pl.when(pl.col("position").is_in(["QB", "RB", "WR", "TE"]))
                          .then(pl.col("xfp").fill_null(0)).otherwise(pl.col("xfp")).alias("xfp")))
    d = (dst_weeks().filter(pl.col("season") == season)
         .select(pl.col("team").alias("player_id"), (pl.col("team") + " DEF").alias("player_display_name"),
                 pl.lit("DEF").alias("position"), "team", pl.col("week").cast(pl.Int32), "fpts_burbs", "fpts_cbs",
                 pl.lit(None, dtype=pl.Float64).alias("luck_gap"), pl.lit(None, dtype=pl.Float64).alias("xfp")))
    return pl.concat([w, d], how="vertical_relaxed")


def to_date(season: int, thru: int) -> pl.DataFrame:
    w = _weekly(season).filter(pl.col("week") <= thru)
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


def fit(thru: int = 8) -> dict:
    """Fit k (prior weight, in games) and u (usage-luck weight) per position and league, and
    the chance of playing each remaining game by status. Saved to data/processed/ros_fit.json."""
    cache = DATA / "processed" / f"ros_history_wk{thru}.parquet"
    h = pl.read_parquet(cache) if cache.exists() else _history(thru=thru)
    h.write_parquet(cache)
    if "role_trend" not in h.columns:  # cache predates the role signal: refresh the season-to-date part
        td = pl.concat([to_date(s_, thru).with_columns(pl.lit(s_).cast(h["season"].dtype).alias("season"))
                        for s_ in h["season"].unique().to_list()])
        keep = [c for c in h.columns if c not in td.columns or c in ("player_id", "season")]
        h = h.select(keep).join(td, on=["player_id", "season"], how="left").with_columns(pl.col("n").fill_null(0))
        h.write_parquet(cache)
    if "sos_left_burbs" not in h.columns:
        parts = []
        for s_ in h["season"].unique().to_list():
            a = sos(s_, thru, "burbs").rename({"sos_faced": "sos_faced_burbs", "sos_left": "sos_left_burbs"})
            b = sos(s_, thru, "cbs").rename({"sos_faced": "sos_faced_cbs", "sos_left": "sos_left_cbs"})
            parts.append(a.join(b, on="player_id", how="full", coalesce=True)
                          .with_columns(pl.lit(s_).cast(h["season"].dtype).alias("season")))
        h = h.join(pl.concat(parts), on=["player_id", "season"], how="left")
        h.write_parquet(cache)
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

    st = status(season, thru)
    d = d.join(st, on="player_id", how="left").with_columns(
        pl.when(pl.col("position") == "DEF").then(pl.col("player_id"))
          .otherwise(pl.coalesce("roster_team", "team_now", "team")).alias("team"),
        pl.when(pl.col("position") == "DEF").then(pl.lit("active"))
          .otherwise(pl.col("status").fill_null("active")).alias("status"))
    d = d.with_columns(pl.when((pl.col("status") == "active") & (pl.col("n") == 0) & (pl.col("position") != "DEF"))
                         .then(pl.lit("no_games")).otherwise(pl.col("status")).alias("status"))
    manual = DATA / "input" / f"status_{season}.csv"  # news the data doesn't have yet: Player, Status
    if manual.exists():
        m = pl.read_csv(manual).select(pl.col("Player").alias("player_display_name"), pl.col("Status").alias("manual"))
        d = d.join(m, on="player_display_name", how="left").with_columns(
            pl.coalesce("manual", "status").alias("status")).drop("manual")
    d = d.with_columns(pl.lit(0).alias("missed_reg"), pl.lit(0).alias("missed_po"), pl.lit("").alias("news"))
    d = apply_news(d, season, thru)
    ratio = fitp.get("ppg_ratio", {})
    d = d.with_columns((pl.col(col) * pl.col("status").replace_strict(ratio, default=1.0, return_dtype=pl.Float64)).alias(col))
    d = d.filter(pl.col("status") != "gone")
    reg, po = team_games(season, thru + 1, 14, unplayed_before=True), team_games(season, 15, 17)
    play = {**play, "dated": play["active"], "season": 0.0}
    rate = (pl.when(pl.col("position").is_in(["K", "DEF"]) & ~pl.col("status").is_in(["season", "dated"])).then(
                pl.when(pl.col("status") == "active").then(0.97).otherwise(play["out_long"]))
            .otherwise(pl.col("status").replace_strict(play, default=play["active"], return_dtype=pl.Float64)))
    d = d.with_columns(
        pl.col("team").replace_strict(reg, default=0, return_dtype=pl.Int64).alias("games_left_reg"),
        pl.col("team").replace_strict(po, default=0, return_dtype=pl.Int64).alias("games_left_po"),
        rate.alias("play_rate"))
    return d.with_columns(
        ((pl.col("games_left_reg") - pl.col("missed_reg")).clip(0) * pl.col("play_rate")).alias("exp_reg"),
        ((pl.col("games_left_po") - pl.col("missed_po")).clip(0) * pl.col("play_rate")).alias("exp_po"))


def _remaining_opps(season: int, thru: int, end: int = 17) -> pl.DataFrame:
    s = data.schedules([season]).filter(pl.col("game_type") == "REG")
    s = s.filter(pl.col("week").is_between(thru + 1, end) | ((pl.col("week") <= thru) & pl.col("home_score").is_null()))
    return pl.concat([s.select(pl.col("home_team").alias("team"), pl.col("away_team").alias("opp")),
                      s.select(pl.col("away_team").alias("team"), pl.col("home_team").alias("opp"))])


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

    def split(c):
        return [pl.col(c).sort_by("week").tail(3).median().alias(f"{c}_l3"),
                pl.col(c).sort_by("week").head(pl.len() - 3).median().alias(f"{c}_pre")]
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


def vacated(season: int, thru: int, league: str) -> pl.DataFrame:
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
    gone = (p.filter(pl.col("status").is_in(["out_long", "gone"]), pl.col("g") >= 2)
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
      not on the list -> healthy, even if last week's report or roster had him out."""
    import datetime as dt
    from .market import _norm
    path = DATA / "input" / f"news_{season}.tsv"
    if not path.exists():
        return d
    news = pl.read_csv(path, separator="\t", quote_char=None).with_columns(pl.col("Note").fill_null(""))
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
    # games missed before the return date, split regular season (wk <= 14) / playoffs
    miss = (d.filter(pl.col("ret").is_not_null()).select("player_id", "team", "ret")
              .join(games, on="team").filter(pl.col("gameday") < pl.col("ret"))
              .group_by("player_id").agg((pl.col("week") <= 14).sum().alias("m_reg"),
                                         (pl.col("week") > 14).sum().alias("m_po")))
    d = d.join(miss, on="player_id", how="left")
    skill = pl.col("position").is_in(["QB", "RB", "WR", "TE", "K"])
    out = pl.col("Status").is_in(["Out", "Injured Reserve"])
    season_over = out & (pl.col("ret") >= dt.date(season + 1, 1, 1))
    dated = out & ~season_over & (pl.col("ret") > pl.col("next_game"))
    d = d.with_columns(
        pl.when(~skill).then(pl.col("status"))
          .when(pl.col("Status").is_null() & pl.col("status").is_in(["out_week", "out_long"])).then(pl.lit("active"))
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
