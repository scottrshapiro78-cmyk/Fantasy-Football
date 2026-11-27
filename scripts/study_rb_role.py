"""Nov 27 2026: role signals for the weekly RB (and WR/TE) model. Scott's ask: close the gap on "model high" RB calls,
which are right only about a third of the time, without using expert rankings as inputs (experts only grade).

Every feature for week t uses only what is known before that week's games (games <= t-1, that week's practice report,
that week's depth chart), from free nflverse data:
  snaps      offense snap share: season so far, last 3 games, last game          (nflverse snap_counts)
  work       share of team carries, of team targets, last 3 vs season             (player_stats)
  red zone   share of the team's carries + targets inside the 10, and carries inside the 5, season and last 3 (pbp)
  committee  how many same-position teammates had 25%+ of the RB work over the last 3 games
  depth      depth-chart rank this week (1 = lead back), and share of earlier weeks he was the lead
  practice   own Limited / Full, and same-position teammates' Questionable / DNP this week weighted by their PPG
  return     1st / 2nd game back after missing games (ramp-up)

Grading is the consensus-miss method (scripts/study_consensus_misses.py): leave-one-season-out over 2021-2026,
weeks 5-17 (2026 through week 11), Burbs scoring, fantasy-relevant pool, and pairs where the model and FantasyPros
disagree. Rule 5: adopt only if the average miss drops 2%+ vs production; 2025 is also checked on its own.

PYTHONPATH=src python3 scripts/study_consensus_misses.py build      # once
PYTHONPATH=src python3 scripts/study_rb_role.py [RB|WR|TE ...]
"""
import sys

import numpy as np
import polars as pl

from ffmodel import data
from ffmodel.config import DATA

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import study_consensus_misses as cm  # noqa: E402

YRS = list(range(2021, 2027))
CACHE = DATA / "processed" / "role_study.parquet"


def _pbp_rz() -> pl.DataFrame:
    """Per player-game: carries + targets inside the opponent's 10, carries inside the 5, and team totals."""
    out = []
    for s in YRS:
        path = DATA / "raw" / "pbp_rz" / f"{s}.parquet"
        if not path.exists():
            import nflreadpy as nfl
            p = nfl.load_pbp([s]).filter(pl.col("season_type") == "REG", pl.col("yardline_100") <= 10,
                                         pl.col("play_type").is_in(["run", "pass"]))
            p = p.select("season", pl.col("week").cast(pl.Int32), pl.col("posteam").alias("team"), "yardline_100",
                         pl.when(pl.col("play_type") == "run").then(pl.col("rusher_player_id")).otherwise(pl.col("receiver_player_id")).alias("player_id"),
                         (pl.col("play_type") == "run").alias("run"))
            path.parent.mkdir(parents=True, exist_ok=True)
            p.write_parquet(path)
        out.append(pl.read_parquet(path))
    p = pl.concat(out)
    tm = p.group_by("season", "week", "team").agg(pl.len().alias("t_rz"), (pl.col("run") & (pl.col("yardline_100") <= 5)).sum().alias("t_gl"))
    pl_ = p.drop_nulls("player_id").group_by("season", "week", "team", "player_id").agg(
        pl.len().alias("rz"), (pl.col("run") & (pl.col("yardline_100") <= 5)).sum().alias("gl"))
    return pl_, tm


def _snaps() -> pl.DataFrame:
    ps = pl.read_parquet(DATA / "processed" / "player_seasons.parquet").select("player_id", "pfr_id").drop_nulls().unique("pfr_id")
    return (data.snap_counts(YRS).filter(pl.col("game_type") == "REG").join(ps, left_on="pfr_player_id", right_on="pfr_id")
            .select("season", pl.col("week").cast(pl.Int32), "player_id", pl.col("offense_pct").cast(pl.Float64).alias("snap"))
            .unique(["season", "week", "player_id"]))


def _depth() -> pl.DataFrame:
    """Depth rank at the player's position for each (season, week): weekly charts through 2024; for 2025+ the last
    daily ESPN snapshot at least 2 days before that week's first game."""
    import datetime as dt
    rows = []
    for s in YRS:
        dc = data.depth_charts([s])[0]
        if "dt" not in dc.columns:
            x = (dc.filter(pl.col("game_type") == "REG", pl.col("formation") == "Offense", pl.col("position").is_in(["RB", "WR", "TE"]),
                           pl.col("depth_position").is_in(["RB", "HB", "WR", "LWR", "RWR", "SWR", "TE"]))
                 .with_columns(pl.col("depth_team").cast(pl.Int32, strict=False).alias("rk"))
                 .group_by(pl.col("week").cast(pl.Int32), pl.col("gsis_id").alias("player_id"))
                 .agg(pl.col("rk").min()))
            rows.append(x.with_columns(pl.lit(s).alias("season")))
            continue
        dc = dc.filter(pl.col("pos_abb").is_in(["RB", "WR", "TE"]), pl.col("gsis_id").is_not_null()).with_columns(pl.col("dt").str.slice(0, 10).alias("day"))
        days = sorted(dc["day"].unique().to_list())
        sch = data.schedules([s]).filter(pl.col("game_type") == "REG").group_by("week").agg(pl.col("gameday").min())
        for w, gd in sch.iter_rows():
            prev = [d for d in days if d <= (dt.date.fromisoformat(gd) - dt.timedelta(days=2)).isoformat()]
            if not prev:
                continue
            snap = dc.filter(pl.col("day") == prev[-1])
            snap = snap.filter(pl.col("dt") == snap["dt"].max())
            rows.append(snap.group_by(pl.col("gsis_id").alias("player_id")).agg(pl.col("pos_rank").min().cast(pl.Int32).alias("rk"))
                        .with_columns(pl.lit(int(w)).cast(pl.Int32).alias("week"), pl.lit(s).alias("season")))
    return pl.concat(rows, how="diagonal_relaxed").select("season", "week", "player_id", pl.col("rk").cast(pl.Int32))


def build_role() -> pl.DataFrame:
    pw = (pl.read_parquet(DATA / "processed" / "player_weeks.parquet").filter(pl.col("season").is_in(YRS))
          .with_columns(pl.col("week").cast(pl.Int32), pl.col("carries").fill_null(0).alias("car"), pl.col("targets").fill_null(0).alias("tgt")))
    team = pw.group_by("season", "week", "team").agg(pl.col("car").sum().alias("t_car"), pl.col("tgt").sum().alias("t_tgt"))
    rz, trz = _pbp_rz()
    g = (pw.join(team, on=["season", "week", "team"]).join(_snaps(), on=["season", "week", "player_id"], how="left")
         .join(rz.drop("team"), on=["season", "week", "player_id"], how="left").join(trz, on=["season", "week", "team"], how="left")
         .with_columns((pl.col("car") / pl.col("t_car")).alias("car_sh"), (pl.col("tgt") / pl.col("t_tgt")).alias("tgt_sh"),
                       ((pl.col("car") + pl.col("tgt")) / (pl.col("t_car") + pl.col("t_tgt"))).alias("work_sh"),
                       (pl.col("rz").fill_null(0) / pl.col("t_rz")).alias("rz_sh"),
                       (pl.col("gl").fill_null(0) / pl.col("t_gl").clip(1, None)).alias("gl_sh"),
                       pl.col("t_gl").fill_null(0))
         .select("season", "week", "player_id", "team", "position", "snap", "car_sh", "tgt_sh", "work_sh", "rz_sh", "gl_sh", "t_gl", "fpts_burbs"))
    dep = _depth()
    inj = (data.injuries(YRS).filter(pl.col("game_type") == "REG")
           .select("season", pl.col("week").cast(pl.Int32), pl.col("gsis_id").alias("player_id"), "report_status", "practice_status")
           .unique(["season", "week", "player_id"]))
    teamwk = pw.group_by("season", "team").agg(pl.col("week").unique().sort().alias("tw"))
    out = []
    for s in YRS:
        gs = g.filter(pl.col("season") == s)
        for t in range(5, 18):
            h = gs.filter(pl.col("week") < t).sort("week")
            if h.is_empty():
                continue
            ok = pl.col("snap").is_not_null()
            f = h.group_by("player_id").agg(
                pl.col("team").last(), pl.col("position").last(), pl.len().alias("gp"), pl.col("week").max().alias("lastwk"),
                pl.col("week").sort().alias("wks"), pl.col("fpts_burbs").mean().alias("ppg"),
                pl.col("snap").filter(ok).mean().alias("snap_pg"), pl.col("snap").filter(ok).tail(3).mean().alias("snap_l3"),
                pl.col("snap").filter(ok).last().alias("snap_l1"),
                *[pl.col(c).mean().alias(f"{c}_pg") for c in ["car_sh", "tgt_sh", "work_sh", "rz_sh"]],
                *[pl.col(c).tail(3).mean().alias(f"{c}_l3") for c in ["car_sh", "tgt_sh", "work_sh", "rz_sh"]],
                (pl.col("gl_sh") * pl.col("t_gl")).sum().alias("gl_n"), pl.col("t_gl").sum().alias("gl_t"),
                (pl.col("gl_sh") * pl.col("t_gl")).tail(3).sum().alias("gl_n3"), pl.col("t_gl").tail(3).sum().alias("gl_t3"))
            # committee: same-position teammates (on his current team) with 25%+ RB work share over the last 3
            com = (f.filter(pl.col("position") == "RB", pl.col("lastwk") >= t - 2, pl.col("work_sh_l3") >= 0.12)
                   .group_by("team").agg(pl.len().alias("n_backs"), pl.col("work_sh_l3").sum().alias("rb_work_l3")))
            # depth: this week's rank, share of earlier weeks as the lead
            dw = dep.filter(pl.col("season") == s)
            now = dw.filter(pl.col("week") == t).select("player_id", pl.col("rk").alias("rk_now"))
            lead = dw.filter(pl.col("week") < t).group_by("player_id").agg((pl.col("rk") == 1).mean().alias("lead_pre"))
            it = inj.filter(pl.col("season") == s, pl.col("week") == t)
            f = (f.join(com, on="team", how="left").join(now, on="player_id", how="left").join(lead, on="player_id", how="left")
                 .join(it, on="player_id", how="left"))
            # same-position teammates who may sit this week (Q / Doubtful / Out on this week's report), weighted by PPG
            mq = (f.filter(pl.col("gp") >= 2, pl.col("lastwk") >= t - 2)
                  .with_columns((pl.col("report_status").is_in(["Questionable"]).fill_null(False) * 0.25
                                 + pl.col("report_status").is_in(["Doubtful", "Out"]).fill_null(False) * 1.0
                                 + (pl.col("report_status").is_null() & pl.col("practice_status").str.contains("Did Not").fill_null(False)) * 0.1)
                                .alias("sitp"))
                  .select("team", "position", "player_id", (pl.col("sitp") * pl.col("ppg")).alias("mq"), (pl.col("sitp") * pl.col("work_sh_l3").fill_null(0)).alias("mqw")))
            f = f.join(mq.group_by("team", "position").agg(pl.col("mq").sum().alias("mq_tot"), pl.col("mqw").sum().alias("mqw_tot")),
                       on=["team", "position"], how="left").join(mq.select("player_id", pl.col("mq").alias("mq_me"), pl.col("mqw").alias("mqw_me")),
                                                                 on="player_id", how="left")
            tw = teamwk.filter(pl.col("season") == s).select("team", pl.col("tw").list.eval(pl.element().filter(pl.element() < t)).alias("tw"))
            f = f.join(tw, on="team", how="left")
            out.append(f.with_columns(pl.lit(s).alias("season"), pl.lit(t).cast(pl.Int32).alias("week")))
        print("role", s, flush=True)
    r = pl.concat(out, how="diagonal_relaxed")
    # games back after missing time: team games since his first game back, counting only if he missed 1+ team game before
    r = r.with_columns(
        pl.struct("wks", "tw").map_elements(_ret_g, return_dtype=pl.Int32).alias("ret_g"))
    return r.drop("wks", "tw")


def _ret_g(x) -> int:
    """1 = first game back this week, 2 = second, 0 otherwise (missed 1+ of his team's games, then played)."""
    wks, tw = x["wks"] or [], x["tw"] or []
    if not wks or not tw:
        return 0
    played = set(wks)
    miss = [w for w in tw if w not in played and w > min(wks)]
    if not miss:
        return 0
    after = [w for w in wks if w > max(miss)]
    return {0: 1, 1: 2}.get(len(after), 0)


def frame(pos: list[str]):
    import pandas as pd
    if not CACHE.exists():
        build_role().write_parquet(CACHE)
    r = pl.read_parquet(CACHE).drop("team", "position").to_pandas()
    q = cm.frame()
    q = q[q.position.isin(pos)].merge(r, on=["player_id", "season", "week"], how="left")
    for c in ["snap_pg", "snap_l3", "snap_l1"]:
        q[c] = q[c].fillna(q.groupby("position")[c].transform("median"))
    z = lambda c: q[c].fillna(0)
    q["d_snap"] = z("snap_l3") - z("snap_pg")
    q["d_snap1"] = z("snap_l1") - z("snap_pg")
    for c in ["car_sh", "tgt_sh", "work_sh", "rz_sh"]:
        q[f"d_{c}"] = z(f"{c}_l3") - z(f"{c}_pg")
    q["gl_sh"] = (q.gl_n.fillna(0) + 0.3) / (q.gl_t.fillna(0) + 1.0)  # shrunk goal-line carry share
    q["d_gl"] = (q.gl_n3.fillna(0) + 0.3) / (q.gl_t3.fillna(0) + 1.0) - q.gl_sh
    q["rk_now"] = q.rk_now.fillna(3).clip(1, 3)
    q["lead_now"] = (q.rk_now == 1).astype(int)
    q["lead_pre"] = q.lead_pre.fillna(0)
    q["lead_new"] = ((q.rk_now == 1) & (q.lead_pre <= 1 / 3)).astype(int)
    q["lead_lost"] = ((q.rk_now > 1) & (q.lead_pre > 2 / 3)).astype(int)
    q["committee"] = (q.n_backs.fillna(1) >= 2).astype(int)
    q["lim"] = q.practice_status.fillna("").str.contains("Limited").astype(int)
    q["mate_sit"] = (q.mq_tot.fillna(0) - q.mq_me.fillna(0)).clip(0, None)
    q["mate_sitw"] = (q.mqw_tot.fillna(0) - q.mqw_me.fillna(0)).clip(0, None)
    q["ret1"], q["ret2"] = (q.ret_g == 1).astype(int), (q.ret_g == 2).astype(int)
    # scaled by the base so a role change is worth more for a better player
    q["b_dsnap"] = q.base * q.d_snap
    q["b_dwork"] = q.base * q.d_work_sh
    return q


BASE = cm.FEATS + ["dnp"]
ROLE = {
    "+snap trend": ["d_snap", "d_snap1"],
    "+work trend": ["d_car_sh", "d_tgt_sh"],
    "+red zone": ["rz_sh_pg", "d_rz_sh", "gl_sh", "d_gl"],
    "+committee/depth": ["committee", "lead_new", "lead_lost", "lead_now"],
    "+practice/teammates": ["lim", "mate_sit", "mate_sitw", "outppg"],
    "+ramp-up": ["ret1", "ret2"],
    "+scaled trends": ["b_dsnap", "b_dwork"],
}


def grade(pos: str) -> None:
    import pandas as pd
    q = frame([pos])
    for c in sum(ROLE.values(), []):
        q[c] = q[c].fillna(0)
    m0 = (q["prod"] - q.y).abs()
    allrole = BASE + sum(ROLE.values(), [])
    specs = {"refit (Vegas+shrink+injury)": BASE, **{k: BASE + v for k, v in ROLE.items()}, "all role signals": allrole}
    rows = [("production", m0.mean(), 0.0, "", m0[q.season == 2025].mean(), *cm.disagreements(q, "prod"), *cm.disagreements(q[q.season == 2026], "prod"))]
    coefs = {}
    for name, f in specs.items():
        q[name], c = cm.loso(q, f)
        coefs[name] = c
        m1 = (q[name] - q.y).abs()
        wins = sum(m1[q.season == s].mean() < m0[q.season == s].mean() for s in q.season.unique())
        rows.append((name, m1.mean(), 100 * (m1.mean() / m0.mean() - 1), f"{wins}/6", m1[q.season == 2025].mean(),
                     *cm.disagreements(q, name), *cm.disagreements(q[q.season == 2026], name)))
    t = pd.DataFrame(rows, columns=["variant", "miss", "vs_prod_%", "seasons_better", "miss_2025", "dis_n", "right", "dis2_n", "right2",
                                    "n26", "r26", "n2_26", "r2_26"])
    pd.set_option("display.width", 260)
    print(f"\n== {pos}  ({len(q)} player-weeks)")
    print(t.round(3).to_string(index=False))
    print("all-role coefficients (median over folds):", coefs["all role signals"].round(3).to_dict())
    q.to_parquet(DATA / "processed" / f"role_study_{pos}.parquet")


if __name__ == "__main__":
    for p in sys.argv[1:] or ["RB"]:
        grade(p)
