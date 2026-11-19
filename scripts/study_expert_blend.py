"""Nov 19 2026: blend the model with FantasyPros expert consensus (ECR), 2020-25. Run with FF_DATA=data_2006.

ECR archive: dynastyprocess db_fpecr.parquet (weekly snapshots since late 2020; fetch from raw.githubusercontent.com).
ECR is a ranking, not points, so per position and snapshot the expert order is given the model's own sorted
predictions (same scale, expert order). Blend = (1-w) model + w expert; w picked leaving the graded season out.
1) Preseason: last redraft PPR positional ECR before the season vs Projector (ridge 2012+, rolling), next-season PPG.
2) ROS after week 10: ROS ECR snapshot up to week 11 Sunday vs the model backtest (production weights).
3) Weekly: weekly ECR (Friday snapshot) vs simplified weekly model (PPG so far + production matchup/Vegas), weeks 5-17.
"""
import json
import sys

import numpy as np
import pandas as pd
import polars as pl

from ffmodel import data
from ffmodel.config import DATA
from ffmodel.projection import Projector

ecr = pl.read_parquet(DATA / "external" / "db_fpecr.parquet").filter(pl.col("pos").is_in(["QB", "RB", "WR", "TE"]))
ids = (pl.read_parquet(DATA / "processed" / "player_seasons.parquet").select("player_id", "yahoo_id", "merge_name", "position")
       .unique(["player_id"]))
yid = ids.drop_nulls("yahoo_id").unique("yahoo_id").select("yahoo_id", pl.col("player_id").alias("pid_y"))
nm = ids.group_by("merge_name", "position").agg(pl.col("player_id").first().alias("pid_n"), pl.len().alias("k")).filter(pl.col("k") == 1)
ecr = (ecr.join(yid, on="yahoo_id", how="left")
       .with_columns(pl.col("mergename").str.to_lowercase().str.replace_all(r"[.'’]", "").str.replace_all(r"\s+(jr|sr|ii|iii|iv)$", "").alias("mn"))
       .join(nm, left_on=["mn", "pos"], right_on=["merge_name", "position"], how="left")
       .with_columns(pl.coalesce("pid_y", "pid_n").alias("player_id")).drop_nulls("player_id")
       .with_columns(pl.col("scrape_date").str.to_date()))
GRID = np.arange(0, 1.01, 0.1)


def to_points(df, model, keys):
    """Expert order mapped onto the model's own sorted values within each group."""
    df = df.sort(keys + ["ecr"]).with_columns(pl.int_range(pl.len()).over(keys).alias("_r"))
    srt = df.sort(keys + [model], descending=[False] * len(keys) + [True]).with_columns(pl.int_range(pl.len()).over(keys).alias("_r"))
    return df.join(srt.select(keys + ["_r", pl.col(model).alias("expert")]), on=keys + ["_r"]).drop("_r")


def grade(df, model, y, label):
    out = []
    for p in ["QB", "RB", "WR", "TE"]:
        q = df.filter(pl.col("position") == p).to_pandas()
        if q.empty:
            continue
        errs, ws = [], []
        for s in sorted(q.season.unique()):
            tr, te = q[q.season != s], q[q.season == s]
            w = min(GRID, key=lambda w: np.abs((1 - w) * tr[model] + w * tr.expert - tr[y]).mean())
            ws.append(w); errs.append(np.abs((1 - w) * te[model] + w * te.expert - te[y]))
        m0 = np.abs(q[model] - q[y]).mean(); ex = np.abs(q.expert - q[y]).mean(); bl = pd.concat(errs).mean()
        out.append((label, p, len(q), q.season.nunique(), round(m0, 3), round(ex, 3), round(bl, 3), f"{100 * (bl / m0 - 1):+.1f}%", np.median(ws)))
    return out


rows = []
which = sys.argv[1:] or ["pre", "ros", "week"]
sched = data.schedules(list(range(2020, 2027))).filter(pl.col("game_type") == "REG").with_columns(pl.col("gameday").str.to_date())
sunday = sched.group_by("season", "week").agg(pl.col("gameday").max().alias("sun"))

if "pre" in which:
    f = pl.read_parquet(DATA / "processed" / "features.parquet")
    POOL = {"QB": 24, "RB": 48, "WR": 60, "TE": 24}
    pre_pages = ecr.filter(pl.col("page_type").is_in([f"redraft-{p.lower()}" for p in POOL]), pl.col("ecr_type") == "rp")
    parts = []
    for t in range(2020, 2026):
        kick = sched.filter(pl.col("season") == t)["gameday"].min()
        e = pre_pages.filter(pl.col("scrape_date") < kick)
        e = e.filter(pl.col("scrape_date") == e["scrape_date"].max()).unique("player_id").select("player_id", "ecr")
        tr = f.filter(pl.col("season") + 1 < t, pl.col("season") >= 2012)
        te = f.filter(pl.col("season") == t - 1, pl.col("next_games") >= 6)
        m = Projector("burbs").fit(tr).predict(te).select("player_id", "position", pl.col("pred_ppg_burbs").alias("model"))
        m = m.join(te.select("player_id", pl.col("next_ppg_burbs").alias("y")), on="player_id").join(e, on="player_id")
        m = pl.concat([m.filter(pl.col("position") == p).filter((pl.col("model").rank(descending=True) <= n) | (pl.col("ecr").rank() <= n))
                       for p, n in POOL.items()])
        parts.append(to_points(m, "model", ["position"]).with_columns(pl.lit(t).alias("season")))
    rows += grade(pl.concat(parts), "model", "y", "preseason")

if "ros" in which:
    fit = json.load(open(DATA / "processed" / "ros_fit.json"))
    MIN_PRI = {"QB": 12, "RB": 6, "WR": 6, "TE": 5}
    h = pl.read_parquet(DATA / "processed" / "ros_history_wk11_long.parquet").filter(pl.col("season") >= 2020, pl.col("n") >= 5, pl.col("n_after") >= 3, pl.col("position").is_in(list(MIN_PRI)))
    parts = []
    for (p, s), c in h.group_by("position", "season"):
        c = c.filter(pl.col("pred_ppg_burbs") >= MIN_PRI[p])
        k, u, r, s_ = (fit[x]["burbs"][p] for x in ("k", "u", "r", "s"))
        so = c["ppg_burbs_sofar"].to_numpy() - s_ * c["sos_faced_burbs"].fill_null(0).to_numpy()
        inner = so + u * c["luck_pg"].fill_null(0).to_numpy() + r * c["role_trend"].fill_null(0).to_numpy()
        pred = (k * c["pred_ppg_burbs"].to_numpy() + c["n"].to_numpy() * inner) / (k + c["n"].to_numpy()) + s_ * c["sos_left_burbs"].fill_null(0).to_numpy()
        c = c.with_columns(pl.Series("model", pred))
        sun11 = sunday.filter(pl.col("season") == s, pl.col("week") == 11)["sun"][0]
        e = ecr.filter(pl.col("page_type") == f"redraft-{p.lower()}", pl.col("fp_page").str.contains("ros"), pl.col("scrape_date") <= sun11,
                       pl.col("scrape_date") > sun11 - pl.duration(days=10))
        if e.is_empty():
            continue
        e = e.filter(pl.col("scrape_date") == e["scrape_date"].max()).unique("player_id").select("player_id", "ecr")
        c = c.join(e, on="player_id").select("player_id", "position", "season", "model", pl.col("ppg_burbs_after").alias("y"), "ecr")
        parts.append(to_points(c, "model", ["position"]))
    rows += grade(pl.concat(parts), "model", "y", "ROS wk11+")

if "week" in which:
    PM = {"QB": 1.0, "RB": 0.45, "WR": 0.35, "TE": 0.55}
    PV = {"QB": 0.0, "RB": 0.075, "WR": 0.0, "TE": 0.075}
    pw = (pl.read_parquet(DATA / "processed" / "player_weeks.parquet").filter(pl.col("position").is_in(list(PM)), pl.col("week") <= 17, pl.col("season") >= 2019)
          .select("player_id", "position", "season", pl.col("week").cast(pl.Int32), "team", "opponent_team", pl.col("fpts_burbs").alias("pts"))
          .sort("player_id", "season", "week"))
    pw = pw.with_columns((pl.col("pts").cum_sum().over("player_id", "season") - pl.col("pts")).alias("cum"),
                         (pl.col("pts").cum_count().over("player_id", "season") - 1).alias("g_before"))
    pw = pw.with_columns((pl.col("cum") / pl.col("g_before")).alias("sofar"))
    res = pw.filter(pl.col("g_before") >= 2).with_columns((pl.col("pts") - pl.col("sofar")).alias("r"))
    dw = res.group_by("season", "opponent_team", "position", "week").agg(pl.col("r").sum().alias("rs")).sort("week")
    dw = dw.with_columns((pl.col("rs").cum_sum().over("season", "opponent_team", "position") - pl.col("rs")).alias("rs_b"),
                         (pl.col("week").cum_count().over("season", "opponent_team", "position") - 1).alias("gm_b"))
    dw = dw.with_columns((pl.col("rs_b") / (pl.col("gm_b") + 4)).alias("eff")).select("season", "opponent_team", "position", "week", "eff")
    sc = sched.select("season", pl.col("week").cast(pl.Int32), "home_team", "away_team", "spread_line", "total_line")
    imp = pl.concat([sc.select("season", "week", pl.col("home_team").alias("team"), ((pl.col("total_line") + pl.col("spread_line")) / 2).alias("imp")),
                     sc.select("season", "week", pl.col("away_team").alias("team"), ((pl.col("total_line") - pl.col("spread_line")) / 2).alias("imp"))]).sort("week")
    imp = imp.with_columns(((pl.col("imp").cum_sum().over("season", "team") - pl.col("imp")) / (pl.col("week").cum_count().over("season", "team") - 1)).alias("avg_imp"))
    d = (pw.filter(pl.col("week").is_between(5, 17), pl.col("g_before") >= 3, pl.col("sofar") >= 6)
         .join(dw, on=["season", "opponent_team", "position", "week"], how="left").join(imp, on=["season", "week", "team"], how="left")
         .with_columns(pl.col("eff").fill_null(0), (pl.col("imp") - pl.col("avg_imp")).fill_null(0).alias("dimp")).drop_nulls("pts"))
    d = d.with_columns((pl.col("sofar") + pl.col("position").replace_strict(PM) * pl.col("eff") + pl.col("position").replace_strict(PV) * pl.col("dimp")).alias("model"))
    # weekly ECR: snapshot taken in the 6 days before that week's Sunday
    we = ecr.filter(pl.col("page_type").str.starts_with("weekly-"), pl.col("ecr_type") == "wp").select("player_id", "ecr", "scrape_date")
    we = we.join(sunday.with_columns(pl.col("week").cast(pl.Int32)), how="cross").filter(
        (pl.col("sun") - pl.col("scrape_date")).dt.total_days().is_between(0, 6))
    we = we.sort("scrape_date").group_by("player_id", "season", "week").agg(pl.col("ecr").last())
    d = d.join(we, on=["player_id", "season", "week"]).select("player_id", "position", "season", "week", "model", pl.col("pts").alias("y"), "ecr")
    rows += grade(to_points(d, "model", ["season", "week", "position"]), "model", "y", "weekly")

r = pd.DataFrame(rows, columns=["horizon", "pos", "n", "seasons", "model_mae", "expert_only_mae", "blend_mae", "blend_vs_model", "w_expert"])
print(r.to_string(index=False))
