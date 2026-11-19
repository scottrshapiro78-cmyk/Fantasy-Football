"""Nov 19 2026: weekly rankings on the 2006+ data. Run with FF_DATA=data_2006 FF_FIRST_SEASON=2006.

Player-weeks, weeks 5-17, 2006-25, players with 3+ games so far and 6+ PPG so far (fantasy-relevant).
Prediction = his PPG so far + MATCH[pos] x opponent's points allowed to the position vs expectation so far
(shrunk like build_weekly: sum / (games + 4)) + VEGAS[pos] x (team's implied points this week - its season average so far).
Compares build_weekly's production weights, no adjustments, and weights refit leave-one-season-out. MAE per position.
"""
import numpy as np
import polars as pl

from ffmodel import data
from ffmodel.config import DATA

PROD_M = {"QB": 1.0, "RB": 0.45, "WR": 0.35, "TE": 0.55}
PROD_V = {"QB": 0.0, "RB": 0.075, "WR": 0.0, "TE": 0.075}
pw = (pl.read_parquet(DATA / "processed" / "player_weeks.parquet")
      .filter(pl.col("position").is_in(list(PROD_M)), pl.col("week") <= 17)
      .select("player_id", "position", "season", pl.col("week").cast(pl.Int32), "team", "opponent_team", pl.col("fpts_burbs").alias("pts"))
      .sort("player_id", "season", "week"))
# player's PPG and games before this week
pw = pw.with_columns(
    (pl.col("pts").cum_sum().over("player_id", "season") - pl.col("pts")).alias("cum"),
    (pl.col("pts").cum_count().over("player_id", "season") - 1).alias("g_before"))
pw = pw.with_columns((pl.col("cum") / pl.col("g_before")).alias("sofar"))
# defense effect so far: residual vs the player's own PPG so far, by opponent and position, prior weeks only
res = pw.filter(pl.col("g_before") >= 2).with_columns((pl.col("pts") - pl.col("sofar")).alias("r"))
dw = res.group_by("season", "opponent_team", "position", "week").agg(pl.col("r").sum().alias("rs"), pl.len().alias("k")).sort("week")
dw = dw.with_columns((pl.col("rs").cum_sum().over("season", "opponent_team", "position") - pl.col("rs")).alias("rs_b"),
                     (pl.col("week").cum_count().over("season", "opponent_team", "position") - 1).alias("gm_b"))
dw = dw.with_columns((pl.col("rs_b") / (pl.col("gm_b") + 4)).alias("eff")).select("season", "opponent_team", "position", "week", "eff")
# implied team points from the betting lines
sc = data.schedules(list(range(2006, 2026))).filter(pl.col("game_type") == "REG").select(
    "season", pl.col("week").cast(pl.Int32), "home_team", "away_team", "spread_line", "total_line")
imp = pl.concat([sc.select("season", "week", pl.col("home_team").alias("team"), ((pl.col("total_line") + pl.col("spread_line")) / 2).alias("imp")),
                 sc.select("season", "week", pl.col("away_team").alias("team"), ((pl.col("total_line") - pl.col("spread_line")) / 2).alias("imp"))]).sort("week")
imp = imp.with_columns(((pl.col("imp").cum_sum().over("season", "team") - pl.col("imp")) /
                        (pl.col("week").cum_count().over("season", "team") - 1)).alias("avg_imp"))
d = (pw.filter(pl.col("week").is_between(5, 17), pl.col("g_before") >= 3, pl.col("sofar") >= 6)
     .join(dw, on=["season", "opponent_team", "position", "week"], how="left")
     .join(imp, on=["season", "week", "team"], how="left")
     .with_columns(pl.col("eff").fill_null(0), (pl.col("imp") - pl.col("avg_imp")).fill_null(0).alias("dimp"))
     .drop_nulls("pts"))
print(f"player-weeks {d.height:,}, seasons {d['season'].min()}-{d['season'].max()}")
for p in PROD_M:
    q = d.filter(pl.col("position") == p)
    y, so, e, v = (q[c].to_numpy() for c in ("pts", "sofar", "eff", "dimp"))
    none = np.abs(so - y).mean()
    prod = np.abs(so + PROD_M[p] * e + PROD_V[p] * v - y).mean()
    errs, ms, vs = [], [], []
    seas = q["season"].to_numpy()
    for s in np.unique(seas):
        tr, te = seas != s, seas == s
        best = min(((m, w) for m in np.arange(0, 1.51, 0.05) for w in np.arange(-0.1, 0.41, 0.025)),
                   key=lambda g: np.abs(so[tr] + g[0] * e[tr] + g[1] * v[tr] - y[tr]).mean())
        ms.append(best[0]); vs.append(best[1])
        errs.append(np.abs(so[te] + best[0] * e[te] + best[1] * v[te] - y[te]))
    refit = np.concatenate(errs).mean()
    print(f"  {p} n={q.height:,}: no adjustment {none:.3f} | production weights {prod:.3f} ({100 * (prod / none - 1):+.2f}%) | "
          f"refit {refit:.3f} ({100 * (refit / prod - 1):+.2f}% vs production); refit MATCH {np.median(ms):.2f} (prod {PROD_M[p]}), "
          f"VEGAS {np.median(vs):.3f} (prod {PROD_V[p]})")
