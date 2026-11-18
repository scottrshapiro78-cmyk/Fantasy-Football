"""Backtest: replacement-level anchor in ROS value (option=True), current vs robust_base.

Mid-season cut-offs (weeks 8-11) of 2021-2025 from the ROS fitting history. For each cut-off:
  model ROS PPG = the fitted ROS blend (prior + results so far, ros_fit.json), expected games = fitted play
  rate for the player's status at the time x his team's games left; value() with option=True, current anchor
  vs the robust anchor. Truth: actual points after the cut-off over the ACTUAL replacement level at the
  position (depth-th best actual PPG after), times actual games.
Scores: cross-position rank correlation (top 250 by model), mean error per position (model - actual VORP),
and how much each position's offset jumps from one cut-off week to the next (stability).
PYTHONPATH=src python3 scripts/study_replacement_anchor.py [burbs|cbs]
"""
import json
import sys

import numpy as np
import polars as pl

from ffmodel.config import DATA
from ffmodel.value import value

lg = sys.argv[1] if len(sys.argv) > 1 else "burbs"
fitp = json.loads((DATA / "processed" / "ros_fit.json").read_text())
POS4 = ["QB", "RB", "WR", "TE"]
col = f"pred_ppg_{lg}"
rows, offs = [], []
for thru in (8, 9, 10, 11):
    h = pl.read_parquet(DATA / "processed" / f"ros_history_wk{thru}.parquet").filter(pl.col("position").is_in(POS4))
    for s in sorted(h["season"].unique().to_list()):
        d = h.filter(pl.col("season") == s, pl.col(col).is_not_null(), pl.col("tg_left").fill_null(0) > 0)
        k = pl.col("position").replace_strict(fitp["k"][lg], return_dtype=pl.Float64)
        u = pl.col("position").replace_strict(fitp["u"][lg], return_dtype=pl.Float64)
        r = pl.col("position").replace_strict(fitp["r"][lg], return_dtype=pl.Float64)
        n = pl.col("n").cast(pl.Float64)
        so = pl.col(f"{'ppg_' + lg}_sofar").fill_null(0)
        pred = (k * pl.col(col) + n * (so + u * pl.col("luck_pg").fill_null(0) + r * pl.col("role_trend").fill_null(0))) / (k + n)
        play = pl.col("status").fill_null("active").replace_strict(fitp["play"], default=fitp["play"].get("active", 0.9),
                                                                   return_dtype=pl.Float64)
        d = d.with_columns(pred.alias("ros"), (play * pl.col("tg_left") / 13).alias("exp_frac"),
                           pl.lit(False).alias("rookie"))
        pool = d.select("player_id", "position", pl.col("ros").alias(col), "exp_frac", "rookie",
                        pl.col(f"ppg_{lg}_after").fill_null(0).alias("act_ppg"), pl.col("n_after").fill_null(0).alias("act_g"))
        res = {}
        for name, rb in (("current", False), ("robust", True)):
            b, _, depth = value(pool.drop("act_ppg", "act_g"), lg, option=True, robust_base=rb)
            res[name] = b.select("player_id", pl.col("vorp").alias(name))
        # actual replacement level: depth-th best actual PPG after the cut-off (3+ games)
        repl = {p: (pool.filter(pl.col("position") == p, pl.col("act_g") >= 3).sort("act_ppg", descending=True)["act_ppg"].to_list()
                    + [0.0] * 999)[depth[p]] for p in POS4}
        j = (pool.join(res["current"], on="player_id").join(res["robust"], on="player_id")
                 .with_columns(((pl.col("act_ppg") - pl.col("position").replace_strict(repl, return_dtype=pl.Float64))
                                * pl.col("act_g")).alias("actual")))
        top = j.sort("current", descending=True).head(250)
        out = {"season": s, "thru": thru}
        for name in ("current", "robust"):
            t = j.sort(name, descending=True).head(250)
            out[f"rank_{name}"] = t.select(pl.corr(name, "actual", method="spearman")).item()
            for p in POS4:
                q = t.filter(pl.col("position") == p)
                out[f"err_{name}_{p}"] = (q[name] - q["actual"]).mean()
        rows.append(out)
        # offset each anchor applies per position (how far the base moved vs a healthy anchor) for stability
        for name in ("current", "robust"):
            for p in POS4:
                offs.append({"season": s, "thru": thru, "variant": name, "pos": p,
                             "med_vorp_top": float(j.filter(pl.col("position") == p).sort(name, descending=True)
                                                   .head(12)[name].median())})
R = pl.DataFrame(rows)
print(f"League {lg}, {R.height} season x cut-off week cases")
print("Cross-position rank correlation with actual ROS value (top 250 by model):")
print(R.select(pl.col("rank_current").mean().round(4), pl.col("rank_robust").mean().round(4),
               (pl.col("rank_robust") > pl.col("rank_current")).sum().alias("robust_better_cases")))
print("Mean |error| of ROS VORP by position (model - actual), averaged over cases:")
print(pl.DataFrame([{"pos": p, "current": round(float(R[f'err_current_{p}'].abs().mean()), 2),
                     "robust": round(float(R[f'err_robust_{p}'].abs().mean()), 2),
                     "bias_current": round(float(R[f'err_current_{p}'].mean()), 2),
                     "bias_robust": round(float(R[f'err_robust_{p}'].mean()), 2)} for p in POS4]))
O = pl.DataFrame(offs).sort("season", "thru")
jump = (O.with_columns(pl.col("med_vorp_top").diff().over("season", "variant", "pos").abs().alias("jump"))
          .group_by("variant", "pos").agg(pl.col("jump").mean().round(2)).sort("pos", "variant"))
print("Week-to-week jump in a position's top-12 median value (lower = steadier):")
print(jump.pivot(on="variant", index="pos", values="jump"))
