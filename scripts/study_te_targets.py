"""Does TE targets per game (weeks 1-10) improve rest-of-season PPG projections (weeks 11-17)? Nov 18 2026.

A) Simple test on 2018-25 (rule 5 window): ppg_after ~ ppg_sofar  vs  ~ ppg_sofar + targets/g, leave-one-season-out.
B) Against the real model (ros_history_wk11: model predictions made after week 10, 2021-25):
   ppg_after ~ model_pred  vs  ~ model_pred + targets/g, leave-one-season-out.
TE seasons with 5+ games through week 10 and 3+ games after. Burbs scoring. Prints average miss (MAE).
"""
import numpy as np
import polars as pl

from ffmodel.config import DATA

pw = pl.read_parquet(DATA / "processed" / "player_weeks.parquet").filter(pl.col("position") == "TE", pl.col("week") <= 17)
pre = pw.filter(pl.col("week") <= 10).group_by("player_id", "season").agg(
    pl.len().alias("g"), pl.col("fpts_burbs").mean().alias("ppg"), pl.col("targets").fill_null(0).mean().alias("tpg"))
post = pw.filter(pl.col("week") >= 11).group_by("player_id", "season").agg(
    pl.len().alias("g2"), pl.col("fpts_burbs").mean().alias("after"))
d = pre.join(post, on=["player_id", "season"]).filter(pl.col("g") >= 5, pl.col("g2") >= 3)


def loso(df, xs, y="after"):
    errs = []
    for s in sorted(df["season"].unique()):
        tr, te = df.filter(pl.col("season") != s), df.filter(pl.col("season") == s)
        X = np.column_stack([np.ones(tr.height)] + [tr[c].to_numpy() for c in xs])
        b = np.linalg.lstsq(X, tr[y].to_numpy(), rcond=None)[0]
        Xt = np.column_stack([np.ones(te.height)] + [te[c].to_numpy() for c in xs])
        errs.append(np.abs(Xt @ b - te[y].to_numpy()))
    return float(np.concatenate(errs).mean())


a = d.filter(pl.col("season").is_between(2018, 2025))
b0, b1 = loso(a, ["ppg"]), loso(a, ["ppg", "tpg"])
print(f"A) 2018-25 simple, n={a.height}: points only {b0:.3f}  + targets {b1:.3f}  change {100 * (b1 / b0 - 1):+.1f}%")

import json
f = json.load(open(DATA / "processed" / "ros_fit.json"))
k, u, r, sw = (f[x]["burbs"]["TE"] for x in ("k", "u", "r", "s"))
h = (pl.read_parquet(DATA / "processed" / "ros_history_wk11.parquet").filter(pl.col("position") == "TE", pl.col("n") >= 5))
# the model's actual rest-of-season formula (ros.project): prior blended with results so far, usage luck,
# role trend and schedule, using the fitted TE weights
so = pl.col("ppg_burbs_sofar") - sw * pl.col("sos_faced_burbs").fill_null(0)
h = h.with_columns(((k * pl.col("pred_ppg_burbs") + pl.col("n") * (so + u * pl.col("luck_pg").fill_null(0)
                    + r * pl.col("role_trend").fill_null(0))) / (k + pl.col("n"))
                   + sw * pl.col("sos_left_burbs").fill_null(0)).alias("model"))
m = h.select("player_id", "season", "model").join(d, on=["player_id", "season"])
raw = float((m["model"] - m["after"]).abs().mean())
c0, c1 = loso(m, ["model"]), loso(m, ["model", "tpg"])
print(f"B) 2021-25 vs the model, n={m.height}: model as-is {raw:.3f}  model recalibrated {c0:.3f}  + targets {c1:.3f}  "
      f"change vs as-is {100 * (c1 / raw - 1):+.1f}%, vs recalibrated {100 * (c1 / c0 - 1):+.1f}%")
X = np.column_stack([np.ones(m.height), m["model"].to_numpy(), m["tpg"].to_numpy()])
print("full-sample fit (intercept, model, targets/g):", np.round(np.linalg.lstsq(X, m["after"].to_numpy(), rcond=None)[0], 3))
