"""Do TE targets per game or snap share (weeks 1-10) improve rest-of-season PPG (weeks 11-17)? Nov 18 2026.

A) Simple test, 2012-25: ppg_after ~ ppg_sofar (+ feature), leave-one-season-out.
B) Against the model's own rest-of-season formula (ros.project weights from ros_fit.json), 2019-25 = every season
   the model backtest can be built for (its preseason projection needs earlier seasons to train on, so 2018 and
   before fail). ppg_after ~ model (+ feature), leave-one-season-out. Rule 5: adopt only if the miss drops >= 2%.
TE seasons with 5+ games through week 10 and 3+ games after. Burbs scoring. Average miss (MAE).
"""
import json

import numpy as np
import polars as pl

from ffmodel import data
from ffmodel.config import DATA

pw = pl.read_parquet(DATA / "processed" / "player_weeks.parquet").filter(pl.col("position") == "TE", pl.col("week") <= 17)
pre = pw.filter(pl.col("week") <= 10).group_by("player_id", "season").agg(
    pl.len().alias("g"), pl.col("fpts_burbs").mean().alias("ppg"), pl.col("targets").fill_null(0).mean().alias("tpg"))
post = pw.filter(pl.col("week") >= 11).group_by("player_id", "season").agg(pl.len().alias("g2"), pl.col("fpts_burbs").mean().alias("after"))
ps = pl.read_parquet(DATA / "processed" / "player_seasons.parquet").select("player_id", "season", "pfr_id").drop_nulls()
sn = (data.snap_counts(list(range(2012, 2026))).filter(pl.col("week") <= 10, pl.col("game_type") == "REG")
      .with_columns(pl.col("season").cast(pl.Int32))
      .join(ps.with_columns(pl.col("season").cast(pl.Int32)), left_on=["pfr_player_id", "season"], right_on=["pfr_id", "season"])
      .group_by("player_id", "season").agg(pl.col("offense_pct").cast(pl.Float64).mean().alias("snap")))
d = (pre.join(post, on=["player_id", "season"]).filter(pl.col("g") >= 5, pl.col("g2") >= 3)
     .with_columns(pl.col("season").cast(pl.Int32)).join(sn, on=["player_id", "season"], how="left")
     .with_columns(pl.col("snap").fill_null(pl.col("snap").median())))


def loso(df, xs, y="after"):
    errs = []
    for s in sorted(df["season"].unique()):
        tr, te = df.filter(pl.col("season") != s), df.filter(pl.col("season") == s)
        X = np.column_stack([np.ones(tr.height)] + [tr[c].to_numpy() for c in xs])
        b = np.linalg.lstsq(X, tr[y].to_numpy(), rcond=None)[0]
        Xt = np.column_stack([np.ones(te.height)] + [te[c].to_numpy() for c in xs])
        errs.append(np.abs(Xt @ b - te[y].to_numpy()))
    return float(np.concatenate(errs).mean())


def report(tag, df, base):
    b0 = loso(df, [base])
    line = f"{tag}, n={df.height}, {df['season'].min()}-{df['season'].max()}: {base} {b0:.3f}"
    for extra in (["tpg"], ["snap"], ["tpg", "snap"]):
        b1 = loso(df, [base] + extra)
        line += f" | +{'+'.join(extra)} {b1:.3f} ({100 * (b1 / b0 - 1):+.1f}%)"
    print(line)


report("A) simple", d, "ppg")

f = json.load(open(DATA / "processed" / "ros_fit.json"))
k, u, r, sw = (f[x]["burbs"]["TE"] for x in ("k", "u", "r", "s"))
hs = [pl.read_parquet(DATA / "processed" / "ros_history_wk11.parquet")]
old = DATA / "processed" / "ros_history_wk11_2019_2020.parquet"
if old.exists():
    hs.append(pl.read_parquet(old))
h = pl.concat(hs, how="diagonal_relaxed").filter(pl.col("position") == "TE", pl.col("n") >= 5)
so = pl.col("ppg_burbs_sofar") - sw * pl.col("sos_faced_burbs").fill_null(0)
h = h.with_columns(((k * pl.col("pred_ppg_burbs") + pl.col("n") * (so + u * pl.col("luck_pg").fill_null(0)
                    + r * pl.col("role_trend").fill_null(0))) / (k + pl.col("n"))
                   + sw * pl.col("sos_left_burbs").fill_null(0)).alias("model"), pl.col("season").cast(pl.Int32))
m = h.select("player_id", "season", "model").join(d, on=["player_id", "season"])
print(f"B) model as-is (no refit) miss {float((m['model'] - m['after']).abs().mean()):.3f}")
report("B) vs the model", m, "model")
report("B') same seasons, simple", m, "ppg")
