"""Nov 19 2026: which in-season signals pick waiver-wire adds that keep scoring. Run with FF_DATA=data_2006.

Pool each season = players outside the drafted pool, approximated by prior-season total Burbs points
(QB top 14, RB top 36, WR top 48, TE top 14 are treated as drafted; rookies are in the pool).
At each decision week W (3..12) rank pool players by each signal, take the top 3 per position, and grade
their points per game over weeks W+1..17 (games played, min 3) and the hit rate (finished at a starter level:
QB 17+, RB 12+, WR 12+, TE 9+ PPG). 'blend' = ridge on all signals, fit leaving the graded season out.
"""
import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import RidgeCV
from ffmodel.config import DATA

pw = pl.read_parquet(DATA / "processed" / "player_weeks.parquet").filter(pl.col("week") <= 17, pl.col("position").is_in(["QB", "RB", "WR", "TE"]))
pw = pw.with_columns((pl.col("targets").fill_null(0) + pl.col("carries").fill_null(0)).alias("opp"),
                     pl.col("target_share").fill_null(0), pl.col("wopr").fill_null(0))
DRAFTED = {"QB": 14, "RB": 36, "WR": 48, "TE": 14}
HIT = {"QB": 17, "RB": 12, "WR": 12, "TE": 9}
tot = pw.group_by("player_id", "position", "season").agg(pl.col("fpts_burbs").sum().alias("tot"))
drafted = pl.concat([tot.filter(pl.col("position") == p).with_columns((pl.col("tot").rank("ordinal", descending=True).over("season") <= n).alias("d"))
                     for p, n in DRAFTED.items()]).filter("d").select("player_id", (pl.col("season") + 1).alias("season"))
pw = pw.join(drafted.with_columns(pl.lit(1).alias("drafted")), on=["player_id", "season"], how="left").filter(pl.col("drafted").is_null())

SIG = ["last1", "last3", "sofar", "opp2", "opp_jump", "tshare2", "wopr2"]
rows = []
for W in range(3, 13):
    a = pw.filter(pl.col("week") <= W)
    sig = a.sort("week").group_by("player_id", "position", "season").agg(
        pl.col("week").max().alias("lastwk"), pl.len().alias("g"),
        pl.col("fpts_burbs").last().alias("last1"), pl.col("fpts_burbs").tail(3).mean().alias("last3"),
        pl.col("fpts_burbs").mean().alias("sofar"), pl.col("opp").tail(2).mean().alias("opp2"),
        (pl.col("opp").tail(2).mean() - pl.col("opp").head(pl.len() - 2).mean()).alias("opp_jump"),
        pl.col("target_share").tail(2).mean().alias("tshare2"), pl.col("wopr").tail(2).mean().alias("wopr2"))
    sig = sig.filter(pl.col("lastwk") >= W - 1)  # active in the last two weeks
    out = pw.filter(pl.col("week") > W).group_by("player_id", "season").agg(pl.len().alias("g_after"), pl.col("fpts_burbs").mean().alias("after"))
    rows.append(sig.join(out, on=["player_id", "season"]).filter(pl.col("g_after") >= 3).with_columns(pl.lit(W).alias("W")))
d = pl.concat(rows).to_pandas()
d[SIG] = d[SIG].fillna(0)
d["hit"] = d.apply(lambda r: r.after >= HIT[r.position], axis=1)

d["blend"] = np.nan
for p in DRAFTED:
    m = d.position == p
    for s in d.loc[m, "season"].unique():
        tr, te = m & (d.season != s), m & (d.season == s)
        r = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(d.loc[tr, SIG], d.loc[tr, "after"])
        d.loc[te, "blend"] = r.predict(d.loc[te, SIG])
    coef = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit((d.loc[m, SIG] - d.loc[m, SIG].mean()) / d.loc[m, SIG].std(), d.loc[m, "after"]).coef_
    print(f"{p} blend weights per 1 sd: " + ", ".join(f"{c} {w:+.2f}" for c, w in zip(SIG, coef)))

res = []
for p in DRAFTED:
    q = d[d.position == p]
    for s in SIG + ["blend"]:
        top = q.sort_values(s, ascending=False).groupby(["season", "W"]).head(3)
        res.append((p, s, top.after.mean(), top.hit.mean(), q.groupby(["season", "W"]).apply(lambda g: g[s].corr(g.after, method="spearman")).mean()))
r = pd.DataFrame(res, columns=["pos", "signal", "ros_ppg_top3", "hit_rate_top3", "rank_corr"])
print(f"seasons {d.season.min()}-{d.season.max()}, decision weeks 3-12, pool rows {len(d)}")
print(r.round(3).to_string(index=False))
