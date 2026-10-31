"""Phase 2 backtest: does opponent-adjusted history predict next season better than raw?"""
import polars as pl

from ffmodel import backtest as bt
from ffmodel.config import DATA
from ffmodel.opponent import adjust, adjusted_seasons

LEAGUES = ["burbs", "cbs"]
weeks = pl.read_parquet(DATA / "processed" / "player_weeks.parquet")
raw = bt.seasons()


def weighted_adj(league, w=(3, 2, 1)):
    def f(hist, t):
        h = hist.filter(pl.col("season") >= t - 3).with_columns(
            pl.col("season").replace_strict({t - 1: w[0], t - 2: w[1], t - 3: w[2]}).alias("w"))
        return (h.group_by("player_id", "position")
                 .agg(((pl.col(f"adj_fpts_{league}") * pl.col("w")).sum()
                       / (pl.col("games") * pl.col("w")).sum()).alias(f"pred_ppg_{league}")))
    return f


rows = []
for k in [10, 30, 100]:
    adj = adjusted_seasons(adjust(weeks, LEAGUES, k), raw, LEAGUES)
    bt.seasons = lambda adj=adj: adj  # harness reads the adjusted table
    for lg in LEAGUES:
        for name, pred in [("raw", bt.weighted_three(lg)), ("adjusted", weighted_adj(lg))]:
            s = bt.summary(bt.grade(pred, lg)).with_columns(
                pl.lit(lg).alias("league"), pl.lit(name).alias("model"), pl.lit(k).alias("k"))
            rows.append(s)
res = pl.concat(rows)
pl.Config.set_tbl_rows(100)
print(res.filter(pl.col("league") == "burbs").pivot(on="model", index=["position", "k"], values="mae"))
print(res.filter(pl.col("league") == "cbs").pivot(on="model", index=["position", "k"], values="mae"))
print(res.filter(pl.col("league") == "burbs").pivot(on="model", index=["position", "k"], values="rank_corr"))
