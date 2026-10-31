"""Grade the projection engine with the standard harness (train on seasons before t, predict t)."""
import polars as pl

from ffmodel import backtest as bt, features
from ffmodel.config import DATA
from ffmodel.projection import Projector

f = pl.read_parquet(DATA / "processed" / "features.parquet")


def engine(league):
    def predict(hist, t):
        train = f.filter(pl.col("season") + 1 < t)       # targets strictly before t
        rows = f.filter(pl.col("season") == t - 1)        # features from the season before t
        return Projector(league).fit(train).predict(rows).select("player_id", "position", f"pred_ppg_{league}")
    return predict


for lg in ["burbs", "cbs"]:
    seasons = [2021, 2022, 2023, 2024, 2025]
    a = bt.summary(bt.grade(bt.weighted_three(lg), lg, seasons)).rename({"mae": "base_mae", "rank_corr": "base_rc"})
    b = bt.summary(bt.grade(engine(lg), lg, seasons)).rename({"mae": "engine_mae", "rank_corr": "engine_rc"})
    print(lg); print(a.join(b, on="position").drop("bias", "bias_right"))
