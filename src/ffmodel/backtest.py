"""Rolling-origin backtest: every model is trained on seasons < t and graded on season t.

A predictor is any function  predict(history: player_seasons before t, target_season: int)
-> DataFrame[player_id, position, pred_ppg_<league>].  The harness grades it per position on
players who actually played 6+ games, the pool that matters for drafting.
"""
from __future__ import annotations

from typing import Callable

import polars as pl

from .config import DATA

TEST_SEASONS = [2023, 2024, 2025]
POOL = {"QB": 24, "RB": 48, "WR": 60, "TE": 20}  # roughly the drafted pool in a 12-team league
MIN_GAMES = 6

Predictor = Callable[[pl.DataFrame, int], pl.DataFrame]


def seasons() -> pl.DataFrame:
    return pl.read_parquet(DATA / "processed" / "player_seasons.parquet")


def grade(predict: Predictor, league: str = "burbs", test_seasons=TEST_SEASONS) -> pl.DataFrame:
    hist = seasons()
    rows = []
    for t in test_seasons:
        pred = predict(hist.filter(pl.col("season") < t), t)
        actual = hist.filter(pl.col("season") == t, pl.col("games") >= MIN_GAMES).select(
            "player_id", "position", pl.col(f"ppg_{league}").alias("actual"))
        j = actual.join(pred.select("player_id", pl.col(f"pred_ppg_{league}").alias("pred")),
                        on="player_id", how="inner")
        for pos, n in POOL.items():
            p = j.filter(pl.col("position") == pos).sort("actual", descending=True).head(n)
            if p.height < 5:
                continue
            rows.append({
                "season": t, "position": pos, "n": p.height,
                "mae": (p["pred"] - p["actual"]).abs().mean(),
                "bias": (p["pred"] - p["actual"]).mean(),
                "rank_corr": p.select(pl.corr("pred", "actual", method="spearman")).item(),
            })
    return pl.DataFrame(rows)


def summary(results: pl.DataFrame) -> pl.DataFrame:
    return (results.group_by("position")
                   .agg(pl.col("mae").mean().round(2), pl.col("bias").mean().round(2),
                        pl.col("rank_corr").mean().round(3))
                   .sort("position"))


# ---- baselines every model must beat --------------------------------------------------

def last_season(league: str) -> Predictor:
    def f(hist: pl.DataFrame, t: int) -> pl.DataFrame:
        return (hist.filter(pl.col("season") == t - 1)
                    .select("player_id", "position", pl.col(f"ppg_{league}").alias(f"pred_ppg_{league}")))
    return f


def weighted_three(league: str, w=(3, 2, 1)) -> Predictor:
    """Last three seasons' PPG, weighted by recency and games played."""
    def f(hist: pl.DataFrame, t: int) -> pl.DataFrame:
        h = hist.filter(pl.col("season") >= t - 3).with_columns(
            pl.col("season").replace_strict({t - 1: w[0], t - 2: w[1], t - 3: w[2]}).alias("w"))
        return (h.group_by("player_id", "position")
                 .agg(((pl.col(f"fpts_{league}") * pl.col("w")).sum()
                       / (pl.col("games") * pl.col("w")).sum()).alias(f"pred_ppg_{league}")))
    return f
