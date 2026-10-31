"""Projection engine v1: next-season points per game, per position and league.

QB, RB, WR: ridge regression on the predictor table (features.FEATURES), chosen because it
beat both the 3-year baseline and LightGBM out of sample (docs/findings.md).
TE: the 3-year recency-weighted baseline, which still beats every model tried.

Rookies with no NFL games are not projected yet (Phase 6: draft-capital prior).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES

MODEL_BY_POS = {"QB": "ridge", "RB": "ridge", "WR": "ridge", "TE": "baseline"}
# Ridge alone under-projects proven QBs (3-yr PPG 20+: -0.7 PPG on average 2017-2025; Josh Allen
# by ~3 PPG six years running). Mixing 25% of the 3-yr history back in removes that bias and
# lowers error out of sample (Burbs 2.65 -> 2.60, CBS 3.06 -> 3.05).
# Same check for RB and WR (2017-2025, proven = 3-yr PPG 16+): ridge alone under-projected
# them by 2.1 (RB) and 0.9 (WR) PPG. History weights below cut that to about -0.6 and -0.2.
HISTORY_WEIGHT = {"QB": 0.25, "RB": 0.40, "WR": 0.30}
# TE uses the 3-yr baseline, which over-projected proven TEs by 1.3 PPG; pulling it 30% of the
# way toward the season's TE average removes the bias and cuts error 2.93 -> 2.58.
TE_SHRINK = 0.30
# Kickers repeat weakly (year-to-year correlation 0.27): pull 70% toward the average.
K_SHRINK = 0.70
ALPHAS = np.logspace(-2, 3, 20)


def league_features(league: str) -> list[str]:
    swap = {"ppg_burbs": f"ppg_{league}", "ppg_burbs_lag1": f"ppg_{league}_lag1"}
    return [swap.get(c, c) for c in FEATURES] + ["first_year_data", "age2", f"base_{league}"]


def prepare(df: pl.DataFrame, league: str, medians: pd.DataFrame | None = None):
    d = df.to_pandas()
    d["first_year_data"] = d[f"ppg_{league}_lag1"].isna().astype(float)
    d[f"ppg_{league}_lag1"] = d[f"ppg_{league}_lag1"].fillna(d[f"ppg_{league}"])
    d["games_lag1"] = d["games_lag1"].fillna(0)
    d["tough_idx"] = d["tough_idx"].clip(0, 3)
    d["age2"] = (d["age_next"] - 27) ** 2
    X = league_features(league)
    for c in X:
        d[c] = d[c].astype(float)
    if medians is None:
        medians = d.groupby("position")[X].median()
    for c in X:
        d[c] = d[c].fillna(d["position"].map(medians[c]))
    return d, medians


class Projector:
    def __init__(self, league: str):
        self.league = league
        self.models: dict = {}

    def fit(self, train: pl.DataFrame) -> "Projector":
        t = train.filter(pl.col("next_games") >= 6, pl.col("games") >= 4, pl.col(f"ppg_{self.league}") >= 3)
        d, self.medians = prepare(t, self.league)
        X, y = league_features(self.league), f"next_ppg_{self.league}"
        for pos, kind in MODEL_BY_POS.items():
            if kind == "ridge":
                p = d[d.position == pos]
                self.models[pos] = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)).fit(p[X], p[y])
        return self

    def predict(self, rows: pl.DataFrame) -> pl.DataFrame:
        d, _ = prepare(rows, self.league, self.medians)
        X = league_features(self.league)
        pred = d[f"base_{self.league}"].astype(float).copy()
        for pos, m in self.models.items():
            idx = d.position == pos
            if idx.any():
                w = HISTORY_WEIGHT.get(pos, 0.0)
                pred[idx] = (1 - w) * m.predict(d.loc[idx, X]) + w * d.loc[idx, f"base_{self.league}"]
        te = d.position == "TE"
        if te.any():
            mu = d.loc[te, f"base_{self.league}"].nlargest(40).mean()
            pred[te] = d.loc[te, f"base_{self.league}"] - TE_SHRINK * (d.loc[te, f"base_{self.league}"] - mu)
        d["pred"] = pred.clip(lower=0)
        return pl.from_pandas(d[["player_id", "player_display_name", "position", "season", "team", "pred"]]
                              ).rename({"pred": f"pred_ppg_{self.league}"})


def project_kickers(seasons: pl.DataFrame, target: int, league: str) -> pl.DataFrame:
    """3-yr weighted PPG for kickers, pulled K_SHRINK of the way to the top-24 average."""
    k = seasons.filter(pl.col("position") == "K", pl.col("season").is_between(target - 3, target - 1))
    k = k.with_columns(pl.col("season").replace_strict(
        {target - 1: 3, target - 2: 2, target - 3: 1}, return_dtype=pl.Float64).alias("w"))
    b = (k.group_by("player_id").agg(
            pl.col("player_display_name").last(), pl.col("team").sort_by("season").last(),
            ((pl.col(f"fpts_{league}") * pl.col("w")).sum() / (pl.col("games") * pl.col("w")).sum()).alias("base"),
            pl.col("season").max().alias("last")))
    b = b.filter(pl.col("last") == target - 1)
    mu = b.sort("base", descending=True).head(24)["base"].mean()
    return b.select("player_id", "player_display_name", pl.lit("K").alias("position"), "team",
                    (pl.col("base") - K_SHRINK * (pl.col("base") - mu)).alias(f"pred_ppg_{league}"))
