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
ALPHAS = np.logspace(-2, 3, 20)


def league_features(league: str) -> list[str]:
    swap = {"ppg_burbs": f"ppg_{league}", "ppg_burbs_lag1": f"ppg_{league}_lag1"}
    return [swap.get(c, c) for c in FEATURES] + ["first_year_data", "age2"]


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
                pred[idx] = m.predict(d.loc[idx, X])
        d["pred"] = pred.clip(lower=0)
        return pl.from_pandas(d[["player_id", "player_display_name", "position", "season", "team", "pred"]]
                              ).rename({"pred": f"pred_ppg_{self.league}"})
