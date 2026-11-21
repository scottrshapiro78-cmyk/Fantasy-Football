"""Nov 19 2026: what the 2006+ data says. Run with FF_DATA=data_2006 FF_FIRST_SEASON=2006.

1) In-season (rest of season after week 10), every position, 2015-25 model backtest: refit the model's own weights
   (k/u/r/s grids) leave-one-season-out with and without an extra term: snap share, and targets per game for
   RB/WR/TE (carries+targets for RB). Also: the production weights (fitted 2021-25) vs weights refit on 2015-25.
2) Preseason (next-season PPG), graded on 2019-25 drafted-size pools: ridge trained on 2012+ (production) vs 2006+,
   and LightGBM on 2006+. Rolling origin: each target season trained only on earlier seasons.
Average miss (MAE), Burbs scoring.
"""
import itertools
import json

import numpy as np
import pandas as pd
import polars as pl

from ffmodel import data
from ffmodel.config import DATA
from ffmodel.projection import Projector, league_features, prepare, HISTORY_WEIGHT
from ffmodel.ros import K_GRID, U_GRID, R_GRID, S_GRID

POS4 = ["QB", "RB", "WR", "TE"]
# ---------- 1) in-season ----------
pw = pl.read_parquet(DATA / "processed" / "player_weeks.parquet").filter(pl.col("week") <= 17)
pre = pw.filter(pl.col("week") <= 10).group_by("player_id", "season").agg(
    pl.col("targets").fill_null(0).mean().alias("tpg"), (pl.col("targets").fill_null(0) + pl.col("carries").fill_null(0)).mean().alias("opp"))
post = pw.filter(pl.col("week") >= 11).group_by("player_id", "season").agg(pl.len().alias("g2"), pl.col("fpts_burbs").mean().alias("after"))
ps = pl.read_parquet(DATA / "processed" / "player_seasons.parquet").select("player_id", "season", "pfr_id").drop_nulls()
sn = (data.snap_counts(list(range(2012, 2026))).filter(pl.col("week") <= 10, pl.col("game_type") == "REG")
      .with_columns(pl.col("season").cast(pl.Int32))
      .join(ps.with_columns(pl.col("season").cast(pl.Int32)), left_on=["pfr_player_id", "season"], right_on=["pfr_id", "season"])
      .group_by("player_id", "season").agg(pl.col("offense_pct").cast(pl.Float64).mean().alias("snap")))
h = pl.concat([pl.read_parquet(f) for f in sorted((DATA / "processed").glob("ros_history_wk11_long*.parquet"))], how="diagonal_relaxed")
h = (h.with_columns(pl.col("season").cast(pl.Int32)).join(pre.with_columns(pl.col("season").cast(pl.Int32)), on=["player_id", "season"], how="left")
     .join(post.with_columns(pl.col("season").cast(pl.Int32)), on=["player_id", "season"], how="left")
     .join(sn, on=["player_id", "season"], how="left"))
fit = json.load(open(DATA / "processed" / "ros_fit.json"))
MIN_PRI = {"QB": 12, "RB": 6, "WR": 6, "TE": 5}  # same relevant pool as ros.fit


def preds(t, k, u, r, s_, w, col):
    so = t["ppg_burbs_sofar"].to_numpy() - s_ * t["sos_faced_burbs"].fill_null(0).to_numpy()
    n = t["n"].to_numpy()
    x = (t[col].to_numpy() - t[col].mean()) if col else 0.0
    inner = so + u * t["luck_pg"].fill_null(0).to_numpy() + r * t["role_trend"].fill_null(0).to_numpy() + w * x
    return (k * t["pred_ppg_burbs"].to_numpy() + n * inner) / (k + n) + s_ * t["sos_left_burbs"].fill_null(0).to_numpy()


def loso(c, ws, col):
    errs, chosen = [], []
    for s in sorted(c["season"].unique()):
        tr, te = c.filter(pl.col("season") != s), c.filter(pl.col("season") == s)
        y = tr["after"].to_numpy()
        g = min(itertools.product(K_GRID, U_GRID, R_GRID, S_GRID, ws), key=lambda g: float(np.mean(np.abs(preds(tr, *g, col) - y))))
        chosen.append(g)
        errs.append(np.abs(preds(te, *g, col) - te["after"].to_numpy()))
    return float(np.concatenate(errs).mean()), chosen


print("1) IN-SEASON, rest of season after week 10, model backtest 2015-25")
for p in POS4:
    c = h.filter(pl.col("position") == p, pl.col("n") >= 5, pl.col("g2") >= 3, pl.col("pred_ppg_burbs") >= MIN_PRI[p])
    c = c.with_columns(pl.col("snap").fill_null(pl.col("snap").median()))
    prod = tuple(fit[x]["burbs"][p] for x in ("k", "u", "r", "s"))
    e_prod = float(np.mean(np.abs(preds(c, *prod, 0.0, None) - c["after"].to_numpy())))
    e0, ch = loso(c, [0.0], None)
    line = f"  {p} n={c.height}: production weights {e_prod:.3f} | refit on 2015-25 {e0:.3f} ({100 * (e0 / e_prod - 1):+.1f}%)"
    extras = [("snap", [0, 2, 4, 6, 8, 12])] + ([("tpg", [0, .1, .2, .3, .5])] if p in ("WR", "TE") else []) + ([("opp", [0, .05, .1, .2, .3])] if p == "RB" else [])
    for col, grid in extras:
        e1, ch1 = loso(c, [float(w) for w in grid], col)
        line += f" | +{col} {e1:.3f} ({100 * (e1 / e0 - 1):+.1f}%)"
    full = min(itertools.product(K_GRID, U_GRID, R_GRID, S_GRID, [0.0]), key=lambda g: float(np.mean(np.abs(preds(c, *g, None) - c["after"].to_numpy()))))
    print(line); print(f"     weights k,u,r,s production {prod} vs fitted on 2015-25 {full[:4]}")

# ---------- 2) preseason ----------
print("2) PRESEASON next-season PPG, graded 2019-25 (players with 6+ games next season)")
f = pl.read_parquet(DATA / "processed" / "features.parquet")
X = league_features("burbs")
POOL = {"QB": 24, "RB": 48, "WR": 60}


def grade(make, first):
    rows = []
    for t in range(2019, 2026):
        tr = f.filter(pl.col("season") + 1 < t, pl.col("season") >= first)
        te = f.filter(pl.col("season") == t - 1, pl.col("next_games") >= 6)
        p_ = make(tr, te).to_pandas()
        p_ = p_.merge(te.select("player_id", "next_ppg_burbs").to_pandas(), on="player_id")
        for pos, n in POOL.items():
            q = p_[p_.position == pos].nlargest(n, "pred")
            rows.append((pos, t, np.abs(q.pred - q.next_ppg_burbs).mean()))
    r = pd.DataFrame(rows, columns=["pos", "t", "mae"])
    return r.groupby("pos").mae.mean()


def ridge(tr, te):
    return Projector("burbs").fit(tr).predict(te).rename({"pred_ppg_burbs": "pred"})


def lgbm(tr, te):
    import lightgbm as lgb
    t = tr.filter(pl.col("next_games") >= 6, pl.col("games") >= 4, pl.col("ppg_burbs") >= 3)
    d, med = prepare(t, "burbs")
    e, _ = prepare(te, "burbs", med)
    e["pred"] = e["base_burbs"].astype(float)
    for pos in POOL:
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=1)
        m.fit(d[d.position == pos][X], d[d.position == pos]["next_ppg_burbs"])
        idx = e.position == pos
        w = HISTORY_WEIGHT.get(pos, 0.0)
        e.loc[idx, "pred"] = (1 - w) * m.predict(e.loc[idx, X]) + w * e.loc[idx, "base_burbs"]
    return pl.from_pandas(e[["player_id", "position", "pred"]])


res = pd.DataFrame({"ridge 2012+ (production)": grade(ridge, 2012), "ridge 2006+": grade(ridge, 2006),
                    "LightGBM 2006+": grade(lgbm, 2006)})
print(res.round(3).to_string())
