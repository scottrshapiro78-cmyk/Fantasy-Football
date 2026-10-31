"""What predicts next-season PPG?  Panel OLS (player-clustered SEs), LASSO, and gradient
boosting with a rolling out-of-sample test.  PYTHONPATH=src python3 scripts/predictor_study.py"""
import warnings

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.formula.api as smf
from lightgbm import LGBMRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ffmodel.config import DATA
from ffmodel.features import FEATURES

warnings.filterwarnings("ignore")
POOL = {"QB": 24, "RB": 48, "WR": 60, "TE": 20}
TEST = [2021, 2022, 2023, 2024, 2025]  # target seasons

f = pl.read_parquet(DATA / "processed" / "features.parquet")
s = pl.read_parquet(DATA / "processed" / "player_seasons.parquet").select(
    "player_id", "season", "fpts_burbs", "games")
# baseline: 3-yr recency-weighted PPG through season t
b = f.select("player_id", "season")
for lag, w in [(0, 3), (1, 2), (2, 1)]:
    b = b.join(s.with_columns((pl.col("season") + lag).alias("season"))
                .rename({"fpts_burbs": f"fp{lag}", "games": f"g{lag}"}), on=["player_id", "season"], how="left")
b = b.with_columns(((3 * pl.col("fp0").fill_null(0) + 2 * pl.col("fp1").fill_null(0) + pl.col("fp2").fill_null(0))
                    / (3 * pl.col("g0").fill_null(0) + 2 * pl.col("g1").fill_null(0) + pl.col("g2").fill_null(0)))
                   .alias("baseline"))
f = f.join(b.select("player_id", "season", "baseline"), on=["player_id", "season"])

d = (f.filter(pl.col("next_games") >= 6, pl.col("games") >= 4, pl.col("ppg_burbs") >= 3)
      .to_pandas())
d["first_year_data"] = d["ppg_burbs_lag1"].isna().astype(int)
d["ppg_burbs_lag1"] = d["ppg_burbs_lag1"].fillna(d["ppg_burbs"])
d["games_lag1"] = d["games_lag1"].fillna(0)
d["tough_idx"] = d["tough_idx"].clip(0, 3)
FEATS = FEATURES + ["first_year_data"]
for c in FEATS:
    d[c] = d[c].astype(float)
    d[c] = d[c].fillna(d.groupby("position")[c].transform("median"))
d["age2"] = (d["age_next"] - 27) ** 2
y = "next_ppg_burbs"
out = []

# ---- 1. Panel OLS, focused spec, player-clustered standard errors ------------------------
spec = (f"{y} ~ ppg_burbs + ppg_burbs_lag1 + xfp_pg + fpoe_pg + td_luck_pg + games + age_next + age2"
        " + nfl_year + draft_round + changed_team + team_fp_pg + team_qb_epa + tough_idx + C(season)")
out.append("## 1. Panel OLS (next-season PPG), player-clustered SEs\n")
for pos in POOL:
    p = d[d.position == pos]
    m = smf.ols(spec, p).fit(cov_type="cluster", cov_kwds={"groups": p["player_id"].astype("category").cat.codes})
    t = pd.DataFrame({"coef": m.params, "se": m.bse, "p": m.pvalues}).drop(
        [i for i in m.params.index if i.startswith("C(") or i == "Intercept"])
    out.append(f"### {pos}  (n={len(p)}, R2={m.rsquared:.3f})\n")
    out.append(t.round(3).to_markdown() + "\n")

# ---- 2. LASSO on everything: which predictors survive -------------------------------------
out.append("\n## 2. LASSO (standardized coefficients; 0 = dropped)\n")
lasso_tab = {}
for pos in POOL:
    p = d[d.position == pos]
    X = p[FEATS + ["age2"]]
    pipe = make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=0)).fit(X, p[y])
    lasso_tab[pos] = pd.Series(pipe[-1].coef_, index=X.columns)
lt = pd.DataFrame(lasso_tab).round(2)
lt = lt[(lt != 0).any(axis=1)].sort_values("WR", key=abs, ascending=False)
out.append(lt.to_markdown() + "\n")

# ---- 3. Rolling out-of-sample: baseline vs ridge vs LightGBM ------------------------------
out.append("\n## 3. Out-of-sample, train on seasons before t, predict t (target seasons 2021-2025)\n")
res, imps = [], []
for pos, n in POOL.items():
    p = d[d.position == pos]
    for tgt in TEST:
        tr, te = p[p.season + 1 < tgt], p[p.season + 1 == tgt]
        if len(te) < 5:
            continue
        X = FEATS + ["age2"]
        ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))).fit(tr[X], tr[y])
        gbm = LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
                            random_state=0).fit(tr[X], tr[y])
        te = te.assign(ridge=ridge.predict(te[X]), gbm=gbm.predict(te[X]))
        te = te.assign(blend=(te.ridge + te.gbm) / 2)
        pool = te.sort_values(y, ascending=False).head(n)
        for mname in ["baseline", "ridge", "gbm", "blend"]:
            err = (pool[mname] - pool[y])
            res.append({"position": pos, "target": tgt, "model": mname, "mae": err.abs().mean(),
                        "rank_corr": pool[[mname, y]].corr("spearman").iloc[0, 1]})
        if tgt == TEST[-1]:
            pi = permutation_importance(gbm, te[X], te[y], n_repeats=10, random_state=0,
                                        scoring="neg_mean_absolute_error")
            imps.append(pd.Series(pi.importances_mean, index=X, name=pos))
r = pd.DataFrame(res).groupby(["position", "model"])[["mae", "rank_corr"]].mean().round(3).unstack()
out.append(r.to_markdown() + "\n")
imp = pd.concat(imps, axis=1).round(3)
imp = imp.loc[imp.max(axis=1).sort_values(ascending=False).index].head(15)
out.append("\n## 4. LightGBM permutation importance (2025 target; MAE increase when shuffled)\n")
out.append(imp.to_markdown() + "\n")

# ---- 5. Two classic regressions: what regresses to the mean --------------------------------
out.append("\n## 5. Stickiness: correlation of each stat with itself next season\n")
nxt = f.select("player_id", (pl.col("season") - 1).alias("season"),
               *[pl.col(c).alias(c + "_n") for c in ["xfp_pg", "fpoe_pg", "td_luck_pg", "tgt_share",
                                                      "yds_per_tgt", "yds_per_car", "car_pg", "catch_rate",
                                                      "pass_epa_per_att", "ppg_burbs"]])
j = f.filter(pl.col("games") >= 8).join(nxt, on=["player_id", "season"]).to_pandas()
st = {}
for pos in POOL:
    q = j[j.position == pos]
    st[pos] = {c: q[[c, c + "_n"]].corr().iloc[0, 1] for c in
               ["xfp_pg", "fpoe_pg", "td_luck_pg", "tgt_share", "yds_per_tgt", "yds_per_car", "car_pg",
                "catch_rate", "pass_epa_per_att", "ppg_burbs"]}
out.append(pd.DataFrame(st).round(2).to_markdown() + "\n")

open("docs/predictor_study.md", "w").write("# Predictor study (auto-generated)\n\n" + "\n".join(out))
print("\n".join(out))
