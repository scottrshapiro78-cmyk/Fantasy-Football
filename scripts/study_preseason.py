"""Two preseason-projection tests, 2021-2025 out of sample (train on seasons before t, predict t).
    PYTHONPATH=src python3 scripts/study_preseason.py burbs
1. Team-down: does a team scoring forecast improve player projections? Team signals tried:
     prev_imp  - the team's average Vegas implied points last season (known in August)
     prev_fp   - the team's skill-player fantasy points per game last season (known in August)
     vegas_now - the team's average implied points in the season being predicted (NOT known in August: game
                 lines are set week by week. It stands in for preseason win totals as a best case.)
   Correction = b_pos x (signal - league average), b fitted on the residuals of earlier seasons.
2. Ranges: can we tell in advance who is boom-or-bust? Player-specific spread fitted on earlier seasons
   (absolute miss ~ projection level, age, experience, team change, games missed last year), 80% range =
   projection +/- spread; checked for coverage and against a one-size range per position."""
import sys
import numpy as np, pandas as pd, polars as pl
from sklearn.linear_model import LinearRegression, RidgeCV
from ffmodel import data
from ffmodel.config import DATA
from ffmodel.projection import Projector

lg = sys.argv[1] if len(sys.argv) > 1 else "burbs"
f = pl.read_parquet(DATA / "processed" / "features.parquet")
POOL = {"QB": 24, "RB": 50, "WR": 60, "TE": 20}
pcol, ycol = f"pred_ppg_{lg}", f"next_ppg_{lg}"

# engine predictions for 2017..2025, each trained only on earlier seasons
frames = []
for t in range(2017, 2026):
    pr = Projector(lg).fit(f.filter(pl.col("season") + 1 < t)).predict(f.filter(pl.col("season") == t - 1))
    rows = f.filter(pl.col("season") == t - 1).select("player_id", ycol, "next_games", "team_next", "opening_team_next", "team",
                                                     "age_next", "nfl_year", "changed_team", "games", "games_lag1", "td_luck_pg", "fpoe_pg")
    frames.append(pr.select("player_id", "position", pcol).join(rows, on="player_id").with_columns(pl.lit(t).alias("t")))
d = pl.concat(frames).filter(pl.col("position").is_in(list(POOL))).to_pandas()
d["tm"] = d["opening_team_next"].fillna(d["team_next"]).fillna(d["team"])
d["rk"] = d.groupby(["t", "position"])[pcol].rank(ascending=False, method="first")
d["pool"] = d.rk <= d.position.map(POOL)

# team signals
sch = data.schedules(list(range(2015, 2026))).filter(pl.col("game_type") == "REG", pl.col("total_line").is_not_null()).to_pandas()
h = pd.DataFrame({"season": sch.season, "team": sch.home_team, "imp": (sch.total_line + sch.spread_line) / 2})
a = pd.DataFrame({"season": sch.season, "team": sch.away_team, "imp": (sch.total_line - sch.spread_line) / 2})
imp = pd.concat([h, a]).groupby(["season", "team"]).imp.mean().reset_index()
imp["z"] = imp.imp - imp.groupby("season").imp.transform("mean")
tf = f.select("season", "team", "team_fp_pg").unique(["season", "team"]).to_pandas().dropna()
tf["zfp"] = tf.team_fp_pg - tf.groupby("season").team_fp_pg.transform("mean")
d = d.merge(imp.rename(columns={"season": "t", "team": "tm", "z": "vegas_now"})[["t", "tm", "vegas_now"]], on=["t", "tm"], how="left")
prev = imp.copy(); prev["t"] = prev.season + 1
d = d.merge(prev.rename(columns={"team": "tm", "z": "prev_imp"})[["t", "tm", "prev_imp"]], on=["t", "tm"], how="left")
tf["t"] = tf.season + 1
d = d.merge(tf.rename(columns={"team": "tm", "zfp": "prev_fp"})[["t", "tm", "prev_fp"]], on=["t", "tm"], how="left")
for c in ("vegas_now", "prev_imp", "prev_fp"):
    d[c] = d[c].fillna(0.0)
ok = d[(d.next_games >= 6) & d[ycol].notna()].copy()
ok["res"] = ok[ycol] - ok[pcol]

print(f"TEST 1: team-down correction ({lg}). Average miss in PPG, drafted pool, 2021-2025 out of sample")
print(f"{'pos':4}{'n':>5}{'model':>8}{'+prev_imp':>11}{'+prev_fp':>10}{'+vegas_now (best case)':>24}   slope per implied point (vegas_now), by test season")
tot = {k: [0.0, 0] for k in ("model", "prev_imp", "prev_fp", "vegas_now")}
for p in POOL:
    o = ok[ok.position == p]
    res = {k: [] for k in tot}; slopes = []
    for t in range(2021, 2026):
        tr, te = o[o.t < t], o[(o.t == t) & o.pool]
        res["model"] += list(np.abs(te.res))
        for sig in ("prev_imp", "prev_fp", "vegas_now"):
            b = LinearRegression(fit_intercept=False).fit(tr[[sig]], tr.res).coef_[0]
            res[sig] += list(np.abs(te.res - b * te[sig]))
            if sig == "vegas_now": slopes.append(round(float(b), 2))
    for k in tot:
        tot[k][0] += sum(res[k]); tot[k][1] += len(res[k])
    print(f"{p:4}{len(res['model']):>5}{np.mean(res['model']):>8.2f}{np.mean(res['prev_imp']):>11.2f}{np.mean(res['prev_fp']):>10.2f}{np.mean(res['vegas_now']):>24.2f}   {slopes}")
print(f"{'ALL':4}{tot['model'][1]:>5}" + "".join(f"{tot[k][0] / tot[k][1]:>{w}.2f}" for k, w in zip(tot, (8, 11, 10, 24))))
sp = imp.groupby("season").imp.std().mean()
print(f"(teams differ by about {sp:.1f} implied points per game, one standard deviation)")

print(f"\nTEST 2: ranges ({lg}). 80% range per player, drafted pool, 2021-2025 out of sample")
X = ["lvl", "age", "exp", "chg", "missed", "luck"]
ok["lvl"], ok["age"], ok["exp"] = ok[pcol], ok.age_next.fillna(27), ok.nfl_year.fillna(4).clip(1, 10)
ok["chg"], ok["missed"], ok["luck"] = ok.changed_team.fillna(0).astype(float), (17 - ok.games.fillna(17)).clip(0, 17), ok.td_luck_pg.fillna(0).abs()
print(f"{'pos':4}{'n':>5}{'in range: player-specific':>27}{'one-size':>10}{'avg half-width':>16}{'narrowest third':>17}{'widest third':>14}{'actual miss: narrow / wide third':>34}{'corr(width, miss)':>19}")
for p in POOL:
    o = ok[ok.position == p]
    cov_s, cov_c, wid, miss, n = [], [], [], [], 0
    for t in range(2021, 2026):
        tr, te = o[o.t < t], o[(o.t == t) & o.pool]
        m = RidgeCV(alphas=np.logspace(-1, 3, 10)).fit(tr[X], np.abs(tr.res))
        s_tr = np.clip(m.predict(tr[X]), 0.5, None)
        c = np.quantile(np.abs(tr.res) / s_tr, 0.8)            # multiplier so 80% of past misses fall inside
        w = c * np.clip(m.predict(te[X]), 0.5, None)
        w0 = np.quantile(np.abs(tr[tr.rk <= POOL[p] * 1.5].res), 0.8)
        cov_s += list(np.abs(te.res) <= w); cov_c += list(np.abs(te.res) <= w0); wid += list(w); miss += list(np.abs(te.res))
    wid, miss = np.array(wid), np.array(miss)
    lo, hi = np.quantile(wid, [1 / 3, 2 / 3])
    print(f"{p:4}{len(wid):>5}{np.mean(cov_s) * 100:>26.0f}%{np.mean(cov_c) * 100:>9.0f}%{wid.mean():>16.1f}{wid[wid <= lo].mean():>17.1f}{wid[wid >= hi].mean():>14.1f}"
          f"{miss[wid <= lo].mean():>20.2f} / {miss[wid >= hi].mean():.2f}{'':>6}{pd.Series(wid).corr(pd.Series(miss), method='spearman'):>13.2f}")
    if p == "RB":
        co = RidgeCV(alphas=np.logspace(-1, 3, 10)).fit(o[o.t < 2025][X], np.abs(o[o.t < 2025].res))
        print("     RB spread drivers (PPG of extra miss per unit):", dict(zip(X, np.round(co.coef_, 2))))
