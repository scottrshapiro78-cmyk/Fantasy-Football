"""QB QUALITY adjustment for RB/WR/TE rest-of-season projections (Nov 21 2026, Scott's idea).  Rule 14 pre-registration:

Hypothesis: a pass catcher's past points are inflated/deflated by how good his QBs were. Restating them to an average QB
and re-applying the QB he will have should cut the ROS miss. (Different from the rejected Nov 18 test, which used the QB's
TE/WR target TENDENCY, not his quality.)
QB quality Q = QB's passing fantasy points per attempt (0.04/yd, 4/TD, -2/INT), from all his games BEFORE the cut
(career, back to 2006), shrunk to the league rate with 150 attempts; expressed relative to league (Q/league - 1).
Cut = the model's own week-11 ROS projection (data_2006 ros_history_wk11_long*: 2015-2025; the model's prior needs the
seasons before, so this uses the 2006+ data). Players: RB/WR/TE, 5+ games before, 3+ after.
Variants (fixed before looking):
  V1  pred x (1 + b_pos x (Q_now - Q_this))       Q_now = starter of his team's latest game; Q_this = avg Q of his QBs this season
  V2  V1 + c_pos x (Q_now - Q_last)               Q_last = avg Q of his QBs last season (restating last year's stats)
  V3  ORACLE: Q of the QBs he actually had after the cut (upper bound only, not adoptable)
  Both run with QB ratings built from 2006+ games and from 2012+ games only.
Metric: MAE of ppg_after (Burbs scoring) vs the model's pred (baseline). Fit leave-one-season-out on 2015-2024; 2025 held out
(fit on 2015-2024, scored once). Rule 5: adopt only if the miss drops 2%+.
FF_DATA=data_2006 FF_FIRST_SEASON=2006 PYTHONPATH=src python3 scripts/study_qb_quality.py
"""
import numpy as np, pandas as pd, polars as pl
from ffmodel.config import DATA

CUT = 11
w = pl.read_parquet(DATA / "processed" / "player_weeks.parquet").filter(pl.col("week") <= 18).to_pandas()
q = w[(w.position == "QB") & (w.attempts.fillna(0) >= 10)].copy()
q["pp"] = 0.04 * q.c_pass_yd.fillna(0) + 4 * q.c_pass_td.fillna(0) - 2 * q.c_pass_int.fillna(0)
starter = q.sort_values("attempts").groupby(["season", "week", "team"]).tail(1)[["season", "week", "team", "player_id"]].rename(columns={"player_id": "qb"})
q["t"] = q.season * 100 + q.week

def qb_rating(first):
    """Q for every (qb, season) at the cut: career-to-date through week CUT of that season."""
    qq = q[q.season >= first].sort_values("t")
    out = {}
    for qid, g in qq.groupby("player_id"):
        for s in g.season.unique():
            h = g[g.t <= s * 100 + CUT]
            out[(qid, s)] = (h.pp.sum(), h.attempts.sum())
    lg = qq[qq.week <= CUT].groupby("season").apply(lambda g: g.pp.sum() / g.attempts.sum()).to_dict()
    K = 150
    def Q(qid, s):
        p, a = out.get((qid, s), (0.0, 0.0))
        L = lg.get(s, np.mean(list(lg.values())))
        return ((p + K * L) / (a + K)) / L - 1
    return Q

h = pl.concat([pl.read_parquet(f) for f in sorted((DATA / "processed").glob("ros_history_wk11_long*.parquet"))], how="diagonal_relaxed").to_pandas()
h = h[h.position.isin(["RB", "WR", "TE"]) & (h.n >= 5) & (h.n_after >= 3)].copy()
pg = w[w.position.isin(["RB", "WR", "TE"])][["player_id", "season", "week", "team"]].merge(starter, on=["season", "week", "team"], how="left")

def features(Q):
    rows = []
    games = {k: g for k, g in pg.groupby(["player_id", "season"])}
    last_qb = starter[starter.week <= CUT].sort_values("week").groupby(["season", "team"]).qb.last().to_dict()
    for r in h.itertuples():
        g = games.get((r.player_id, r.season))
        if g is None: rows.append((np.nan,) * 4); continue
        pre, post = g[g.week <= CUT].dropna(subset=["qb"]), g[g.week > CUT].dropna(subset=["qb"])
        q_this = np.mean([Q(x, r.season) for x in pre.qb]) if len(pre) else np.nan
        cur = last_qb.get((r.season, r.team_now))
        q_now = Q(cur, r.season) if cur else np.nan
        gl = games.get((r.player_id, r.season - 1))
        q_last = np.mean([Q(x, r.season - 1) for x in gl.dropna(subset=["qb"]).qb]) if gl is not None and gl.qb.notna().any() else q_this
        q_fut = np.mean([Q(x, r.season) for x in post.qb]) if len(post) else q_now
        rows.append((q_this, q_now, q_last, q_fut))
    f = pd.DataFrame(rows, columns=["q_this", "q_now", "q_last", "q_fut"], index=h.index)
    return f

def fit_b(d, cols):
    """least squares on the residual ratio: (actual - pred) = pred * sum b_i x_i, per position"""
    out = {}
    for p, g in d.groupby("position"):
        X = np.column_stack([g.pred * g[c] for c in cols]); y = g.act - g.pred
        out[p] = np.linalg.lstsq(X, y, rcond=None)[0]
    return out

def apply(d, cols, b):
    adj = np.zeros(len(d))
    for i, c in enumerate(cols):
        adj += d.pred.values * d[c].values * d.position.map(lambda p: b[p][i]).values
    return d.pred.values + adj

for first in (2006, 2012):
    Q = qb_rating(first)
    f = features(Q)
    d = pd.concat([h[["player_id", "season", "position"]], f], axis=1)
    d["pred"], d["act"] = h.pred_ppg_burbs, h.ppg_burbs_after
    d = d.dropna()
    d["x1"] = d.q_now - d.q_this; d["x2"] = d.q_now - d.q_last; d["x3"] = d.q_fut - d.q_this
    print(f"\n=== QB ratings from {first}+ games; {len(d)} player-seasons 2015-2025; QB changed (|x1|>.05): {(d.x1.abs() > .05).mean():.0%}")
    dev = d[d.season <= 2024]
    for name, cols in [("V1 now vs this season", ["x1"]), ("V2 + last season", ["x1", "x2"]), ("V3 oracle (not adoptable)", ["x3"])]:
        p = pd.Series(np.nan, index=dev.index)
        for s in sorted(dev.season.unique()):
            tr, te = dev[dev.season != s], dev[dev.season == s]
            p[te.index] = apply(te, cols, fit_b(tr, cols))
        base = (dev.act - dev.pred).abs().mean(); new = (dev.act - p).abs().mean()
        byseason = dev.assign(e0=(dev.act - dev.pred).abs(), e1=(dev.act - p).abs()).groupby("season")[["e0", "e1"]].mean()
        wins = int((byseason.e1 < byseason.e0).sum())
        ch = dev.x1.abs() > .05
        sub = f"; QB-changed players: {((dev.act - dev.pred).abs()[ch].mean()):.3f} -> {((dev.act - p).abs()[ch].mean()):.3f} (n={ch.sum()})"
        ho = d[d.season == 2025]; b = fit_b(dev, cols); ph = apply(ho, cols, b)
        hb, hn = (ho.act - ho.pred).abs().mean(), np.abs(ho.act - ph).mean()
        print(f"{name}: LOSO MAE {base:.3f} -> {new:.3f} ({(base - new) / base:+.1%}), seasons better {wins}/{len(byseason)}{sub}; "
              f"2025 holdout {hb:.3f} -> {hn:.3f} ({(hb - hn) / hb:+.1%}); b={ {k: np.round(v, 2).tolist() for k, v in b.items()} }")
