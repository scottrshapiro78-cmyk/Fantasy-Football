"""QB change -> pass-catcher / RB effect, out-of-sample (leave-one-season-out). Events since 2021.
PYTHONPATH=src python3 scripts/study_qb_change.py
Tendency of a QB = share of his team's targets that went to TE (or WR) in his games BEFORE the event (shrunk to league mean).
Event: team's starting QB changes from A to B (A had >=4 games in the pre window, B >=3 games in the post window).
Outcome: player's post PPG minus pre PPG. Feature: delta tendency (B - A) x player's pre share of position targets.
Baseline = no change (0). Skill = reduction in MAE/RMSE of predicted post PPG vs baseline of pre PPG.
"""
import numpy as np, pandas as pd, polars as pl
w = pl.read_parquet("data/processed/player_weeks.parquet").to_pandas()
w = w.sort_values(["season", "week"]).reset_index(drop=True)
w["t"] = w.season * 100 + w.week
qb = w[(w.position == "QB") & (w.attempts >= 15)].sort_values("attempts").groupby(["season", "week", "team"]).tail(1)
qbmap = qb.set_index(["season", "week", "team"])["player_id"]
qbname = qb.set_index("player_id")["player_display_name"].to_dict()
w["qb"] = [qbmap.get((s, k, t)) for s, k, t in zip(w.season, w.week, w.team)]
pc = w[w.position.isin(["WR", "TE", "RB"])].copy()
# team-week targets by position
tw = pc.groupby(["season", "week", "team", "position"]).targets.sum().unstack(fill_value=0).reset_index()
tw["tot"] = tw.RB + tw.TE + tw.WR
tw["qb"] = [qbmap.get((s, k, t)) for s, k, t in zip(tw.season, tw.week, tw.team)]
tw["t"] = tw.season * 100 + tw.week
tw = tw.dropna(subset=["qb"])
qc = w[w.position == "QB"].groupby(["season", "week", "player_id"]).carries.sum().reset_index().rename(columns={"player_id": "qb"})
tw = tw.merge(qc, on=["season", "week", "qb"], how="left").fillna({"carries": 0})
LG = {p: tw[p].sum() / tw.tot.sum() for p in ["TE", "WR", "RB"]}
K = 200.0  # shrink in targets

def tendency(qid, before_t, pos):
    g = tw[(tw.qb == qid) & (tw.t < before_t)]
    return (g[pos].sum() + K * LG[pos]) / (g.tot.sum() + K), g.tot.sum()

def carries_pg(qid, before_t):
    g = tw[(tw.qb == qid) & (tw.t < before_t)]
    return (g.carries.sum() + 3 * 3.0) / (len(g) + 3)  # shrink to ~3/game

rows = []
for (team, season), g in pc[pc.season >= 2021].groupby(["team", "season"]):
    tws = tw[(tw.team == team) & (tw.season == season)].sort_values("t")
    if len(tws) < 7: continue
    # find QB runs
    runs = []
    for q, t in zip(tws.qb, tws.t):
        if runs and runs[-1][0] == q: runs[-1][1].append(t)
        else: runs.append([q, [t]])
    # merge tiny interruptions: only consider consecutive significant runs
    sig = [r for r in runs if len(r[1]) >= 3]
    for a, b in zip(sig[:-1], sig[1:]):
        if a[0] == b[0] or len(a[1]) < 4: continue
        t0 = b[1][0]
        post_t = set(b[1][:6]); pre_t = set(a[1][-8:])
        for pid, pg in g.groupby("player_id"):
            pre = pg[pg.t.isin(pre_t)]; post = pg[pg.t.isin(post_t)]
            if len(pre) < 3 or len(post) < 2: continue
            if pre.fpts_burbs.mean() < 4: continue
            pos = pg.position.iloc[0]
            twa = tws[tws.t.isin(pre_t)]
            share_pos = pre.targets.sum() / max(twa[pos].sum(), 1)  # player's share of his position's targets
            share_team = pre.targets.sum() / max(twa.tot.sum(), 1)
            ta, na = tendency(a[0], t0, pos); tb, nb = tendency(b[0], t0, pos)
            rows.append(dict(season=season, team=team, pid=pid, name=pg.player_display_name.iloc[0], pos=pos,
                             qa=qbname.get(a[0]), qb=qbname.get(b[0]), pre=pre.fpts_burbs.mean(), post=post.fpts_burbs.mean(),
                             n_post=len(post), d_tend=tb - ta, tb=tb, ta=ta, cb=carries_pg(b[0], t0), ca=carries_pg(a[0], t0), share_pos=share_pos, share_team=share_team,
                             d_carry=carries_pg(b[0], t0) - carries_pg(a[0], t0), n_hist=min(na, nb),
                             pre_tgt=pre.targets.mean(), post_tgt=post.targets.mean(), pre_car=pre.carries.mean(), post_car=post.carries.mean()))
ev = pd.DataFrame(rows)
ev["d"] = ev.post - ev.pre
print("events", len(ev), ev.groupby("pos").size().to_dict())
def loso(df, x, y="d", name=""):
    err0, err1 = [], []
    for s in sorted(df.season.unique()):
        tr, te = df[df.season != s], df[df.season == s]
        X = (tr[x] * tr.n_post).sum() / ((tr[x] ** 2) * tr.n_post).sum() if False else None
        b = (tr[x] * tr[y] * tr.n_post).sum() / ((tr[x] ** 2) * tr.n_post).sum()  # through-origin slope
        err0 += list((te[y]).abs()); err1 += list((te[y] - b * te[x]).abs())
    full = (df[x] * df[y] * df.n_post).sum() / ((df[x] ** 2) * df.n_post).sum()
    m0, m1 = np.mean(err0), np.mean(err1)
    r0, r1 = np.sqrt(np.mean(np.square(err0))), np.sqrt(np.mean(np.square(err1)))
    print(f"{name:34s} n={len(df):3d} slope={full:6.2f} MAE {m0:.2f}->{m1:.2f} ({(1-m1/m0)*100:+.1f}%)  RMSE {r0:.2f}->{r1:.2f} ({(1-r1/r0)*100:+.1f}%)")
    return full
for pos in ["TE", "WR"]:
    d = ev[ev.pos == pos].copy()
    d["x"] = d.d_tend * d.share_pos * 100  # points proxy: change in position share x player's slice
    d["x2"] = d.d_tend * d.share_team * 100
    loso(d, "x", name=f"{pos} all: dTend*share_pos")
    big = d[d.d_tend.abs() >= d.d_tend.abs().quantile(0.5)]
    loso(big, "x", name=f"{pos} big dTend (top half)")
    big2 = d[(d.d_tend.abs() >= 0.03)]
    loso(big2, "x", name=f"{pos} |dTend|>=3pts share")
    top = d[d.share_pos >= 0.35]
    loso(top, "x", name=f"{pos} lead pass-catchers (share>=35%)")
rb = ev[ev.pos == "RB"].copy()
rb["x"] = rb.d_carry
loso(rb, "x", name="RB: dQB carries/g")
rbb = rb[rb.d_carry.abs() >= 1.5]
loso(rbb, "x", name="RB: |dQB carries/g|>=1.5")
rbt = ev[ev.pos == "RB"].copy(); rbt["x"] = rbt.d_tend * rbt.share_pos * 100
loso(rbt, "x", name="RB: dTend(RB share)")
ev.to_csv("data/output/qb_change_events.csv", index=False)
print(ev[ev.name.isin(["Mark Andrews"])|(ev.qb=="Jacoby Brissett")][["season","name","pos","qa","qb","pre","post","d_tend","share_pos"]].round(2).to_string())

print("--- restricted: incoming QB above league average ---")
lg_car = tw.carries.mean()
print("league QB carries/game", round(lg_car, 2), "league TE share", round(LG["TE"], 3))
def shift(df, name):
    # simple test: is mean (post-pre) different from 0, and does a fixed shift predict out of sample
    err0, err1 = [], []
    for s_ in sorted(df.season.unique()):
        tr, te = df[df.season != s_], df[df.season == s_]
        m = tr.d.mean(); err0 += list(te.d.abs()); err1 += list((te.d - m).abs())
    m0, m1 = np.mean(err0), np.mean(err1)
    print(f"   {name:30s} mean d={df.d.mean():+.2f} (se {df.d.std()/np.sqrt(len(df)):.2f}) n={len(df)}  fixed-shift MAE {m0:.2f}->{m1:.2f} ({(1-m1/m0)*100:+.1f}%)")
te_ = ev[ev.pos == "TE"].copy(); te_["x"] = te_.d_tend * te_.share_pos * 100
a = te_[te_.tb > LG["TE"]]
loso(a, "x", name="TE: incoming QB TE share > league")
loso(a[a.d_tend > 0], "x", name="TE: incoming TE-heavy AND up vs old")
loso(a[a.share_pos >= 0.5], "x", name="TE: TE-heavy QB, TE1 (>=50% of TE tgts)")
shift(a, "TE: TE-heavy incoming QB")
shift(a[a.d_tend > 0], "TE: TE-heavy and up vs old")
shift(te_[te_.tb <= LG["TE"]], "TE: incoming NOT TE-heavy")
rb_ = ev[ev.pos == "RB"].copy()
r = rb_[rb_.cb > lg_car]
r = r.assign(x=r.d_carry)
loso(r, "x", name="RB: incoming QB carries > league")
loso(r[r.d_carry > 0], "x", name="RB: runner AND more than old QB")
shift(r, "RB: running incoming QB")
shift(r[r.d_carry > 0], "RB: runner and up vs old")
shift(rb_[rb_.cb <= lg_car], "RB: incoming not a runner")
