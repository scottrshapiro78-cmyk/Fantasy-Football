"""Do known-in-August team changes predict preseason projection misses? 2021-2026 out of sample
(2026 = games played so far).   PYTHONPATH=src python3 scripts/study_team_changes.py burbs
Flags per team and season t, all knowable before week 1:
  new_hc   - week-1 head coach differs from the week-1 head coach of t-1
  new_qb   - week-1 starting QB differs from the QB with the most starts in t-1
  qb_back  - same QB as last year, but he started 10 or fewer games in t-1 (back from injury)
  ol_turn  - share of last season's offensive line snaps NOT on the week-1 roster (0 = everyone back)
  plus any columns in data/input/coach_changes.csv (season,team,new_oc,new_ol_coach,...), if present.
For each flag: average miss (actual - projected PPG) of that team's drafted-pool players, and whether a
correction fitted on earlier seasons lowers the average miss, alone and on top of the Vegas best case."""
import sys
import numpy as np, pandas as pd, polars as pl
from sklearn.linear_model import LinearRegression
from ffmodel import data
from ffmodel.config import DATA, ROOT
from ffmodel.projection import Projector

lg = sys.argv[1] if len(sys.argv) > 1 else "burbs"
LAST = 2026
f = pl.read_parquet(DATA / "processed" / "features.parquet")
POOL = {"QB": 24, "RB": 50, "WR": 60, "TE": 20}
pcol, ycol = f"pred_ppg_{lg}", f"next_ppg_{lg}"
frames = []
for t in range(2017, LAST + 1):
    pr = Projector(lg).fit(f.filter(pl.col("season") + 1 < t)).predict(f.filter(pl.col("season") == t - 1))
    rows = f.filter(pl.col("season") == t - 1).select("player_id", ycol, "next_games", "team_next", "opening_team_next", "team")
    frames.append(pr.select("player_id", "position", pcol).join(rows, on="player_id").with_columns(pl.lit(t).alias("t")))
d = pl.concat(frames).filter(pl.col("position").is_in(list(POOL))).to_pandas()
d["tm"] = d["opening_team_next"].fillna(d["team_next"]).fillna(d["team"])
d["rk"] = d.groupby(["t", "position"])[pcol].rank(ascending=False, method="first")
d["pool"] = d.rk <= d.position.map(POOL)

ss = list(range(2015, LAST + 1))
sch = data.schedules(ss).filter(pl.col("game_type") == "REG").to_pandas()
g = pd.concat([sch[["season", "week", "home_team", "home_coach", "home_qb_id", "total_line", "spread_line"]].set_axis(["season", "week", "team", "coach", "qb", "tot", "spr"], axis=1),
               sch[["season", "week", "away_team", "away_coach", "away_qb_id", "total_line", "spread_line"]].set_axis(["season", "week", "team", "coach", "qb", "tot", "spr"], axis=1).assign(spr=lambda x: -x.spr)])
g["imp"] = (g.tot + g.spr) / 2
w1 = g.sort_values("week").groupby(["season", "team"]).first().reset_index()[["season", "team", "coach", "qb"]]
played = g[g.qb.notna()]
mainqb = played.groupby(["season", "team", "qb"]).size().reset_index(name="st").sort_values("st").groupby(["season", "team"]).last().reset_index()
tmf = w1.rename(columns={"season": "t", "team": "tm"}).merge(
    w1.assign(t=w1.season + 1).rename(columns={"team": "tm", "coach": "coach0", "qb": "qb0w1"})[["t", "tm", "coach0", "qb0w1"]], on=["t", "tm"]).merge(
    mainqb.assign(t=mainqb.season + 1).rename(columns={"team": "tm", "qb": "qb0", "st": "st0"})[["t", "tm", "qb0", "st0"]], on=["t", "tm"])
tmf["new_hc"] = (tmf.coach != tmf.coach0).astype(float)
tmf["new_qb"] = (tmf.qb != tmf.qb0).astype(float)
qst = played.groupby(["season", "team", "qb"]).size().reset_index(name="st"); qst["t"] = qst.season + 1
tmf = tmf.merge(qst.rename(columns={"team": "tm", "st": "w1qb_prev_st"})[["t", "tm", "qb", "w1qb_prev_st"]], on=["t", "tm", "qb"], how="left")
tmf["qb_back"] = ((tmf.new_qb == 0) & (tmf.w1qb_prev_st <= 10)).astype(float)
tmf.loc[tmf.new_qb == 0, "qb_back"] = (tmf.w1qb_prev_st <= 10).astype(float)
tmf.loc[(tmf.qb == tmf.qb0w1) & (tmf.w1qb_prev_st <= 10), ["qb_back", "new_qb"]] = [1.0, 0.0]   # last year's opener, missed time, back now

nrm = lambda x: x.str.lower().str.replace(r"[^a-z ]", "", regex=True).str.replace(r" (jr|sr|ii|iii|iv)$", "", regex=True).str.strip()
sn = data.snap_counts(ss).filter(pl.col("game_type") == "REG", pl.col("position").is_in(["T", "G", "C", "OL"])).to_pandas()
sn["nm"] = nrm(sn.player)
ol = sn.groupby(["season", "team", "nm"]).offense_snaps.sum().reset_index()
ro = data.rosters_weekly(ss).filter(pl.col("game_type") == "REG").select("season", "team", "week", "full_name").to_pandas()
ro = ro[ro.week == ro.groupby(["season", "team"]).week.transform("min")]
ro1 = ro.assign(nm=nrm(ro.full_name))[["season", "team", "nm"]].drop_duplicates().assign(on=1.0)
ol["t"] = ol.season + 1
ol = ol.merge(ro1.rename(columns={"season": "t"}), on=["t", "team", "nm"], how="left").fillna({"on": 0.0})
olt = ol.groupby(["t", "team"]).apply(lambda x: 1 - (x.offense_snaps * x.on).sum() / x.offense_snaps.sum()).reset_index(name="ol_turn").rename(columns={"team": "tm"})
tmf = tmf.merge(olt, on=["t", "tm"], how="left")
tmf["ol_turn_raw"] = tmf.ol_turn
tmf["ol_turn"] = tmf.ol_turn - tmf.groupby("t").ol_turn.transform("mean")
tmf["ol_rebuilt"] = (tmf.ol_turn_raw >= 0.40).astype(float)
imp = g[g.imp.notna()].groupby(["season", "team"]).imp.mean().reset_index()
imp["vegas_now"] = imp.imp - imp.groupby("season").imp.transform("mean")
tmf = tmf.merge(imp.rename(columns={"season": "t", "team": "tm"})[["t", "tm", "vegas_now"]], on=["t", "tm"], how="left")
FLAGS = ["new_hc", "new_qb", "qb_back", "ol_turn", "ol_rebuilt"]
extra = ROOT / "data" / "input" / "coach_changes.csv"
if extra.exists():
    e = pd.read_csv(extra).rename(columns={"season": "t", "team": "tm"})
    ec = [c for c in e.columns if c not in ("t", "tm", "note")]
    tmf = tmf.merge(e[["t", "tm"] + ec], on=["t", "tm"], how="left"); FLAGS += ec
    for c in ec: tmf[c] = tmf[c].fillna(0.0)
d = d.merge(tmf[["t", "tm", "vegas_now", "ol_turn_raw"] + FLAGS], on=["t", "tm"], how="left")
for c in FLAGS + ["vegas_now"]: d[c] = d[c].fillna(0.0)
ok = d[(d.next_games >= 6) & d[ycol].notna()].copy()
ok["res"] = ok[ycol] - ok[pcol]

tt = tmf[tmf.t.between(2021, LAST)]
print(f"Team-seasons 2021-{LAST}: {len(tt)}.  " + ", ".join(f"{c}: {int(tt[c].sum())}" for c in FLAGS if c != "ol_turn")
      + f".  Line turnover averages {tt.ol_turn_raw.mean():.0%} (10th-90th pct {tt.ol_turn_raw.quantile(.1):.0%}-{tt.ol_turn_raw.quantile(.9):.0%})")

print(f"\nA. Average miss (actual - projected PPG, {lg}) of drafted-pool players, 2021-{LAST}. +/- is one standard error (clustered by team-season).")
def cell(o):
    if len(o) < 8: return f"{'n/a':>18}"
    tm = o.groupby(["t", "tm"]).res.mean()
    return f"{o.res.mean():>+7.2f} ±{tm.std() / np.sqrt(len(tm)):.2f} n={len(o):<4}"
print(f"{'flag':12}" + "".join(f"{p:>20}" for p in POOL))
pp = ok[ok.pool & (ok.t >= 2021)]
print(f"{'everyone':12}" + "".join(f"{cell(pp[pp.position == p]):>20}" for p in POOL))
for c in FLAGS:
    if c == "ol_turn":
        sel = [("line: top third turnover", pp[pp.ol_turn_raw >= tt.ol_turn_raw.quantile(2 / 3)]), ("line: bottom third", pp[pp.ol_turn_raw <= tt.ol_turn_raw.quantile(1 / 3)])]
    else:
        sel = [(c, pp[pp[c] == 1])]
    for nm, o in sel:
        print(f"{nm[:24]:24}"[:24].ljust(12) + "".join(f"{cell(o[o.position == p]):>20}" for p in POOL) + f"   <- {nm}")

print(f"\nB. Does a correction fitted on earlier seasons lower the average miss? Drafted pool, 2021-{LAST} out of sample (PPG)")
print(f"{'pos':4}{'n':>5}{'model':>8}" + "".join(f"{'+' + c:>13}" for c in FLAGS) + f"{'+all flags':>12}{'vegas best':>12}{'vegas+all':>11}")
tot = {}
for p in POOL:
    o = ok[ok.position == p]
    res = {}
    for t in range(2021, LAST + 1):
        tr, te = o[o.t < t], o[(o.t == t) & o.pool]
        res.setdefault("model", []).extend(np.abs(te.res))
        for c in FLAGS:
            b = LinearRegression(fit_intercept=False).fit(tr[[c]], tr.res).coef_[0]
            res.setdefault(c, []).extend(np.abs(te.res - b * te[c]))
        fl = [c for c in FLAGS if c != "ol_rebuilt"]
        m = LinearRegression(fit_intercept=False).fit(tr[fl], tr.res)
        res.setdefault("all", []).extend(np.abs(te.res - m.predict(te[fl])))
        b = LinearRegression(fit_intercept=False).fit(tr[["vegas_now"]], tr.res)
        res.setdefault("vegas", []).extend(np.abs(te.res - b.predict(te[["vegas_now"]])))
        m = LinearRegression(fit_intercept=False).fit(tr[fl + ["vegas_now"]], tr.res)
        res.setdefault("vegas_all", []).extend(np.abs(te.res - m.predict(te[fl + ["vegas_now"]])))
    for k, v in res.items(): tot.setdefault(k, []).extend(v)
    print(f"{p:4}{len(res['model']):>5}{np.mean(res['model']):>8.2f}" + "".join(f"{np.mean(res[c]):>13.2f}" for c in FLAGS) + f"{np.mean(res['all']):>12.2f}{np.mean(res['vegas']):>12.2f}{np.mean(res['vegas_all']):>11.2f}")
print(f"{'ALL':4}{len(tot['model']):>5}{np.mean(tot['model']):>8.2f}" + "".join(f"{np.mean(tot[c]):>13.2f}" for c in FLAGS) + f"{np.mean(tot['all']):>12.2f}{np.mean(tot['vegas']):>12.2f}{np.mean(tot['vegas_all']):>11.2f}")
tmf[tmf.t >= 2021][["t", "tm", "coach", "new_hc", "new_qb", "qb_back", "ol_turn_raw"]].to_csv(DATA / "processed" / "team_changes.csv", index=False)
