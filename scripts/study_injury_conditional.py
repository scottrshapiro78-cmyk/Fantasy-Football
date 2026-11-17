"""Player-specific injury odds during the season, 2018-2025.   PYTHONPATH=src python3 scripts/study_injury_conditional.py
The playoff sim gives every healthy player the same weekly chance of getting hurt for his position (weeks 9-17).
Does what we know about the player sharpen it?  At-risk row = fantasy-relevant player who PLAYED week w-1 and
whose team plays week w (w = 9..17); event = he does not play week w.  Known before week w:
  tag      - on the injury report in week w-1 (limited / DNP practice or a game status) but played anyway
  back     - he missed week w-2 and returned in w-1
  missed   - he missed 1+ team games earlier this season (before w-2)
  old      - age 29+ (RB) / 30+ (WR, TE) / 33+ (QB)
  heavy    - top third of his position in touches per game so far (RB carries+targets, WR/TE targets, QB n/a)
  lastyr   - missed 4+ games last season
A: weekly odds by group.  B: logistic model fitted on 2018-23, scored on 2024-25 against the flat position rate.
C: rest-of-season view at week 9: share of weeks 10-17 missed, same features."""
import numpy as np, pandas as pd, polars as pl
from sklearn.linear_model import LogisticRegression, LinearRegression
from ffmodel import data
from ffmodel.config import DATA

SS = list(range(2018, 2026))
POS = ["QB", "RB", "WR", "TE"]
MINP = {"QB": 12.0, "RB": 7.0, "WR": 7.0, "TE": 5.0}
OLD = {"QB": 33, "RB": 29, "WR": 30, "TE": 30}
w = pl.read_parquet(DATA / "processed" / "player_weeks.parquet").filter(pl.col("season").is_in(SS + [2017]), pl.col("position").is_in(POS), pl.col("week") <= 17).to_pandas()
w["touch"] = np.where(w.position == "RB", w.carries.fillna(0) + w.targets.fillna(0), w.targets.fillna(0))
ps = pl.read_parquet(DATA / "processed" / "player_seasons.parquet").select("player_id", "season", "games", "birthdate").to_pandas()
sch = data.schedules(SS).filter(pl.col("game_type") == "REG", pl.col("week") <= 17).to_pandas()
tw = pd.concat([sch[["season", "week", "home_team"]].set_axis(["season", "week", "team"], axis=1), sch[["season", "week", "away_team"]].set_axis(["season", "week", "team"], axis=1)])
TW = set(map(tuple, tw.values))
inj = data.injuries(SS).filter(pl.col("game_type") == "REG").to_pandas()
inj["tag"] = inj.report_status.isin(["Questionable", "Doubtful"]) | inj.practice_status.fillna("").str.contains("Limited|Did Not")
TAG = set(map(tuple, inj[inj.tag][["season", "week", "gsis_id"]].values))
lastg = {(r.player_id, r.season + 1): r.games for r in ps.itertuples()}
bd = {r.player_id: r.birthdate for r in ps.dropna(subset=["birthdate"]).itertuples()}

rows = []
for (s, pid), g in w[w.season.isin(SS)].groupby(["season", "player_id"]):
    g = g.sort_values("week"); wk = dict(zip(g.week, zip(g.team, g.fpts_burbs, g.touch))); pos = g.position.iloc[0]
    first = g.week.min()
    for x in range(9, 18):
        if x - 1 not in wk: continue
        team = wk[x - 1][0]
        if (s, x, team) not in TW: continue
        past = [k for k in wk if k < x]
        if len(past) < 4: continue
        ppg = np.mean([wk[k][1] for k in past])
        if ppg < MINP[pos]: continue
        tg = [k for k in range(first, x) if (s, k, wk[min(past, key=lambda z: abs(z - k))][0]) in TW]  # team games since his debut
        miss_weeks = [k for k in tg if k not in wk]
        age = (pd.Timestamp(f"{s}-09-01") - pd.Timestamp(bd[pid])).days / 365.25 if pid in bd else np.nan
        fut = [k for k in range(x, 18) if (s, k, team) in TW]
        rows.append(dict(season=s, pid=pid, pos=pos, week=x, y=int(x not in wk), tag=int((s, x - 1, pid) in TAG),
                         back=int(x - 2 in miss_weeks), missed=int(any(k < x - 2 for k in miss_weeks)),
                         old=int(age >= OLD[pos]) if age == age else 0, tpg=np.mean([wk[k][2] for k in past]),
                         lastyr=int(lastg.get((pid, s), 17) <= 12), ros_miss=np.mean([k not in wk for k in fut]), nfut=len(fut)))
d = pd.DataFrame(rows)
d["heavy"] = 0
for p in ("RB", "WR", "TE"):
    m = d.pos == p
    d.loc[m, "heavy"] = (d[m].tpg >= d[m].groupby("season").tpg.transform(lambda z: z.quantile(2 / 3))).astype(int)
F = ["tag", "back", "missed", "old", "heavy", "lastyr"]
print(f"At-risk player-weeks 2018-2025, weeks 9-17: {len(d)}; missed the next game: {d.y.mean():.1%}")
print("\nA. Chance of missing next week's game, by what we know (n in brackets)")
print(f"{'':28}" + "".join(f"{p:>16}" for p in POS))
def cell(o): return f"{o.y.mean():>8.1%} ({len(o):>5})" if len(o) >= 40 else f"{'n/a':>16}"
print(f"{'everyone':28}" + "".join(cell(d[d.pos == p]) for p in POS))
clean = (d[F[:3]].sum(axis=1) == 0)
print(f"{'no tag, no missed games':28}" + "".join(cell(d[(d.pos == p) & clean]) for p in POS))
for f, nm in zip(F, ["played through injury tag", "just back (missed wk-2)", "missed games earlier", "older", "heavy workload", "missed 4+ last season"]):
    print(f"{nm:28}" + "".join(cell(d[(d.pos == p) & (d[f] == 1)]) for p in POS))
print(f"{'tag + missed games earlier':28}" + "".join(cell(d[(d.pos == p) & (d.tag == 1) & ((d.missed == 1) | (d.back == 1))]) for p in POS))

print("\nB. Predicting next week, fitted 2018-23, scored 2024-25 (Brier score, lower is better)")
tr, te = d[d.season <= 2023], d[d.season >= 2024]
tb = tm = 0
for p in POS:
    a, b = tr[tr.pos == p], te[te.pos == p]
    m = LogisticRegression(C=1.0).fit(a[F], a.y)
    pr = m.predict_proba(b[F])[:, 1]; base = a.y.mean()
    hi = pr >= np.quantile(pr, 0.8); lo = pr <= np.quantile(pr, 0.5)
    print(f"{p}: flat {np.mean((b.y - base) ** 2):.4f}  model {np.mean((b.y - pr) ** 2):.4f} | riskiest fifth: predicted {pr[hi].mean():.1%}, actual {b.y[hi].mean():.1%} | safest half: predicted {pr[lo].mean():.1%}, actual {b.y[lo].mean():.1%} | odds multipliers "
          + ", ".join(f"{f} x{np.exp(c):.2f}" for f, c in zip(F, m.coef_[0])))

print("\nC. Rest of season from week 9: share of weeks 10-17 missed by players who played week 9 (fitted 2018-23, scored 2024-25)")
c = d[(d.week == 10) & (d.nfut >= 6)].copy()   # row for week 10 = played week 9
F2 = ["tag", "back", "missed", "old", "heavy", "lastyr"]
for p in POS:
    a, b = c[(c.pos == p) & (c.season <= 2023)], c[(c.pos == p) & (c.season >= 2024)]
    m = LinearRegression().fit(a[F2], a.ros_miss); pr = np.clip(m.predict(b[F2]), 0, 1)
    al = c[c.pos == p]
    print(f"{p}: n={len(al)}, avg share missed {al.ros_miss.mean():.1%} | clean {al[(al.tag + al.back + al.missed) == 0].ros_miss.mean():.1%} (n={((al.tag + al.back + al.missed) == 0).sum()}) | tagged wk9 {al[al.tag == 1].ros_miss.mean():.1%} (n={(al.tag == 1).sum()}) | missed earlier {al[(al.missed + al.back) > 0].ros_miss.mean():.1%} (n={((al.missed + al.back) > 0).sum()}) | "
          f"test MAE flat {np.mean(np.abs(b.ros_miss - a.ros_miss.mean())):.3f} model {np.mean(np.abs(b.ros_miss - pr)):.3f}; top-third predicted risk actual {b.ros_miss[pr >= np.quantile(pr, 2 / 3)].mean():.1%} vs bottom third {b.ros_miss[pr <= np.quantile(pr, 1 / 3)].mean():.1%}")
