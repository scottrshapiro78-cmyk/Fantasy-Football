"""Will he play?  Chance a player on the injury report plays that week, 2018-2025.
PYTHONPATH=src python3 scripts/study_play_given_status.py [week=11]   (also scores Scott's Burbs roster for that week)

Row = fantasy-relevant player (QB/RB/WR/TE, top of his team's position by touches over his last 6 games or season PPG)
on the week-w injury report (final report: game status and/or practice status) whose team plays week w.
Event = he appears in the week-w game (a row in weekly stats).  Features known before kickoff:
  status    - game status (Questionable / Doubtful / Out / none) and last practice (DNP / Limited / Full)
  inj       - body part (knee, ankle, hamstring, ...)  (grouped)
  hist      - weeks on the report over the previous 8 team games, and games missed in the previous 8
  streak    - weeks in a row on the report with this same body part
  team_win  - team win % entering the week; alive = within 1.5 games of a playoff spot proxy (win% >= .4 or week < 12)
  week, pos
Evaluation: leave-one-season-out logistic regression; Brier score against the plain status table (what Q/D/Out rates give).
"""
import sys
import numpy as np, pandas as pd, polars as pl
from sklearn.linear_model import LogisticRegression
from ffmodel import data
from ffmodel.config import DATA

WK = int(sys.argv[1]) if len(sys.argv) > 1 else 11
SS = list(range(2018, 2027))
POS = ["QB", "RB", "WR", "TE"]
pw = pl.read_parquet(DATA / "processed" / "player_weeks.parquet").filter(pl.col("season").is_in(SS), pl.col("position").is_in(POS)).to_pandas()
pw["touch"] = np.where(pw.position == "RB", pw.carries.fillna(0) + pw.targets.fillna(0), pw.targets.fillna(0) + pw.attempts.fillna(0) * 0.3)
sch = data.schedules(SS).filter(pl.col("game_type") == "REG").to_pandas()
home = sch[["season", "week", "home_team", "home_score", "away_score"]].set_axis(["season", "week", "team", "pf", "pa"], axis=1)
away = sch[["season", "week", "away_team", "away_score", "home_score"]].set_axis(["season", "week", "team", "pf", "pa"], axis=1)
tw = pd.concat([home, away]).sort_values(["season", "team", "week"])
tw["win"] = np.where(tw.pf.isna(), np.nan, (tw.pf > tw.pa) + 0.5 * (tw.pf == tw.pa))
tw["gp"] = tw.groupby(["season", "team"]).win.transform(lambda s: s.shift().notna().cumsum())
tw["w_before"] = tw.groupby(["season", "team"]).win.transform(lambda s: s.shift().cumsum())
tw["team_win"] = (tw.w_before / tw.gp).where(tw.gp > 0, 0.5)
played = set(map(tuple, pw[["season", "week", "player_id"]].values))
inj = data.injuries(SS).filter(pl.col("game_type") == "REG", pl.col("position").is_in(POS)).to_pandas().rename(columns={"gsis_id": "player_id"})
inj = inj.drop_duplicates(["season", "week", "player_id"], keep="last")
inj["gs"] = inj.report_status.fillna("none").where(inj.report_status.isin(["Questionable", "Doubtful", "Out"]), "none")
inj["pr"] = inj.practice_status.fillna("").map(lambda s: "DNP" if s.startswith("Did Not") else "Lim" if s.startswith("Limited") else "Full" if s.startswith("Full") else "none")
BP = {"knee": "knee", "ankle": "ankle", "hamstring": "hamstring", "foot": "foot", "groin": "groin", "shoulder": "shoulder", "concussion": "head", "head": "head", "calf": "calf", "back": "back", "quad": "quad", "rib": "torso", "chest": "torso", "hip": "hip", "thigh": "quad"}
def bp(s):
    s = str(s).lower()
    return next((v for k, v in BP.items() if k in s), "other")
inj["inj"] = inj.report_primary_injury.fillna(inj.practice_primary_injury).map(bp)
# who counts as fantasy relevant: top-N touches among his team/position in the 6 weeks before (or season-to-date)
pw = pw.sort_values(["player_id", "season", "week"])
pw["t6"] = pw.groupby(["player_id", "season"]).touch.transform(lambda s: s.rolling(6, min_periods=1).mean())
# games on the report / missed over previous 8 team weeks
tweeks = tw[["season", "week", "team", "team_win"]]
inj = inj.merge(tweeks, on=["season", "week", "team"], how="inner")
inj["played"] = [(s, w, p) in played for s, w, p in zip(inj.season, inj.week, inj.player_id)]
rep = set(map(tuple, inj[(inj.gs != "none") | (inj.pr != "none")][["season", "week", "player_id"]].values))
def hist(r):
    n = sum((r.season, r.week - k, r.player_id) in rep for k in range(1, 9))
    m = sum(((r.season, r.week - k, r.player_id) in rep) and ((r.season, r.week - k, r.player_id) not in played) for k in range(1, 9))
    return n, m
h = [hist(r) for r in inj.itertuples()]
inj["on8"] = [a for a, b in h]; inj["miss8"] = [b for a, b in h]
bpmap = {(r.season, r.week, r.player_id): r.inj for r in inj.itertuples()}
def streak(r):
    k = 1
    while bpmap.get((r.season, r.week - k, r.player_id)) == r.inj and k < 8:
        k += 1
    return k - 1
inj["streak"] = [streak(r) for r in inj.itertuples()]
# prior-season games missed (injury history)
g = pw.groupby(["player_id", "season"]).size().rename("g").reset_index()
g["season"] += 1
inj = inj.merge(g, on=["player_id", "season"], how="left").rename(columns={"g": "gp_last"})
inj["gp_last"] = inj.gp_last.fillna(17)
# as-of: his rolling touches from the last game he played BEFORE this week (so absent players are kept)
a = pw[["player_id", "season", "week", "t6"]].sort_values("week").rename(columns={"week": "wk_a"})
inj["week"] = inj.week.astype("int64"); inj["season"] = inj.season.astype("int64"); inj = inj.sort_values("week"); a["wk_a"] = a.wk_a.astype("int64"); a["season"] = a.season.astype("int64")
rel = pd.merge_asof(inj, a, left_on="week", right_on="wk_a", by=["player_id", "season"], allow_exact_matches=False)
# players with no earlier game this year: use last season's tail is not available -> dropped
rel = rel[rel.t6.fillna(0) >= rel.position.map({"QB": 12, "RB": 7, "WR": 7, "TE": 5}) ]  # relevant players only
rel = rel[(rel.gs != "none") | (rel.pr != "none")].copy()
X = pd.concat([pd.get_dummies(rel.gs, prefix="gs"), pd.get_dummies(rel.pr, prefix="pr"), pd.get_dummies(rel.inj, prefix="inj"),
               pd.get_dummies(rel.position, prefix="pos"),
               rel[["on8", "miss8", "streak", "team_win", "week"]].astype(float), (17 - rel.gp_last).clip(lower=0).rename("missed_last_yr").astype(float)], axis=1).astype(float)
cur = rel.season == 2026
hist_m = ~cur & rel.played.notna()
print(f"{hist_m.sum()} historical player-weeks on the report ({rel[hist_m].played.mean():.0%} played)")
# A: plain status table
tab = rel[hist_m].groupby(["gs", "pr"]).played.agg(["mean", "size"]).round(2).reset_index()
print("\nA. Played rate by game status x last practice (2018-2025)\n", tab.to_string(index=False))
# B: LOSO
P = pd.Series(np.nan, index=rel.index); Pb = P.copy()
for s in sorted(rel[hist_m].season.unique()):
    tr = hist_m & (rel.season != s); te = hist_m & (rel.season == s)
    m = LogisticRegression(C=0.5, max_iter=2000).fit(X[tr], rel.played[tr]); P[te] = m.predict_proba(X[te])[:, 1]
    base = rel[tr].groupby(["gs", "pr"]).played.mean(); Pb[te] = [base.get((a, b), rel.played[tr].mean()) for a, b in zip(rel.gs[te], rel.pr[te])]
y = rel.played[hist_m].astype(float)
brier = lambda p: float(((p[hist_m] - y) ** 2).mean())
print(f"\nB. Brier (lower is better): flat {((y - y.mean()) ** 2).mean():.4f}  status x practice table {brier(Pb):.4f}  logistic with history/team {brier(P):.4f}")
# calibration
cal = pd.DataFrame({"p": P[hist_m], "y": y}); cal["bin"] = pd.cut(cal.p, [0, .2, .4, .6, .8, .9, 1.0])
print("\nCalibration (LOSO)\n", cal.groupby("bin", observed=True).agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size")).round(2).to_string())
m = LogisticRegression(C=0.5, max_iter=2000).fit(X[hist_m], rel.played[hist_m])
co = pd.Series(m.coef_[0], index=X.columns).sort_values()
print("\nC. What moves it (logit coefficients, + = more likely to play)\n", co.round(2).to_string())
# slices: Questionable+Limited
q = rel[hist_m & (rel.gs == "Questionable")]
print("\nD. Questionable only: played rate by ...")
for col, bins in [("team_win", [0, .35, .5, .65, 1]), ("on8", [-1, 1, 3, 8]), ("miss8", [-1, 0, 1, 8]), ("streak", [-1, 0, 2, 8])]:
    print(q.groupby(pd.cut(q[col], bins), observed=True).played.agg(["mean", "size"]).round(2).T.to_string(), f"  <- {col}")
print(q.groupby("inj").played.agg(["mean", "size"]).round(2).sort_values("mean").T.to_string())
# E: 2026 current week
now = rel[cur & (rel.week == WK)].copy()
now["p_play"] = m.predict_proba(X[cur & (rel.week == WK)])[:, 1]
now.to_csv("/tmp/claude-0/-home-user/536a3433-245d-53ae-b698-b7139fba29ad/scratchpad/now_report.csv", index=False)
ros = pd.read_csv(DATA / "input" / f"burbs_rosters_2026wk{WK}.csv")
names = pd.DataFrame(pl.read_parquet(DATA / "processed" / "player_weeks.parquet").filter(pl.col("season") == 2026).select("player_id", "player_display_name").unique().to_pandas())
print("\nE. Week", WK, "report, Burbs roster")
mine = now.merge(names, on="player_id").merge(ros, left_on="player_display_name", right_on="Player")
print(mine[["Team", "Slot", "Player", "gs", "pr", "inj", "on8", "miss8", "p_play"]].round(2).to_string(index=False))
# F: Questionable + limited practice, by how often he was listed and missed before
f = rel[hist_m & (rel.gs == "Questionable") & (rel.pr == "Lim")]
print("\nF. Questionable + Limited: played rate by (listed in prior 8 wks, missed in prior 8)")
print(f.groupby([pd.cut(f.on8, [-1, 1, 3, 8]), pd.cut(f.miss8, [-1, 0, 8])], observed=True).played.agg(["mean", "size"]).round(2).to_string())
print(f[f.inj == "ankle"].played.agg(["mean", "size"]).round(2).to_dict(), "ankle Q/Lim")
