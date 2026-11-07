"""Rest-of-season team strength and playoff odds for a league from pasted rosters + standings.

PYTHONPATH=src python3 scripts/playoff_odds.py burbs 9 data/input/burbs_rosters_2026wk9.csv data/input/burbs_standings_2026wk8.csv [playoff_teams=6]
Each remaining regular-season week: every team's best lineup from ROS PPG x chance he plays, byes removed
(this week: the weekly projection, with final games at actual points). Unknown schedule -> random pairings
each week (fixed pairs can be given in KNOWN). Team-week score ~ Normal(proj, 22.5); ties on record broken by points for.
Writes data/output/playoff_odds_<league>_2026_wk<week>.xlsx.
"""
import sys

import numpy as np
import pandas as pd
import polars as pl

from ffmodel import data, ros
from ffmodel.config import DATA, load_leagues

lg, week, rpath, spath = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
n_po = int(sys.argv[5]) if len(sys.argv) > 5 else 6
season, out = 2026, DATA / "output"
# Scott's schedule: round robin weeks 1-11 (Broken Ankles and BKs Ballers still to play, order assumed),
# then weeks 12-14 repeat weeks 1-3 (usual Yahoo 12-team pattern; assumed)
ME = "Gin and Jews"
KNOWN = {9: [(ME, "Who Are You People")], 10: [(ME, "The Broken Ankles")], 11: [(ME, "BKs Ballers")],
         12: [(ME, "Gaza Vybz")], 13: [(ME, "Daddy Fat Sax")], 14: [(ME, "🙏")]}
SD, N = 22.5, 20000
cfg = load_leagues()[lg]
slots, flex = cfg["roster"], cfg["flex_eligible"]
last_reg = cfg["regular_season_weeks"][1]

roster = pd.read_csv(rpath).fillna("")
st = pd.read_csv(spath)
rs = pd.read_excel(out / f"ros_{season}_wk{week}_{lg}.xlsx", sheet_name="Overall")
wk = pd.read_excel(out / f"weekly_{season}_wk{week}_{lg}.xlsx", sheet_name="Overall")
done = ros._weekly(season).filter(pl.col("week") == week).select("player_id", pl.col(f"fpts_{lg}").alias("actual")).to_pandas()
wk = wk.merge(done, left_on="nflverse ID", right_on="player_id", how="left")
wk["pts"] = wk["actual"].fillna(wk["Wk Proj"])
rs["avail"] = (rs["Exp Games Left"] / (rs["Games Left (reg season)"] + rs["Playoff Games (wk 15-17)"]).replace(0, np.nan)).fillna(0)
# healthy players count at full PPG (a missed game is covered from the bench); hurt/IR ones at PPG x availability
rs["exp"] = np.where(rs["avail"] >= 0.6, rs["ROS PPG"], rs["ROS PPG"] * rs["avail"])
sch = data.schedules([season]).filter(pl.col("game_type") == "REG").to_pandas()
plays = {w: set(sch[sch.week == w].home_team) | set(sch[sch.week == w].away_team) for w in range(week, 18)}


def best(pool):
    pool = pool.sort_values("v", ascending=False)
    used, tot = set(), 0.0
    for s in ["QB", "RB", "WR", "TE", "K", "DEF"]:
        c = pool[(pool.Pos == s) & ~pool.Player.isin(used)].head(slots.get(s, 0))
        used |= set(c.Player); tot += c.v.sum()
    c = pool[pool.Pos.isin(flex) & ~pool.Player.isin(used)].head(slots.get("FLEX", 0))
    return tot + c.v.sum()


teams = list(st.Team)
proj = {t: {} for t in teams}
for t in teams:
    g = roster[(roster.Team == t) & (roster.Slot != "IR")]
    # this week: weekly projection, final games locked as started/benched
    p = g.merge(wk[["Player", "Pos", "pts", "actual"]], on="Player")
    start = (p.Slot != "BN")
    lock = p[start & p.actual.notna()]
    pool = p[p.actual.isna()].rename(columns={"pts": "v"})
    sl = dict(slots)
    for s in lock.Slot:
        sl[s] = sl.get(s, 0) - 1
    saved = dict(slots); slots.clear(); slots.update(sl)
    proj[t][week] = lock.pts.sum() + best(pool)
    slots.clear(); slots.update(saved)
    q = g.merge(rs[["Player", "Pos", "Team", "exp"]], on="Player", suffixes=("", "_nfl"))
    for w in range(week + 1, 18):
        proj[t][w] = best(q[q.Team_nfl.isin(plays[w])].rename(columns={"exp": "v"}))

rng = np.random.default_rng(7)
W = np.tile(st.W.to_numpy(float) + 0.5 * st["T"].to_numpy(float), (N, 1))
PF = np.tile(st.PF.to_numpy(float), (N, 1))
idx = {t: i for i, t in enumerate(teams)}
first_win = np.zeros(N, dtype=bool)
my_w = {w: np.zeros(N, dtype=bool) for w in KNOWN}
for w in range(week, last_reg + 1):
    mu = np.array([proj[t][w] for t in teams])
    sd = np.full(len(teams), SD)
    if w == week:  # part of this week is already final
        sd = sd * 0.85
    sc = rng.normal(mu, sd, size=(N, len(teams)))
    PF += sc
    fixed = KNOWN.get(w, [])
    fixed_ids = [(idx[a], idx[b]) for a, b in fixed]
    rest = [i for i in range(len(teams)) if all(i not in pr for pr in fixed_ids)]
    for k in range(N):
        perm = rng.permutation(rest)
        pairs = fixed_ids + [(perm[j], perm[j + 1]) for j in range(0, len(perm), 2)]
        for a, b in pairs:
            if w == week and fixed_ids and (a, b) == fixed_ids[0]:
                first_win[k] = sc[k, a] > sc[k, b]
            if fixed_ids and (a, b) == fixed_ids[0]:
                my_w[w][k] = sc[k, a] > sc[k, b]
            if sc[k, a] > sc[k, b]:
                W[k, a] += 1
            else:
                W[k, b] += 1
key = W * 10000 + PF
rank = (-key).argsort(axis=1).argsort(axis=1) + 1
res = pd.DataFrame({
    "Team": teams, "Record": [f"{a}-{b}" for a, b in zip(st.W, st.L)], "PF": st.PF,
    f"Wk {week} Proj": [round(proj[t][week], 1) for t in teams],
    f"ROS Avg (wk {week + 1}-{last_reg})": [round(np.mean([proj[t][w] for w in range(week + 1, last_reg + 1)]), 1) for t in teams],
    "Playoff Avg (wk 15-17)": [round(np.mean([proj[t][w] for w in range(15, 18)]), 1) for t in teams],
    "Proj Wins": (W.mean(axis=0)).round(1),
    "Make Playoffs %": ((rank <= n_po).mean(axis=0) * 100).round(0),
    "Top-2 Bye %": ((rank <= 2).mean(axis=0) * 100).round(0),
    "Avg Finish": rank.mean(axis=0).round(1),
}).sort_values("Make Playoffs %", ascending=False)
byw = pd.DataFrame({t: {f"Wk {w}": round(proj[t][w], 1) for w in range(week, 18)} for t in teams}).T.reset_index().rename(columns={"index": "Team"})
fp = out / f"playoff_odds_{lg}_{season}_wk{week}.xlsx"
with pd.ExcelWriter(fp) as xw:
    res.to_excel(xw, sheet_name="Playoff Odds", index=False)
    byw.to_excel(xw, sheet_name="Weekly Team Proj", index=False)
    pd.DataFrame({"Notes": [
        f"Simulated {N:,} times. Playoffs: top {n_po} by record, ties broken by points for. Top 2 = bye (assumed).",
        "Team projection each week = best legal lineup from rest-of-season PPG, byes removed; injured/IR players discounted by the chance they play.",
        f"Week {week}: weekly projections, with games already played at actual points.",
        "Gin and Jews schedule fixed: wk 10 Broken Ankles, wk 11 BKs Ballers (order assumed), wks 12-14 repeat wks 1-3 (Gaza Vybz, Daddy Fat Sax, 🙏; assumed). Other teams: random opponents.",
        "Each team's weekly score varies about +/-22.5 points (from 2021-2025 player-level misses). Rosters as pasted: no future trades/pickups.",
    ]}).to_excel(xw, sheet_name="Notes", index=False)
if KNOWN.get(week):
    a = idx[KNOWN[week][0][0]]
    print(f"{teams[a]}: wins wk {week} {first_win.mean():.0%}; playoffs if win {(rank[first_win, a] <= n_po).mean():.0%}, if loss {(rank[~first_win, a] <= n_po).mean():.0%}")
for w in sorted(KNOWN):
    print(f"wk {w} vs {KNOWN[w][0][1]}: win {my_w[w].mean():.0%}")
print(fp)
print(res.to_string(index=False))
