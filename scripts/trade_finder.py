"""Find 1-for-2 / 2-for-2 trades that raise your weekly starting lineup and also help the other team.

PYTHONPATH=src python3 scripts/trade_finder.py burbs 9 <rosters csv> "Gin and Jews" "Jaylen Warren,Bhayshul Tuten"
Value = sum over weeks 10-17 of the team's best legal lineup (rest-of-season PPG, schedule taken out, plus that
week's opponent; byes = 0; injured players until their return week = 0); playoff weeks 15-17 count 1.5x
(PLAYOFF_FOCUS=1: playoff weeks 2x, regular season 0.5x; the other team still values it the normal way).
Players "ALL" = any 1 or 2 of your QB/RB/WR/TE for any 1 or 2 of theirs.
A 2-for-1 drops your lowest-value player. Only trades where the partner's lineup also improves are kept.
"""
import datetime as dt
import itertools
import re
import sys

import numpy as np
import pandas as pd
import polars as pl

from ffmodel import data
from ffmodel.config import DATA, load_leagues
from ffmodel.opponent import defense_effects
from ffmodel.ros import dst_weeks

lg, week, rpath, ME, keys = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5].split(",")
season, out, thru = 2026, DATA / "output", week - 1
WEEKS = list(range(week + 1, 18))
import os
PO_FOCUS = os.environ.get("PLAYOFF_FOCUS") == "1"  # playoff weeks 2x, regular season 0.5x (still need to get in)
WT = np.array([(2.0 if w >= 15 else 0.5) if PO_FOCUS else (1.5 if w >= 15 else 1.0) for w in WEEKS])
WT_THEM = np.array([1.5 if w >= 15 else 1.0 for w in WEEKS])  # how the other team values it
MATCH = {"QB": 1.0, "RB": 0.45, "WR": 0.35, "TE": 0.55, "K": 0.0, "DEF": 0.5}
cfg = load_leagues()[lg]
slots, flex = cfg["roster"], cfg["flex_eligible"]
roster = pd.read_csv(rpath).fillna("")
rs = pd.read_excel(out / f"ros_{season}_wk{week}_{lg}.xlsx", sheet_name="Overall")
w_ = pl.read_parquet(DATA / "processed" / "player_weeks.parquet").filter(pl.col("season") == season, pl.col("week") <= thru)
eff = defense_effects(w_, f"fpts_{lg}").select("position", "opponent_team", pl.col(f"def_fpts_{lg}").alias("eff")).to_pandas()
E = {(r.position, r.opponent_team): r.eff for r in eff.itertuples()}
sch = data.schedules([season]).filter(pl.col("game_type") == "REG").to_pandas()
opp = {}
for r in sch.itertuples():
    opp[(r.week, r.home_team)] = (r.away_team, r.gameday)
    opp[(r.week, r.away_team)] = (r.home_team, r.gameday)


def first_week(r):
    s, news = str(r["Status"]), str(r["Injury News"])
    if s == "Out for season":
        return 99
    if s in ("Practice squad", "No games yet"):
        return 99
    if s in ("Out", "Injured Reserve", "Doubtful", "IR / PUP / suspended"):
        m = re.search(r"est\. return (\w{3}) (\d+)", news)
        if not m:
            return week + 4
        mon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"].index(m.group(1)) + 1
        d = dt.date(season + (1 if mon <= 3 else 0), mon, int(m.group(2)))
        for w in WEEKS:
            o = opp.get((w, r["Team"]))
            if o and dt.date.fromisoformat(o[1]) >= d:
                return w
        return 99
    return week + 1


rs = rs.set_index("Player")
VAL = {}
for name, r in rs.iterrows():
    base = r["ROS PPG"] - (r["Schedule Adj (PPG)"] if pd.notna(r["Schedule Adj (PPG)"]) else 0)
    fw = first_week(r)
    v = []
    for w in WEEKS:
        o = opp.get((w, r["Team"]))
        v.append(0.0 if (o is None or w < fw) else max(0.0, base + MATCH.get(r["Pos"], 0) * E.get((r["Pos"], o[0]), 0.0)))
    VAL[name] = (r["Pos"], np.array(v))


def team_value(names):
    tot = np.zeros(len(WEEKS))
    ps = [(n, *VAL[n]) for n in names if n in VAL]
    for i in range(len(WEEKS)):
        used, s = set(), 0.0
        for q in ["QB", "RB", "WR", "TE", "K", "DEF"]:
            c = sorted([(v[i], n) for n, p, v in ps if p == q and n not in used], reverse=True)[:slots.get(q, 0)]
            used |= {n for _, n in c}; s += sum(x for x, _ in c)
        c = sorted([(v[i], n) for n, p, v in ps if p in flex and n not in used], reverse=True)[:slots.get("FLEX", 0)]
        tot[i] = s + sum(x for x, _ in c)
    return tot


def score(t, wt=None):
    return float((t * (WT if wt is None else wt)).sum())


mine = list(roster[roster.Team == ME].Player)
base_me = score(team_value(mine))
skill = [p for p in mine if VAL.get(p, ("K",))[0] in ("QB", "RB", "WR", "TE")]
if keys == ["ALL"]:  # any 1 or 2 of my skill players
    gives = [[a] for a in skill] + [list(c) for c in itertools.combinations(skill, 2)]
else:
    gives = [[k] for k in keys] + [[k, o] for k in keys for o in skill if o not in keys]
GET_SIZES = (1, 2) if keys == ["ALL"] else (2,)
rows = []
for t in roster.Team.unique():
    if t == ME:
        continue
    theirs = [p for p in roster[roster.Team == t].Player if p in VAL and VAL[p][0] in ("QB", "RB", "WR", "TE")]
    base_them = score(team_value(list(roster[roster.Team == t].Player)), WT_THEM)
    for give in gives:
        for get in (g for k_ in GET_SIZES for g in itertools.combinations(theirs, k_)):
            new_me = [p for p in mine if p not in give] + list(get)
            if len(get) > len(give):  # drop my lowest-value player to make room
                worst = min((p for p in new_me if p not in get), key=lambda p: score(team_value([p])) if p in VAL else -1)
                new_me = [p for p in new_me if p != worst]
            else:
                worst = ""
            new_them = [p for p in roster[roster.Team == t].Player if p not in get] + give
            d_me = score(team_value(new_me)) - base_me
            if len(give) > len(get):  # they drop their lowest-value player
                tw = min((p for p in new_them if p not in give), key=lambda p: score(team_value([p]), WT_THEM) if p in VAL else -1)
                new_them = [p for p in new_them if p != tw]
            d_them = score(team_value(new_them), WT_THEM) - base_them
            if d_me > 0 and d_them > 0:
                rows.append({"Partner": t, "You give": " + ".join(give), "You get": " + ".join(get), "Drop": worst,
                             f"Your gain (weighted, wk {WEEKS[0]}-17)": round(d_me, 1),
                             f"Your reg season gain/wk (wk {WEEKS[0]}-14)": round((team_value(new_me) - team_value(mine))[:-3].mean(), 2),
                             "Their gain (pts)": round(d_them, 1),
                             "Your playoff wks gain/wk": round((team_value(new_me) - team_value(mine))[-3:].mean(), 2)})
gcol = f"Your gain (weighted, wk {WEEKS[0]}-17)"
res = pd.DataFrame(rows).sort_values([gcol], ascending=False)
# balanced: both gain, rank by the smaller of the two gains (most likely to be accepted)
res["Min gain"] = res[[gcol, "Their gain (pts)"]].min(axis=1)
fp = out / (f"trades_{lg}_{season}_wk{week}" + ("_playoffs" if PO_FOCUS else "") + ".xlsx")
with pd.ExcelWriter(fp) as xw:
    res.head(200).to_excel(xw, sheet_name="Best for you", index=False)
    res.sort_values("Min gain", ascending=False).head(200).to_excel(xw, sheet_name="Most balanced", index=False)
    pd.DataFrame({"Notes": __doc__.strip().splitlines()}).to_excel(xw, sheet_name="Notes", index=False)
print(fp, len(res))
pd.set_option("display.width", 250)
print(res.head(15).to_string(index=False))
print(res.sort_values("Min gain", ascending=False).head(15).to_string(index=False))
