"""Best lineup + projected score for every team in a league, and the best free agents.

PYTHONPATH=src python3 scripts/league_week.py burbs 9 data/input/burbs_rosters_2026wk9.csv
Roster CSV: Team, Slot, Player, Yahoo Status (paste of the league's rosters). Games already played use actual points.
Writes data/output/league_<league>_2026_wk<week>.xlsx: Teams, Lineups, Free Agents (week), Free Agents (ROS).
"""
import sys

import pandas as pd
import polars as pl

from ffmodel import ros
from ffmodel.config import DATA, load_leagues
from ffmodel.guide import add_guide

lg, week, path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
season = 2026
out = DATA / "output"
wk = pd.read_excel(out / f"weekly_{season}_wk{week}_{lg}.xlsx", sheet_name="Overall")
rs = pd.read_excel(out / f"ros_{season}_wk{week}_{lg}.xlsx", sheet_name="Overall")
roster = pd.read_csv(path).fillna("")
# actual points for games already final
done = ros._weekly(season).filter(pl.col("week") == week).select("player_id", pl.col(f"fpts_{lg}").alias("actual")).to_pandas()
wk = wk.merge(done, left_on="nflverse ID", right_on="player_id", how="left")
wk["Final"] = wk["actual"].notna()
wk["Pts"] = wk["actual"].fillna(wk["Wk Proj"])
cfg = load_leagues()[lg]
slots = cfg["roster"]
flex = cfg["flex_eligible"]

p = roster.merge(wk[["Player", "Pos", "Team", "Opp", "Pts", "Final", "Status"]], on="Player", how="left")
p = p.merge(rs[["Player", "Pos Rk", "ROS VORP"]], on="Player", how="left")
lineups, teams = [], []
for t, g in p.groupby("Team_x", sort=False):
    # games already final are locked: a started player stays in his slot, a benched one can't come in
    starting = (g.Slot != "BN") & (g.Slot != "IR")
    locked = g[starting & (g.Final == True)]
    pool = g[(g.Slot != "IR") & g.Pts.notna() & (g.Final != True)].sort_values("Pts", ascending=False)
    used, rows = set(locked.Player), [(r.Slot, r) for r in locked.itertuples(index=False)]
    left = dict(slots)
    for sl in locked.Slot:
        left[sl] -= 1
    for s in ["QB", "RB", "WR", "TE", "K", "DEF"]:
        for _ in range(left.get(s, 0)):
            c = pool[(pool.Pos == s) & ~pool.Player.isin(used)].head(1)
            if len(c):
                used.add(c.Player.iloc[0]); rows.append((s, c.iloc[0]))
    for _ in range(left.get("FLEX", 0)):
        c = pool[pool.Pos.isin(flex) & ~pool.Player.isin(used)].head(1)
        if len(c):
            used.add(c.Player.iloc[0]); rows.append(("FLEX", c.iloc[0]))
    order = {"QB": 0, "RB": 1, "WR": 2, "TE": 3, "FLEX": 4, "K": 5, "DEF": 6}
    rows.sort(key=lambda x: order[x[0]])
    tot = sum(r.Pts for _, r in rows)
    set_now = set(g[(g.Slot != "BN") & (g.Slot != "IR")].Player)
    cur = g[g.Player.isin(set_now)].Pts.fillna(0).sum()
    changes = sorted(used - set_now)
    for s, r in rows:
        lineups.append({"Team": t, "Slot": s, "Player": r.Player, "NFL": r.Team_y, "Opp": r.Opp,
                        "Pts": round(r.Pts, 1), "Final": "Final" if r.Final else "", "Status": r.Status if isinstance(r.Status, str) else "",
                        "Change": "" if r.Player in set_now else "start (bench now)"})
    teams.append({"Team": t, "Best Lineup Pts": round(tot, 1), "As Set Pts": round(cur, 1),
                  "Gain From Changes": round(tot - cur, 1), "Suggested Starts": ", ".join(changes)})
teams = pd.DataFrame(teams).sort_values("Best Lineup Pts", ascending=False)
teams.insert(0, "Rk", range(1, len(teams) + 1))
owned = set(roster.Player)
fa_w = wk[~wk.Player.isin(owned) & ~wk.Final].sort_values("Pts", ascending=False)
fa_w = pd.concat([fa_w[fa_w.Pos == q].head(12) for q in ["QB", "RB", "WR", "TE", "K", "DEF"]])
fa_w = fa_w[["Pos", "Pos Rk", "Player", "Team", "Opp", "Pts", "Final", "Status", "Play Chance", "Role", "Last 3 PPG"]].rename(columns={"Pts": "Wk Pts", "Pos Rk": "Wk Pos Rk"})
fa_r = rs[~rs.Player.isin(owned)]
fa_r = pd.concat([fa_r[fa_r.Pos == q].head(12) for q in ["QB", "RB", "WR", "TE", "K", "DEF"]])
fa_r = fa_r[["Pos", "Pos Rk", "Player", "Team", "Status", "ROS PPG", "Role", "Work Share (before → last 3)", "PPG So Far", "Exp Games Left", "ROS VORP"]].rename(columns={"Pos Rk": "ROS Pos Rk"})
fp = out / f"league_{lg}_{season}_wk{week}.xlsx"
with pd.ExcelWriter(fp) as xw:
    teams.to_excel(xw, sheet_name="Teams", index=False)
    pd.DataFrame(lineups).to_excel(xw, sheet_name="Lineups", index=False)
    fa_w.to_excel(xw, sheet_name="Free Agents (week)", index=False)
    fa_r.to_excel(xw, sheet_name="Free Agents (ROS)", index=False)
print(fp)
print(teams.to_string(index=False))
print(fa_w.groupby("Pos").head(4)[["Pos", "Player", "Opp", "Wk Pts", "Wk Pos Rk"]].to_string(index=False))
print(fa_r.groupby("Pos").head(4)[["Pos", "Player", "ROS PPG", "ROS Pos Rk", "Role"]].to_string(index=False))
