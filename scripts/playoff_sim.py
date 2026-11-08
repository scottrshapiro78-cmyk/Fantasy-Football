"""Season simulation with player injuries, week-by-week matchups and bench depth.

PYTHONPATH=src python3 scripts/playoff_sim.py burbs 9 <rosters csv> <standings csv> [playoff_teams=6]

For each simulated season (weeks after this one):
  * each player's points that week = his rest-of-season PPG with the season-long schedule taken out, plus
    that week's opponent (same matchup weights as the weekly rankings); 0 on bye
  * healthy players get hurt at the weekly rate for their position in weeks 9-17 of 2018-2025
    (QB 6.9%, RB 8.7%, WR 8.0%, TE 8.7%, K 3.6%), and miss a number of games drawn from what those
    injuries actually cost (half are one game; ~10-15% end the season); players already hurt stay out
    until their reported return date
  * every team starts its best available lineup; if a slot can't be filled the team adds a free agent
    (the 3rd-best unowned player at the position, from the pasted rosters)
  * team score ~ Normal(lineup points, 21); this week uses the weekly projections (final games at actual)
Ties on record broken by points for. Writes data/output/playoff_sim_<league>_2026_wk<week>.xlsx.
"""
import datetime as dt
import re
import sys

import numpy as np
import pandas as pd
import polars as pl

from ffmodel import data, ros
from ffmodel.config import DATA, load_leagues
from ffmodel.opponent import defense_effects
from ffmodel.ros import dst_weeks

lg, week, rpath, spath = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
n_po = int(sys.argv[5]) if len(sys.argv) > 5 else 6
season, out, thru = 2026, DATA / "output", week - 1
N, SD = 10000, 21.0
HAZ = {"QB": 0.069, "RB": 0.087, "WR": 0.080, "TE": 0.087, "K": 0.036, "DEF": 0.0}
# games missed per injury (6 = rest of season), weeks 9-17 2018-2025
DUR = {"QB": [.40, .18, .15, .07, .04, .16], "RB": [.50, .21, .11, .08, .03, .07], "WR": [.51, .16, .10, .08, .03, .12],
       "TE": [.50, .15, .15, .02, .06, .12], "K": [.30, .18, .12, .17, .03, .20], "DEF": [1, 0, 0, 0, 0, 0]}
MATCH = {"QB": 1.0, "RB": 0.45, "WR": 0.35, "TE": 0.55, "K": 0.0, "DEF": 0.5}
ME = "Gin and Jews"
MC = "I can't find McConkeys"
# known pairings (pasted schedules). Weeks 12-14 repeat weeks 1-3 (confirmed by both pasted schedules).
SCHED = {9: [(ME, "Who Are You People"), (MC, "Bring the Ruckus")],
         10: [(ME, "BKs Ballers"), (MC, "TY Dolla $ign")],
         11: [(ME, "The Broken Ankles"), (MC, "Degenerates")],
         12: [(ME, "Gaza Vybz"), (MC, "Daddy Fat Sax")],
         13: [(ME, "Daddy Fat Sax"), (MC, "More Than A Thielen")],
         14: [(ME, "🙏"), (MC, "Who Are You People")]}
cfg = load_leagues()[lg]
slots, flex = cfg["roster"], cfg["flex_eligible"]
last_reg = cfg["regular_season_weeks"][1]
WEEKS = list(range(week + 1, 18))

roster = pd.read_csv(rpath).fillna("")
st = pd.read_csv(spath)
teams = list(st.Team)
rs = pd.read_excel(out / f"ros_{season}_wk{week}_{lg}.xlsx", sheet_name="Overall")
wk = pd.read_excel(out / f"weekly_{season}_wk{week}_{lg}.xlsx", sheet_name="Overall")
done = ros._weekly(season).filter(pl.col("week") == week).select("player_id", pl.col(f"fpts_{lg}").alias("actual")).to_pandas()
wk = wk.merge(done, left_on="nflverse ID", right_on="player_id", how="left")
wk["pts"] = wk["actual"].fillna(wk["Wk Proj"])

# opponent effects (this season through last week)
w_ = pl.read_parquet(DATA / "processed" / "player_weeks.parquet").filter(pl.col("season") == season, pl.col("week") <= thru)
eff = defense_effects(w_, f"fpts_{lg}").select("position", "opponent_team", pl.col(f"def_fpts_{lg}").alias("eff")).to_pandas()
dw = dst_weeks().filter(pl.col("season") == season, pl.col("week") <= thru)
mu = dw[f"fpts_{lg}"].mean()
off = dw.group_by("opponent_team").agg(((pl.col(f"fpts_{lg}") - mu).sum() / (pl.len() + 4)).alias("eff")).to_pandas()
off["position"] = "DEF"
E = {(r.position, r.opponent_team): r.eff for r in pd.concat([eff, off]).itertuples()}
sch = data.schedules([season]).filter(pl.col("game_type") == "REG").to_pandas()
opp = {}
for r in sch.itertuples():
    opp[(r.week, r.home_team)] = (r.away_team, r.gameday)
    opp[(r.week, r.away_team)] = (r.home_team, r.gameday)


def ret_week(team, news):
    m = re.search(r"est\. return (\w{3}) (\d+)", news or "")
    if not m:
        return None
    mon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"].index(m.group(1)) + 1
    d = dt.date(season + (1 if mon <= 3 else 0), mon, int(m.group(2)))
    for w in WEEKS:
        o = opp.get((w, team))
        if o and dt.date.fromisoformat(o[1]) >= d:
            return w
    return 99


def player_table(names):
    p = rs[rs.Player.isin(names)].copy()
    p["base"] = p["ROS PPG"] - p["Schedule Adj (PPG)"].fillna(0)
    V = np.zeros((len(WEEKS), len(p)))
    for j, r in enumerate(p.itertuples()):
        for i, w in enumerate(WEEKS):
            o = opp.get((w, r.Team))
            V[i, j] = 0.0 if o is None else max(0.0, r.base + MATCH[r.Pos] * E.get((r.Pos, o[0]), 0.0))
    first = []  # first week he can play
    for r in p.itertuples():
        s = str(r.Status)
        news = r[p.columns.get_loc("Injury News") + 1]
        if s == "Out for season":
            first.append(99)
        elif s in ("Practice squad", "No games yet"):
            first.append(99 if r.Pos != "DEF" else week + 1)
        elif s in ("Out", "Injured Reserve", "Doubtful", "IR / PUP / suspended"):
            rw = ret_week(r.Team, news if isinstance(news, str) else "")
            first.append(rw if rw else week + 4)
        else:
            first.append(week + 1)
    return p.reset_index(drop=True), V, np.array(first)


owned = set(roster.Player)
fa = rs[~rs.Player.isin(owned)]
waiver = {q: fa[fa.Pos == q]["ROS PPG"].nlargest(3).iloc[-1] for q in ["QB", "RB", "WR", "TE", "K", "DEF"]}
rng = np.random.default_rng(11)


def sim_team(names, drop=()):
    p, V, first = player_table([n for n in names if n not in drop])
    nP, nW = len(p), len(WEEKS)
    pos = p.Pos.to_numpy()
    avail = np.ones((N, nW, nP), dtype=bool)
    for j in range(nP):
        start = first[j] - (week + 1)
        if start > 0:
            avail[:, :min(start, nW), j] = False
    # new injuries
    out_until = np.zeros((N, nP), dtype=int)
    for i in range(nW):
        healthy = avail[:, i, :] & (out_until <= i)
        hz = np.array([HAZ[q] for q in pos])
        hit = healthy & (rng.random((N, nP)) < hz)
        if hit.any():
            dur = np.zeros((N, nP), dtype=int)
            for j in range(nP):
                k = hit[:, j].sum()
                if k:
                    d = rng.choice(np.arange(1, 7), size=k, p=np.array(DUR[pos[j]]) / sum(DUR[pos[j]]))
                    d = np.where(d == 6, 99, d)
                    dur[hit[:, j], j] = d
            out_until = np.where(hit, i + dur, out_until)
        avail[:, i, :] &= ~(out_until > i)
    tot = np.zeros((N, nW))
    for i in range(nW):
        vals = np.where(avail[:, i, :], V[i][None, :], -np.inf)
        lineup = np.zeros(N)
        extras = []
        for q, k in [("QB", slots["QB"]), ("RB", slots["RB"]), ("WR", slots["WR"]), ("TE", slots["TE"]), ("K", slots["K"]), ("DEF", slots["DEF"])]:
            cols = np.where(pos == q)[0]
            s = -np.sort(-vals[:, cols], axis=1) if len(cols) else np.full((N, 0), -np.inf)
            pad = np.full((N, k + 1), waiver[q])
            s = np.concatenate([np.where(np.isfinite(s), s, -np.inf), pad], axis=1)
            s = -np.sort(-s, axis=1)
            lineup += s[:, :k].sum(axis=1)
            if q in flex:
                extras.append(s[:, k])
        lineup += np.max(np.stack(extras, axis=1), axis=1)
        tot[:, i] = lineup
    return tot


wk9 = {}
for t in teams:
    g = roster[(roster.Team == t) & (roster.Slot != "IR")].merge(wk[["Player", "Pos", "pts", "actual"]], on="Player")
    lock = g[(g.Slot != "BN") & g.actual.notna()]
    pool = g[g.actual.isna()].sort_values("pts", ascending=False)
    left = dict(slots)
    for s in lock.Slot:
        left[s] -= 1
    used, tot = set(), lock.pts.sum()
    for q in ["QB", "RB", "WR", "TE", "K", "DEF"]:
        c = pool[(pool.Pos == q) & ~pool.Player.isin(used)].head(left.get(q, 0))
        used |= set(c.Player); tot += c.pts.sum()
    c = pool[pool.Pos.isin(flex) & ~pool.Player.isin(used)].head(left.get("FLEX", 0))
    wk9[t] = tot + c.pts.sum()

S = {t: sim_team(list(roster[roster.Team == t].Player)) for t in teams}
idx = {t: i for i, t in enumerate(teams)}
Wn = np.tile(st.W.to_numpy(float), (N, 1))
PF = np.tile(st.PF.to_numpy(float), (N, 1))
mywin = {}
for w in range(week, last_reg + 1):
    if w == week:
        mean = np.tile(np.array([wk9[t] for t in teams]), (N, 1)); sd = SD * 0.85
    else:
        mean = np.stack([S[t][:, WEEKS.index(w)] for t in teams], axis=1); sd = SD
    sc = rng.normal(mean, sd)
    PF += sc
    fixed = [(idx[a], idx[b]) for a, b in SCHED.get(w, [])]
    rest = [i for i in range(len(teams)) if all(i not in f for f in fixed)]
    for k in range(N):
        perm = rng.permutation(rest)
        for a, b in fixed + [(perm[j], perm[j + 1]) for j in range(0, len(perm), 2)]:
            Wn[k, a if sc[k, a] > sc[k, b] else b] += 1
    if fixed:
        a, b = fixed[0]
        mywin[w] = (sc[:, a] > sc[:, b])
rank = (-(Wn * 10000 + PF)).argsort(axis=1).argsort(axis=1) + 1
mi = idx[ME]
# playoffs (6 teams): wk 15 3v6, 4v5 (1-2 bye); wk 16 1 v lowest seed left, 2 v other; wk 17 final
order = (-(Wn * 10000 + PF)).argsort(axis=1)  # seeds -> team index
P = {w: rng.normal(np.stack([S[t][:, WEEKS.index(w)] for t in teams], axis=1), SD) for w in (15, 16, 17)}
final = np.zeros((N, len(teams)), bool); champ = np.zeros((N, len(teams)), bool)
for k in range(N):
    sd_ = order[k]
    g = lambda w, a, b: a if P[w][k, a] > P[w][k, b] else b
    w36, w45 = g(15, sd_[2], sd_[5]), g(15, sd_[3], sd_[4])
    r1 = sorted([w36, w45], key=lambda t: list(sd_).index(t))
    f1, f2 = g(16, sd_[0], r1[1]), g(16, sd_[1], r1[0])
    final[k, f1] = final[k, f2] = True
    champ[k, g(17, f1, f2)] = True
made = rank <= n_po
res = pd.DataFrame({
    "Team": teams, "Record": [f"{a}-{b}" for a, b in zip(st.W, st.L)], "PF": st.PF,
    f"Wk {week}": [round(wk9[t], 1) for t in teams],
    f"Avg wk {week + 1}-{last_reg} (w/ injuries)": [round(S[t][:, :last_reg - week].mean(), 1) for t in teams],
    "Avg wk 15-17 (w/ injuries)": [round(S[t][:, -3:].mean(), 1) for t in teams],
    "Proj Wins": Wn.mean(axis=0).round(1),
    "Proj Final Record": [f"{w:.0f}-{14 - w:.0f}" for w in Wn.mean(axis=0).round(0)],
    "Proj Final PF": PF.mean(axis=0).round(0),
    "Avg Final Seed": rank.mean(axis=0).round(1),
    "Make Playoffs %": ((rank <= n_po).mean(axis=0) * 100).round(0),
    "Top-2 Bye %": ((rank <= 2).mean(axis=0) * 100).round(0),
    "Make Final %": (final.mean(axis=0) * 100).round(1),
    "Champion %": (champ.mean(axis=0) * 100).round(1),
    "Final % if in playoffs": [round(final[made[:, i], i].mean() * 100, 0) if made[:, i].any() else 0 for i in range(len(teams))],
    "Champion % if in playoffs": [round(champ[made[:, i], i].mean() * 100, 0) if made[:, i].any() else 0 for i in range(len(teams))],
}).sort_values("Make Playoffs %", ascending=False)
mine = pd.DataFrame([{"Week": w, "Opponent": SCHED[w][0][1],
                      "You": round(wk9[ME] if w == week else S[ME][:, WEEKS.index(w)].mean(), 1),
                      "Them": round(wk9[SCHED[w][0][1]] if w == week else S[SCHED[w][0][1]][:, WEEKS.index(w)].mean(), 1),
                      "Win %": round(mywin[w].mean() * 100)} for w in sorted(SCHED)])
# what the bench is worth: same sim without the bench players
bench = list(roster[(roster.Team == ME) & (roster.Slot == "BN")].Player)
nob = sim_team(list(roster[roster.Team == ME].Player), drop=bench)
depth = pd.DataFrame([{"Week": w, "With bench": round(S[ME][:, i].mean(), 1), "Without bench": round(nob[:, i].mean(), 1),
                       "Bench worth (pts)": round(S[ME][:, i].mean() - nob[:, i].mean(), 1)} for i, w in enumerate(WEEKS)])
fp = out / f"playoff_sim_{lg}_{season}_wk{week}.xlsx"
with pd.ExcelWriter(fp) as xw:
    res.to_excel(xw, sheet_name="Playoff Odds", index=False)
    mine.to_excel(xw, sheet_name=f"{ME} schedule", index=False)
    depth.to_excel(xw, sheet_name="Bench value", index=False)
    pd.DataFrame({"Notes": __doc__.strip().splitlines()}).to_excel(xw, sheet_name="Notes", index=False)
print(fp)
print(res.to_string(index=False))
print(mine.to_string(index=False))
print(depth.to_string(index=False))
w9 = mywin[week]
print(f"playoffs if win wk{week} {(rank[w9, mi] <= n_po).mean():.0%}, if loss {(rank[~w9, mi] <= n_po).mean():.0%}")
print("waiver", waiver)
