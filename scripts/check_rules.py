"""Standing rules for Scott's leagues, checked automatically. Exit 1 on any violation.

PYTHONPATH=src python3 scripts/check_rules.py [week=11]
Rules (add new ones here and in docs/rules.md; weekly_refresh.py and trade_finder.py call this):
 1. Keep-list (data/input/trade_keep.txt) players never appear in a trade's "You give" or "Drop" column.
 2. Injured players (Out / IR / Doubtful / Out for season) are never offered in a trade.
 3. Every player Scott gave news on (a note containing "Scott" in data/input/news_2026.tsv) carries that status
    in the rankings (his word outranks the feed until a newer official report).
 6. Add/drop plans are tested week by week with byes and injuries: `check_rules.py plan <week> "adds" "drops"`.
 5. No stale injury news (return date long past).
 4. Add/drop plans (check_plan) must fit the Yahoo roster: 1 QB 2 RB 3 WR 1 TE 1 FLEX 1 K 1 DEF + 5 BN (+1 IR),
    never leave the lineup without a K, DEF, TE or QB, and never drop a keep-list player.
"""
import sys

import pandas as pd

from ffmodel.config import DATA, load_leagues

OUT = ("Out", "Injured Reserve", "Doubtful", "IR / PUP / suspended", "Out for season")


def keep_list():
    p = DATA / "input" / "trade_keep.txt"
    return {l.strip() for l in p.read_text().splitlines() if l.strip() and not l.startswith("#")} if p.exists() else set()


def check_trades(week):
    f = DATA / "output" / f"trades_burbs_2026_wk{week}_playoffs.xlsx"
    if not f.exists():
        return []
    ros = pd.read_excel(DATA / "output" / f"ros_2026_wk{week}_burbs.xlsx", sheet_name="Overall").set_index("Player")
    bad = []
    for sheet in ("Best for you", "Most balanced"):
        d = pd.read_excel(f, sheet_name=sheet).fillna("")
        for r in d.itertuples():
            for p in [x for x in str(r._2).split(" + ")] + ([r.Drop] if r.Drop else []):
                if p in keep_list():
                    bad.append(f"keep-list player {p} in trade ({r.Partner})")
            for p in str(r._2).split(" + "):
                if p in ros.index and str(ros.loc[p, "Status"]) in OUT:
                    bad.append(f"injured player {p} offered ({r.Partner})")
    return sorted(set(bad))


def check_scott_news(week):
    news = pd.read_csv(DATA / "input" / "news_2026.tsv", sep="\t").fillna("")
    ros = pd.read_excel(DATA / "output" / f"ros_2026_wk{week}_burbs.xlsx", sheet_name="Overall").set_index("Player")
    bad = []
    for r in news[news.Note.str.contains("Scott")].itertuples():
        got = str(ros.loc[r.Player, "Status"]) if r.Player in ros.index else r.Status
        if got == "Out for season" and r.Status in ("Out", "Injured Reserve") and r.Return in ("Feb 15", "Jan 15"):
            continue  # same thing: out through the end of the season
        if got != r.Status:
            bad.append(f"{r.Player}: Scott news says {r.Status}, rankings show {got}")
    return bad


def check_stale(week):
    """Rule 5: a status with a return date already past (before this week's first game) must not still say Out/Questionable."""
    import datetime as dt
    news = pd.read_csv(DATA / "input" / "news_2026.tsv", sep="\t").fillna("")
    bad = []
    mons = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    # the week's first game day: Thursday; news rows whose return date is more than 6 days before it are stale
    import nflreadpy as nfl, polars as pl
    first = min(nfl.load_schedules([2026]).filter(pl.col("week") == week)["gameday"].to_list())
    first = dt.date.fromisoformat(first)
    owned = set(pd.read_csv(DATA / "input" / f"burbs_rosters_2026wk{week}.csv").Player)  # only players someone owns matter
    for r in news[news.Player.isin(owned)].itertuples():
        m = str(r.Return).split()
        if len(m) == 2 and m[0] in mons and r.Status in ("Out", "Questionable", "Doubtful"):
            d = dt.date(2026 + (1 if mons.index(m[0]) < 2 else 0), mons.index(m[0]) + 1, int(m[1]))
            if (first - d).days > 6 and "Scott" not in r.Note:
                bad.append(f"stale news: {r.Player} {r.Status}, return {r.Return} (before {first}), note '{r.Note[:40]}'")
    return bad


def my_roster(week):
    """Roster ledger: base file + moves in data/input/burbs_my_moves.csv (action,Player,Pos). Single source of truth."""
    from ffmodel import ledger
    base = ledger.apply(pd.read_csv(DATA / "input" / f"burbs_rosters_2026wk{week}.csv").fillna(""))
    return list(base[base.Team == ledger.ME].Player)


def check_plan(roster, adds, drops, pos, league="burbs"):
    """roster: list of player names; pos: {name: position}. Returns a list of violations."""
    slots = load_leagues()[league]["roster"]
    new = [p for p in roster if p not in drops] + list(adds)
    bad = [f"cannot drop keep-list player {p}" for p in drops if p in keep_list()]
    cap = sum(v for k, v in slots.items() if k != "IR") + 0
    if len(new) > cap:
        bad.append(f"roster {len(new)} > {cap} spots")
    for need, n in (("QB", 1), ("TE", 1), ("K", 1), ("DEF", 1)):
        if sum(pos.get(p) == need for p in new) < n:
            bad.append(f"no {need} left after the move")
    if sum(pos.get(p) == "RB" for p in new) < 2 or sum(pos.get(p) == "WR" for p in new) < 3:
        bad.append("not enough RB/WR to fill the lineup")
    return bad


def plan_lineup_check(adds, drops, week, horizon=2):
    """Rule 6: after the add/drop, each of the next `horizon` weeks must still field a full legal starting lineup from
    players who are NOT on bye and NOT Out/IR/Doubtful. Returns a list of violations (empty = ok)."""
    import nflreadpy as nfl, polars as pl
    ros = pd.read_excel(DATA / "output" / f"ros_2026_wk{week}_burbs.xlsx", sheet_name="Overall").set_index("Player")
    sch = nfl.load_schedules([2026]).filter(pl.col("game_type") == "REG")
    cur = [p for p in my_roster(week) if p not in drops] + [a for a in adds if a not in my_roster(week)]
    slots = load_leagues()["burbs"]["roster"]
    bad = [f"cannot drop keep-list player {p}" for p in drops if p in keep_list()]
    cap = sum(v for k, v in slots.items() if k != "IR")
    if len(cur) > cap:
        bad.append(f"roster {len(cur)} > {cap} spots")
    for w in range(week, week + horizon):
        playing = set(sch.filter(pl.col("week") == w)["home_team"].to_list() + sch.filter(pl.col("week") == w)["away_team"].to_list())
        ok = {}
        for p in cur:
            if p not in ros.index:
                continue
            r = ros.loc[p]
            if r["Team"] in playing and str(r["Status"]) not in OUT:
                ok.setdefault(r["Pos"], []).append(p)
        for pos, n in (("QB", 1), ("RB", 2), ("WR", 3), ("TE", 1), ("K", 1), ("DEF", 1)):
            if len(ok.get(pos, [])) < n:
                bad.append(f"week {w}: only {len(ok.get(pos, []))} available {pos} (need {n}); byes/injuries counted")
        flex = len(ok.get("RB", [])) - 2 + len(ok.get("WR", [])) - 3 + len(ok.get("TE", [])) - 1
        if flex < 1:
            bad.append(f"week {w}: no player left for FLEX")
    return bad


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "plan":
    # python scripts/check_rules.py plan <week> "Add1,Add2" "Drop1,Drop2"
    wk = int(sys.argv[2]); adds = [x for x in sys.argv[3].split(",") if x]; drops = [x for x in sys.argv[4].split(",") if x]
    res = plan_lineup_check(adds, drops, wk)
    for r in res:
        print("PLAN VIOLATION:", r)
    print("plan ok" if not res else f"{len(res)} problem(s): do NOT give this advice")
    sys.exit(1 if res else 0)

if __name__ == "__main__":
    wk = int(sys.argv[1]) if len(sys.argv) > 1 else 11
    problems = check_trades(wk) + check_scott_news(wk) + check_stale(wk)
    for p in problems:
        print("RULE VIOLATION:", p)
    print("rules ok" if not problems else f"{len(problems)} rule violation(s)")
    sys.exit(1 if problems else 0)
