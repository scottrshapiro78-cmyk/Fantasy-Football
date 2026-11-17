"""Standing rules for Scott's leagues, checked automatically. Exit 1 on any violation.

PYTHONPATH=src python3 scripts/check_rules.py [week=11]
Rules (add new ones here and in docs/rules.md; weekly_refresh.py and trade_finder.py call this):
 1. Keep-list (data/input/trade_keep.txt) players never appear in a trade's "You give" or "Drop" column.
 2. Injured players (Out / IR / Doubtful / Out for season) are never offered in a trade.
 3. Every player Scott gave news on (a note containing "Scott" in data/input/news_2026.tsv) carries that status
    in the rankings (his word outranks the feed until a newer official report).
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


if __name__ == "__main__":
    wk = int(sys.argv[1]) if len(sys.argv) > 1 else 11
    problems = check_trades(wk) + check_scott_news(wk)
    for p in problems:
        print("RULE VIOLATION:", p)
    print("rules ok" if not problems else f"{len(problems)} rule violation(s)")
    sys.exit(1 if problems else 0)
