"""Rule 11: game days, opponents and byes in advice come from the nflverse schedule; injury tags belong to this week.
Exit 1 = a problem: do NOT send the advice until it is fixed.

  # facts to write advice from (team, opponent, day, date, bye, status) for this week and next:
  PYTHONPATH=src python3 scripts/check_schedule.py facts "Kenneth Walker III,Jaxon Smith-Njigba"
  # check drafted advice text before sending it (file path or the text itself):
  PYTHONPATH=src python3 scripts/check_schedule.py text advice.md
  # stale Questionable/Doubtful tags on rostered players in this week's ROS files (both leagues):
  PYTHONPATH=src python3 scripts/check_schedule.py tags [week]
  # manual notes (data/input/news_2026.tsv) vs. the live ESPN feed and this week's official report; --fix rewrites
  # the notes a newer source contradicts (a Scott note with no date, or dated this week, is left for a person):
  PYTHONPATH=src python3 scripts/check_schedule.py notes [--fix]

Week defaults to the week whose games are next (US Eastern date). Player teams come from the newest ROS file
(this season's rosters, rule 9). Options: --week N, --today YYYY-MM-DD.
"""
import datetime as dt
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from ffmodel import data
from ffmodel.config import DATA
from ffmodel.schedule_check import (apply_fixes, check_text, current_week, norm, notes_check, stale_tags,
                                    week_games)

SEASON = 2026
DIRS = [DATA / "output", Path("/mnt/project-files/ff-model")]


def opt(name, default=None):
    a = sys.argv
    if name in a:
        i = a.index(name)
        v = a[i + 1]
        del a[i:i + 2]
        return v
    return default


def newest(pattern):
    fs = [f for d in DIRS if d.exists() for f in d.glob(pattern)]
    key = lambda f: int(f.stem.split("wk")[1].split("_")[0])
    return max(fs, key=key) if fs else None


def ros_table(week, lg):
    f = None
    for d in DIRS:
        p = d / f"ros_{SEASON}_wk{week}_{lg}.xlsx"
        if p.exists():
            f = p
            break
    f = f or newest(f"ros_{SEASON}_wk*_{lg}.xlsx")
    if f is None:
        sys.exit(f"no ROS file for {lg}: run build_ros.py first (stale or missing data)")
    if f"wk{week}_" not in f.name:
        print(f"NOTE: using {f.name}; no week {week} ROS file for {lg} (stale data)")
    return pd.read_excel(f, sheet_name="Overall"), f


def owned(lg):
    out = set()
    for d in (DATA / "input", Path("/mnt/project-files/ff-model")):
        fs = sorted(d.glob(f"{lg}_rosters_{SEASON}wk*.csv"), key=lambda f: int(f.stem.split("wk")[1]))
        if fs:
            out |= set(pd.read_csv(fs[-1]).Player)
    return out


def espn_feed():
    """{norm name: dict(status, date, ret, comment)} from ESPN's league injury feed, or None if unreachable."""
    import json
    import urllib.request
    try:
        with urllib.request.urlopen("https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries", timeout=30) as r:
            d = json.load(r)
    except Exception as e:  # noqa: BLE001
        print(f"STALE DATA: ESPN injury feed unreachable ({e}); notes cannot be checked against it")
        return None
    out = {}
    for t in d.get("injuries", []):
        for i in t.get("injuries", []):
            a = i.get("athlete", {})
            rd = (i.get("details") or {}).get("returnDate")
            out[norm(a.get("displayName", ""))] = dict(
                status=i.get("status", ""), date=dt.date.fromisoformat(i["date"][:10]),
                ret=dt.date.fromisoformat(rd) if rd else None, comment=i.get("shortComment", ""))
    return out


def official_report(week):
    """{norm name: 'practice status / game status'} from this week's official NFL injury report (nflverse)."""
    try:
        inj = data.injuries([SEASON], refresh_current=True).to_pandas()
    except Exception as e:  # noqa: BLE001
        print(f"STALE DATA: official injury report unavailable ({e})")
        return {}
    inj = inj[inj.week == week]
    if inj.empty:
        print(f"NOTE: no official injury report for week {week} yet")
    return {norm(r.full_name): f"{r.practice_status or ''} / game status {r.report_status or 'not set yet'}"
            for r in inj.itertuples()}


def notes_mode(sched, week, fix):
    path = DATA / "input" / f"news_{SEASON}.tsv"
    news = pd.read_csv(path, sep="\t", quoting=3, dtype=str)
    feed = espn_feed()
    prev = week_games(sched, week - 1)
    start = max(g["date"] for g in prev.values()) + dt.timedelta(days=1)
    mine = owned("cbs") | owned("burbs")
    rows = notes_check(news, feed or {}, official_report(week), start, mine, SEASON)
    problems = []
    for x in sorted(rows, key=lambda x: (not x["owned"], x["player"])):
        line = f"{x['kind']}: {x['player']} | note: {x['note']} | newer: {x['newer']}"
        if x["owned"]:
            problems.append(line + ("" if x["fixable"] else " | needs a person to decide (Scott note with no newer date, practice report vs. note, or not on the feed)"))
        else:
            print("unowned, not blocking:", line)
    if fix and rows:
        new = apply_fixes(news, rows)
        new.to_csv(path, sep="\t", index=False, quoting=3)
        print(f"--fix: rewrote {sum(x['fixable'] for x in rows)} note(s) in {path.name}; "
              f"{sum(not x['fixable'] for x in rows)} left for Scott")
        problems = [p for p, x in zip(problems, [x for x in sorted(rows, key=lambda x: (not x["owned"], x["player"])) if x["owned"]])
                    if not x["fixable"]]
    if feed is None:
        problems.append("ESPN feed unreachable: notes not verified (stale-data rule)")
    return problems


def main():
    today = dt.date.fromisoformat(opt("--today")) if "--today" in sys.argv else dt.datetime.now(ZoneInfo("America/New_York")).date()
    wk_opt = opt("--week")
    sched = data.schedules([SEASON]).to_pandas()
    week = int(wk_opt) if wk_opt else current_week(sched, today)
    mode = sys.argv[1] if len(sys.argv) > 1 else "tags"
    tabs = {lg: ros_table(week, lg)[0] for lg in ("cbs", "burbs")}
    teams = {}
    for t in tabs.values():
        teams.update(dict(zip(t.Player, t.Team)))
    problems = []
    if mode == "facts":
        names = [x.strip() for x in sys.argv[2].split(",") if x.strip()]
        st = tabs["cbs"].set_index("Player")
        for n in names:
            if n not in teams:
                problems.append(f"{n}: not in the ROS files (check spelling / current team)")
                continue
            t = teams[n]
            for w in (week, week + 1):
                g = week_games(sched, w).get(t)
                s = f"week {w}: " + (f"{g['weekday']} {g['date']:%b %d} {'vs' if g['home'] else 'at'} {g['opp']}" if g else "BYE")
                if w == week and n in st.index:
                    s += f"  status: {st.loc[n, 'Status'] if pd.notna(st.loc[n, 'Status']) else 'healthy'}"
                print(f"{n} ({t}) {s}")
    elif mode == "text":
        arg = sys.argv[2]
        text = Path(arg).read_text() if len(arg) < 255 and Path(arg).is_file() else arg
        problems = check_text(text, teams, sched, today, week, owned("cbs") | owned("burbs"))
    elif mode == "tags":
        if len(sys.argv) > 2:
            week = int(sys.argv[2])
        for lg in ("cbs", "burbs"):
            t, _ = ros_table(week, lg)
            problems += [f"[{lg}] {p}" for p in stale_tags(t, sched, week, owned(lg))]
    elif mode == "notes":
        problems = notes_mode(sched, week, "--fix" in sys.argv)
    else:
        sys.exit(__doc__)
    for p in problems:
        print("SCHEDULE VIOLATION:", p)
    print(f"schedule ok (week {week}, today {today})" if not problems else
          f"{len(problems)} problem(s): do NOT give this advice until fixed")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
