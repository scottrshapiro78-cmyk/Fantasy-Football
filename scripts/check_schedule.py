"""Rule 11: game days, opponents and byes in advice come from the nflverse schedule; injury tags belong to this week.
Exit 1 = a problem: do NOT send the advice until it is fixed.

  # facts to write advice from (team, opponent, day, date, bye, status) for this week and next:
  PYTHONPATH=src python3 scripts/check_schedule.py facts "Kenneth Walker III,Jaxon Smith-Njigba"
  # check drafted advice text before sending it (file path or the text itself):
  PYTHONPATH=src python3 scripts/check_schedule.py text advice.md
  # stale Questionable/Doubtful tags on rostered players in this week's ROS files (both leagues):
  PYTHONPATH=src python3 scripts/check_schedule.py tags [week]

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
from ffmodel.schedule_check import check_text, current_week, stale_tags, week_games

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
    else:
        sys.exit(__doc__)
    for p in problems:
        print("SCHEDULE VIOLATION:", p)
    print(f"schedule ok (week {week}, today {today})" if not problems else
          f"{len(problems)} problem(s): do NOT give this advice until fixed")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
