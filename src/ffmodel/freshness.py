"""Data check: is every input current? Flags what's stale so outputs and notes can say so."""
from __future__ import annotations

import datetime as dt
import glob
import os

import polars as pl

from . import data
from .config import DATA


def checks(season: int, week: int) -> list[dict]:
    """week = the upcoming / current NFL week the outputs are for. Returns rows: Item, OK, Detail."""
    out = []
    today = dt.date.today()

    def add(item, ok, detail):
        out.append({"Item": item, "OK": "yes" if ok else "CHECK", "Detail": detail})

    s = data.schedules([season]).filter(pl.col("game_type") == "REG")
    done = s.filter(pl.col("home_score").is_not_null())
    last_done = int(done.filter(pl.col("week") < week)["week"].max() or 0)
    w = pl.read_parquet(DATA / "processed" / "player_weeks.parquet").filter(pl.col("season") == season)
    stats_wk = int(w["week"].max() or 0)
    add("Game stats", stats_wk >= last_done, f"stats through week {stats_wk}; last completed week {last_done}")
    # this week's finished games (e.g. Thursday) that aren't in the stats yet
    played_now = done.filter(pl.col("week") == week).height
    have_now = w.filter(pl.col("week") == week)["team"].n_unique()
    add(f"Week {week} games already played", have_now >= 2 * played_now,
        f"{played_now} game(s) final, stats for {have_now // 2} game(s)")
    r = data.rosters_weekly([season])
    rw = int(r["week"].max() or 0)
    add("NFL rosters / trades", rw >= week, f"team rosters through week {rw} (trades and signings after that are missing)")
    inj = data.injuries([season])
    iw = int(inj["week"].max() or 0)
    final = inj.filter(pl.col("week") == week, pl.col("report_status").is_not_null()).height
    dow = today.weekday()  # Mon 0 .. Sun 6
    if iw < week:
        add("Injury report", False, f"no week {week} practice report yet (latest week {iw}); reports start Wednesday")
    elif final == 0 and dow in (4, 5, 6):
        add("Injury report", False, f"week {week} practice reports in, but no game statuses yet (final report comes Friday)")
    else:
        add("Injury report", True, f"week {week}: {'final game statuses in' if final else 'practice reports in, game statuses come Friday'}")
    sn = int(data.snap_counts([season])["week"].max() or 0)
    op = int(data.opportunity([season])["week"].cast(pl.Int32).max() or 0)
    add("Roles (snaps, usage)", min(sn, op) >= last_done, f"snap counts through week {sn}, usage through week {op}")
    dc = data.depth_charts([season])[0]
    asof = dt.datetime.fromisoformat(str(dc["dt"].max()).replace("Z", "+00:00"))
    hrs = (dt.datetime.now(dt.timezone.utc) - asof).total_seconds() / 3600
    add("Depth charts (starters)", hrs <= 36, f"ESPN depth chart as of {asof:%a %b %d %H:%M} UTC ({hrs:.0f}h old)")
    nf = DATA / "input" / f"news_{season}.tsv"
    if nf.exists():
        age = (dt.datetime.now() - dt.datetime.fromtimestamp(os.path.getmtime(nf))).days
        add("Injury / news notes", age <= 3, f"last updated {age} day(s) ago")
    from .ros import sleeper_injuries
    add("Live injury feed", sleeper_injuries() is not None,
        "Sleeper feed reachable" if sleeper_injuries() is not None else "live feed blocked here; using the official report + your pastes")
    for lg in ("burbs", "cbs"):
        import re
        fs = sorted(glob.glob(str(DATA / "input" / f"{lg}_rosters_{season}wk*.csv")),
                    key=lambda f: int(re.search(r"wk(\d+)", f).group(1)))
        if fs:
            age = (dt.datetime.now() - dt.datetime.fromtimestamp(os.path.getmtime(fs[-1]))).days
            add(f"{lg} league rosters", age <= 7, f"{os.path.basename(fs[-1])}, pasted {age} day(s) ago (pickups / trades since then are missing)")
        else:
            add(f"{lg} league rosters", False, "none pasted yet")
    add("Gameday inactives", False, "not in the data feeds; checked by the Sunday-morning sweep (or tell me)")
    return out


def add_sheet(path, rows: list[dict]) -> None:
    """Append a Data Check tab to an output workbook."""
    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill
    wb = load_workbook(path)
    if "Data Check" in wb.sheetnames:
        del wb["Data Check"]
    ws = wb.create_sheet("Data Check")
    ws.append(["Item", "OK", "Detail", f"checked {dt.datetime.now():%a %b %d %H:%M} UTC"])
    for c in ws[1]:
        c.font = Font(bold=True)
    for r in rows:
        ws.append([r["Item"], r["OK"], r["Detail"]])
        if r["OK"] != "yes":
            ws.cell(ws.max_row, 2).fill = PatternFill("solid", fgColor="FFE699")
    ws.column_dimensions["A"].width, ws.column_dimensions["C"].width = 28, 95
    wb.save(path)


def summary(rows: list[dict]) -> str:
    bad = [r for r in rows if r["OK"] != "yes" and r["Item"] != "Gameday inactives"]
    return "all inputs current" if not bad else "; ".join(f"{r['Item']}: {r['Detail']}" for r in bad)
