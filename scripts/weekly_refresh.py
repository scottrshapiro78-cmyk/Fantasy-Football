"""One-command weekly rebuild (replaces the step-by-step routine).

PYTHONPATH=src python3 scripts/weekly_refresh.py [--copy-to /mnt/project-files/ff-model]

Refreshes 2026 data, finds the last completed week W, builds features/dst tables, ROS (W) and weekly (W+1),
scorecard (W), and prints a short summary plus the DATA CHECK lines. Does NOT do the web news sweep;
that stays manual and optional.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import nflreadpy as nfl
import polars as pl

from ffmodel import data, dst, features, ros
from ffmodel.config import DATA

ROOT = Path(__file__).resolve().parent.parent
SEASON = 2026


def run(*args):
    r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True,
                       env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
    if r.returncode:
        print(r.stdout[-1500:], r.stderr[-1500:])
        sys.exit(f"FAILED: {' '.join(args)}")
    return r.stdout


def last_completed_week() -> int:
    s = nfl.load_schedules([SEASON]).filter(pl.col("game_type") == "REG")
    done = s.group_by("week").agg(pl.col("home_score").is_not_null().all().alias("d")).filter("d")
    return int(done["week"].max())


def main():
    for f in (data.rosters_weekly, data.opportunity, data.team_weeks, data.injuries, data.snap_counts, data.depth_charts):
        f([SEASON], refresh_current=True)
    run("scripts/build_history.py")
    out = DATA / "processed"
    features.build().write_parquet(out / "features.parquet")
    dst.team_seasons().write_parquet(out / "dst_seasons.parquet")
    (DATA / "output").mkdir(parents=True, exist_ok=True)
    w = last_completed_week()
    print(f"feed: {ros.injury_feed(SEASON)[1]}; sleeper ok: {ros.sleeper_injuries() is not None}")
    logs = [run("scripts/build_ros.py", str(SEASON), str(w), "--refit"),
            run("scripts/build_weekly.py", str(SEASON), str(w + 1)),
            run("scripts/scorecard.py", str(SEASON), str(w))]
    print(f"last completed week {w}; built ROS (through {w}), weekly {w + 1}, scorecard {w}")
    for l in logs:
        for line in l.splitlines():
            if line.startswith("DATA CHECK"):
                print(line)
    chk = subprocess.run([sys.executable, "scripts/check_rules.py", str(w + 1)], cwd=ROOT, capture_output=True, text=True,
                         env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
    print(chk.stdout.strip())
    if "--copy-to" in sys.argv:
        dest = Path(sys.argv[sys.argv.index("--copy-to") + 1])
        for p in (DATA / "output").glob(f"*_wk{w + 1}_*.xlsx"):
            shutil.copy(p, dest)
        for p in (DATA / "output").glob("accuracy_2026_*.xlsx"):
            shutil.copy(p, dest)
        print("copied to", dest)


main()
