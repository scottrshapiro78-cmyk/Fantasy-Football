"""Rest-of-season rankings for both leagues.

PYTHONPATH=src python3 scripts/build_ros.py [season=2026] [thru_week=8] [--refit]
Writes data/output/ros_<season>_wk<next>_<league>.xlsx: Overall, QB, RB, WR, TE, K, DEF, Notes, Column Guide.
"""
import sys

import polars as pl
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill

from ffmodel import ros
from ffmodel.board import _tiers
from ffmodel.config import DATA
from ffmodel.guide import GUIDE, add_guide
from ffmodel.value import value

args = [a for a in sys.argv[1:] if not a.startswith("--")]
season = int(args[0]) if args else 2026
thru = int(args[1]) if len(args) > 1 else 8
if "--refit" in sys.argv or not (DATA / "processed" / "ros_fit.json").exists():
    ros.fit(thru)
POS = ["QB", "RB", "WR", "TE", "K", "DEF"]
TIER_SHAPE = {"QB": (20, 5), "RB": (45, 7), "WR": (60, 8), "TE": (20, 5), "K": (14, 3), "DEF": (14, 3)}
TEAMS = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers", "CHI": "Chicago Bears", "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys", "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars", "KC": "Kansas City Chiefs",
    "LV": "Las Vegas Raiders", "LAC": "Los Angeles Chargers", "LA": "Los Angeles Rams", "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings", "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers", "SF": "San Francisco 49ers",
    "SEA": "Seattle Seahawks", "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}
STATUS = {"active": "", "out_week": "Out this week", "out_long": "IR / PUP / suspended"}
GUIDE.update({
    "Status": "Roster status right now. 'IR / PUP / suspended' = on a reserve list; 'Out this week' = ruled out or doubtful on this week's injury report. Both lower expected games.",
    "ROS PPG": "Projected points per game for the rest of the season: a blend of the preseason projection and this season's results so far (more games played = more weight on this season).",
    "Preseason PPG": "What the model projected per game before the season (draft night).",
    "PPG So Far": "Actual points per game this season.",
    "Games Played": "Games he has played this season.",
    "Games Left (reg season)": "His team's games left in your regular season (through week 14), byes taken out.",
    "Playoff Games (wk 15-17)": "His team's games in fantasy playoff weeks 15-17.",
    "Exp Games Left": "Games left (weeks through 17) he is expected to actually play, after injury risk and his current status.",
    "ROS Pts": "Projected points for the rest of the season through week 17: ROS PPG x expected games left.",
    "Role": "'Role up' / 'Role down' = his usage in the last 3 games is worth 3+ points per game more / less than before, both on average and on a typical (median) game, so one huge game can't trigger it.",
    "Role Trend (xFP)": "Usage trend: expected fantasy points per game from his targets and carries (and where on the field) in his last 3 games, minus his season average. Part of it is built into ROS PPG.",
    "Schedule Adj (PPG)": "How much his remaining opponents add (+) or take away (-) per game, based on how those defenses have played this season against his position (run defense for RBs, pass defense for QB/WR/TE; for team defenses, the offenses they still face).",
    "ROS VORP": "Rest-of-season value over replacement: extra points over a waiver-level player at his position for the rest of the season. Drives the ranking; use it to compare trades.",
})

out_dir = DATA / "output"
for lg in ("burbs", "cbs"):
    d = ros.project(season, thru, lg)
    col = f"pred_ppg_{lg}"
    d = d.with_columns(((pl.col("exp_reg") + pl.col("exp_po")) / 13).alias("exp_frac"))
    b, repl, depth = value(d, lg, option=True)
    parts = []
    for p in POS:
        s = b.filter(pl.col("position") == p).sort("vorp", descending=True)
        parts.append(s.with_columns(pl.int_range(1, pl.len() + 1).alias("pos_rank"),
                                    pl.Series("pos_tier", _tiers(s["vorp"], *TIER_SHAPE[p]))))
    b = pl.concat(parts).sort("vorp", descending=True).with_columns(pl.int_range(1, pl.len() + 1).alias("overall_rank"))
    b = b.with_columns(pl.when(pl.col("position") == "DEF")
                         .then(pl.col("team").replace_strict(TEAMS, default=pl.col("player_display_name")))
                         .otherwise(pl.col("player_display_name")).alias("player_display_name"))
    r1 = lambda c: pl.col(c).round(1)
    df = b.select(
        pl.col("overall_rank").alias("Overall Rk"), pl.col("position").alias("Pos"), pl.col("pos_rank").alias("Pos Rk"),
        pl.col("pos_tier").alias("Pos Tier"), pl.col("player_display_name").alias("Player"), pl.col("team").alias("Team"),
        pl.col("status").replace_strict(STATUS, default="").alias("Status"),
        r1(col).alias("ROS PPG"),
        pl.when((pl.col("role_trend") >= 3) & (pl.col("xfp_med_last3") - pl.col("xfp_med_before") >= 3))
          .then(pl.lit("Role up"))
          .when((pl.col("role_trend") <= -3) & (pl.col("xfp_med_last3") - pl.col("xfp_med_before") <= -3))
          .then(pl.lit("Role down")).otherwise(pl.lit("")).alias("Role"),
        r1("role_trend").alias("Role Trend (xFP)"), r1("sched_adj").alias("Schedule Adj (PPG)"),
        r1("prior").alias("Preseason PPG"), r1(f"ppg_{lg}_sofar").alias("PPG So Far"),
        pl.col("n").alias("Games Played"), pl.col("games_left_reg").alias("Games Left (reg season)"),
        pl.col("games_left_po").alias("Playoff Games (wk 15-17)"), r1("exp_games").alias("Exp Games Left"),
        (pl.col(col) * pl.col("exp_games")).round(1).alias("ROS Pts"), r1("repl_ppg").alias("Repl PPG"),
        r1("vorp").alias("ROS VORP"),
        pl.when(pl.col("rookie")).then(pl.lit("Rookie")).otherwise(pl.lit("")).alias("Note"),
        pl.col("player_id").alias("nflverse ID"))
    path = out_dir / f"ros_{season}_wk{thru + 1}_{lg}.xlsx"
    wb = Workbook()

    def sheet(ws, frame, freeze):
        ws.append(frame.columns)
        for c in ws[1]:
            c.font, c.fill = Font(bold=True, color="FFFFFF"), PatternFill("solid", fgColor="1F3864")
        for row in frame.iter_rows():
            ws.append(list(row))
        ws.freeze_panes = freeze
        for c in ws.columns:
            ws.column_dimensions[c[0].column_letter].width = max(9, min(26, max(len(str(x.value or "")) for x in c[:60]) + 2))

    ws = wb.active
    ws.title = "Overall"
    sheet(ws, df, "F2")
    for p in POS:
        sub = df.filter(pl.col("Pos") == p).sort("Pos Rk").drop("Pos")
        sheet(wb.create_sheet(p), sub.select(["Pos Rk"] + [c for c in sub.columns if c != "Pos Rk"]), "D2")
    ws = wb.create_sheet("Notes")
    for n in [
        f"{'THE BURBS' if lg == 'burbs' else 'CBS'}: rest-of-season rankings for {season}, weeks {thru + 1}-17, using games through week {thru}.",
        "Ranked by ROS VORP: extra points over a waiver-level player at the position for the rest of the season.",
        "ROS PPG blends the preseason projection with this season's results. After 8 games this season counts about",
        "60% for QB/RB and 70-75% for WR/TE (fitted on 2021-2025); kickers and defenses still lean mostly on the",
        "preseason projection because their weekly scoring is so noisy.",
        "Role: usage over the last 3 games vs the season (expected points from targets and carries) also counts for RBs and TEs,",
        "where it improved the backtest; for QBs and WRs it did not help, so it is shown but not used.",
        "Schedule: remaining opponents' defense vs the position this season (run D for RBs, pass D for QB/WR/TE),",
        "and the offenses left for team defenses. It helped most for defenses and TEs, a little for RBs/WRs, not for QBs.",
        "Expected games count byes, current IR / suspension / injury-report status and normal injury risk.",
        "A game not yet played this week (e.g. Monday night) counts as a game left.",
        "Playoff weeks 15-17 are counted the same as regular-season weeks.",
        "Tested on 2021-2025 (Burbs, error in PPG for weeks 9-17): QB 3.43 vs 3.82 preseason-only / 3.81 season-so-far-only;",
        "RB 3.01 vs 3.66 / 3.40; WR 2.99 vs 3.42 / 3.27; TE 2.44 vs 2.98 / 2.71; DEF 2.06 vs 2.15 / 2.93.",
        "Rerun each week: PYTHONPATH=src python3 scripts/build_ros.py 2026 <last completed week>",
    ]:
        ws.append([n])
    ws.column_dimensions["A"].width = 115
    wb.save(path)
    add_guide(path)
    print(path, df.height)
