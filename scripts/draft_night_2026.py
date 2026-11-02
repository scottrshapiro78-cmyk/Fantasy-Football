"""2026 draft-night boards (model fit only on data before 2026) plus how each player is doing so far.

Run build_boards.py <workbook> 2026 first, then:
PYTHONPATH=src python3 scripts/draft_night_2026.py <auction workbook.xlsx>
Writes data/output/draft_night_2026_burbs.xlsx and draft_night_2026_cbs.xlsx.
"""
import sys

import openpyxl
import polars as pl
from openpyxl.styles import Font, PatternFill

from ffmodel.config import DATA
from ffmodel.market import _norm, sales

wb_path = sys.argv[1]
s = pl.read_parquet(DATA / "processed" / "player_seasons.parquet").filter(pl.col("season") == 2026)
d = pl.read_parquet(DATA / "processed" / "dst_seasons.parquet").filter(pl.col("season") == 2026)
weeks = int(pl.read_parquet(DATA / "processed" / "player_weeks.parquet").filter(pl.col("season") == 2026)["week"].max())
actual = pl.concat([
    s.select("player_id", pl.col("position").replace({"FB": "RB"}), "games", "fpts_burbs", "fpts_cbs"),
    d.select(pl.col("team").alias("player_id"), pl.lit("DEF").alias("position"), "games",
             (pl.col("ppg_burbs") * pl.col("games")).alias("fpts_burbs"),
             (pl.col("ppg_cbs") * pl.col("games")).alias("fpts_cbs")),
], how="vertical_relaxed")
for lg in ("burbs", "cbs"):
    actual = actual.with_columns(
        pl.col(f"fpts_{lg}").rank("min", descending=True).over("position").cast(pl.Int32).alias(f"rk_{lg}"))
act = {r["player_id"]: r for r in actual.iter_rows(named=True)}
paid = {r["key"]: r["price"] for r in sales(wb_path, 2026).iter_rows(named=True)}

GOLD = PatternFill("solid", fgColor="7F6000")
for lg in ("burbs", "cbs"):
    wb = openpyxl.load_workbook(DATA / "output" / f"draft_boards_2026_{lg}.xlsx")
    for ws in wb.worksheets:
        if ws.title == "Notes":
            notes = [
                "DRAFT NIGHT 2026: these are the boards the model would have produced before the 2026 season,",
                "fit only on data through 2025 (no peeking). Teams shown are 2025 teams.",
                f"Gold columns are added afterwards: what has actually happened through week {weeks} of 2026.",
                f"Actual Pts / PPG / Games = 2026 results so far; Actual Pos Rk = rank by total points at the position.",
                "Paid in 2026 = the price the player sold for in The Burbs 2026 auction (QB/RB/WR/TE only).",
                "",
            ]
            ws.insert_rows(1, len(notes))
            for i, n in enumerate(notes, 1):
                ws.cell(i, 1, n)
            continue
        head = [c.value for c in ws[1]]
        id_col, name_col = head.index("nflverse ID") + 1, head.index("Player") + 1
        pos_col = head.index("Pos") + 1 if "Pos" in head else None
        new = (["Paid in 2026 $"] if lg == "burbs" else []) + [
            f"Actual Pts (wk 1-{weeks})", "Actual PPG", "Actual Games", "Actual Pos Rk", "Pos Rk Diff (+ = beat)"]
        start = ws.max_column + 1
        for j, h in enumerate(new):
            c = ws.cell(1, start + j, h)
            c.font, c.fill = Font(bold=True, color="FFFFFF"), GOLD
            ws.column_dimensions[c.column_letter].width = 14
        rk_col = head.index("Pos Rk") + 1
        for r in range(2, ws.max_row + 1):
            pid = ws.cell(r, id_col).value
            a = act.get(pid)
            vals = []
            if lg == "burbs":
                pos = ws.cell(r, pos_col).value if pos_col else ws.title
                vals.append(paid.get(_norm(str(ws.cell(r, name_col).value))) if pos in ("QB", "RB", "WR", "TE") else None)
            if a and a["games"]:
                vals += [round(a[f"fpts_{lg}"], 1), round(a[f"fpts_{lg}"] / a["games"], 1), a["games"], a[f"rk_{lg}"],
                         ws.cell(r, rk_col).value - a[f"rk_{lg}"]]
            else:
                vals += [0, None, 0, None, None]
            for j, v in enumerate(vals):
                ws.cell(r, start + j, v)
    wb.save(DATA / "output" / f"draft_night_2026_{lg}.xlsx")
print("weeks", weeks)
