"""Build both draft boards and the Draft Optimizer export.

PYTHONPATH=src python3 scripts/build_boards.py <auction workbook.xlsx> [target, default 2027] [market.csv]
Writes data/output/draft_boards_<target>.xlsx with sheets: Burbs Auction, CBS Snake, Notes.
"""
import sys

import polars as pl
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill

from ffmodel.board import burbs_board, cbs_board
from ffmodel.config import DATA

wb_path = sys.argv[1]
target = int(sys.argv[2]) if len(sys.argv) > 2 else 2027
market_csv = sys.argv[3] if len(sys.argv) > 3 else None  # optional: Player,Price expected 2027 prices
r1 = lambda c: pl.col(c).round(1)

b = burbs_board(target, wb_path, market_csv)
burbs = b.select(
    pl.col("position").alias("Pos"), pl.col("player_display_name").alias("Player"), pl.col("team").alias("Team"),
    pl.col("overall_rank").alias("Overall Rk"), pl.col("pos_rank").alias("Pos Rk"),
    pl.col("bid").alias("Max Bid (Suggested)"), pl.col("max_bid").alias("Max Bid $"),
    pl.col("suggested_bid").alias("Suggested $"),
    pl.col("market_value").round(0).alias("League $"), pl.col("bargain").alias("Bargain $"),
    pl.col("flag").alias("Flag"),
    r1("pred_ppg_burbs").alias("PPG"), r1("exp_games").alias("Exp Games (of 13)"),
    (pl.col("pred_ppg_burbs") * pl.col("exp_games")).round(1).alias("Proj Pts (wk 1-14)"),
    r1("repl_ppg").alias("Repl PPG"), r1("vorp").alias("VORP"),
    pl.col("risk_score").alias("Injury Risk (0-100)"),
    pl.when(pl.col("rookie")).then(pl.lit("Rookie")).otherwise(pl.lit("")).alias("Note"),
    pl.col("player_id").alias("nflverse ID"))

c = cbs_board(target)
cbs = c.select(
    pl.int_range(1, pl.len() + 1).alias("Overall Rk"), pl.col("tier").alias("Tier"),
    pl.col("position").alias("Pos"), pl.col("pos_rank").alias("Pos Rk"),
    pl.col("player_display_name").alias("Player"), pl.col("team").alias("Team"),
    r1("pred_ppg_cbs").alias("PPG"), r1("exp_games").alias("Exp Games (of 13)"),
    r1("vorp").alias("VORP"), pl.col("risk_score").alias("Injury Risk (0-100)"),
    pl.when(pl.col("rookie")).then(pl.lit("Rookie")).otherwise(pl.lit("")).alias("Note"),
    pl.col("player_id").alias("nflverse ID"))

notes = [
    f"Draft boards for {target}, built from nflverse data through the latest completed games.",
    "THE BURBS (auction): 'Max Bid (Suggested)' = the most to pay, and where to aim.",
    "Max bid = the model's value of the player (value over replacement in dollars). Never bid past it.",
    "League $ = what this league usually pays for that position slot (2025-2026 average).",
    "Suggested = the league's usual price, capped at 90% of the max bid.",
    "Bargain $ = max bid - league price. 'Bargain' = model value at least $5 over the league price;",
    "'League overpays' = at least $5 under. Expect to let those go.",
    "League $ is a slot price by projected rank, not a player-specific price. Load real 2027",
    "ADP or expert dollar values in August (a Player,Price file) to make it player-specific.",
    "K and DEF are capped at $3: none sold above $4 in two years.",
    "CBS (snake): ranked by value over replacement (VORP); a new tier starts at a big drop in value.",
    "VORP = expected games x (projected PPG - replacement PPG). Injury risk works only through",
    "expected games; the risk score (chance of missing 4+ games) is shown for reference.",
    "Rookies are projected from draft pick only, and the 2027 class appears after the April draft.",
    "Bye weeks are blank until the 2027 schedule is released in May.",
]
if target == 2027:
    notes.insert(1, "PREVIEW: the 2026 season is still in progress. Rerun in January for the real board.")

out = DATA / "output" / f"draft_boards_{target}.xlsx"
wb = Workbook()
for i, (name, df) in enumerate([("Burbs Auction", burbs), ("CBS Snake", cbs)]):
    ws = wb.active if i == 0 else wb.create_sheet()
    ws.title = name
    ws.append(df.columns)
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F3864")
    for row in df.iter_rows():
        ws.append(list(row))
    ws.freeze_panes = "C2"
    for col in ws.columns:
        ws.column_dimensions[col[0].column_letter].width = max(9, min(26, max(len(str(c.value or "")) for c in col[:60]) + 2))
ws = wb.create_sheet("Notes")
for n in notes:
    ws.append([n])
ws.column_dimensions["A"].width = 110
wb.save(out)
print("wrote", out, burbs.height, cbs.height)
