"""Build both draft boards and the Draft Optimizer import file.

PYTHONPATH=src python3 scripts/build_boards.py <auction workbook.xlsx> [target, default 2027] [market.csv]
Writes to data/output/:
  draft_boards_<target>_burbs.xlsx      Overall, QB, RB, WR, TE, K, DEF, Notes
  draft_boards_<target>_cbs.xlsx        Overall, QB, RB, WR, TE, K, DEF, Notes
  draft_optimizer_import_<target>.xlsx  one sheet per Draft Optimizer position tab, same columns
"""
import sys

import polars as pl
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill

from ffmodel.board import _tiers, burbs_board, cbs_board
from ffmodel.config import DATA

wb_path = sys.argv[1]
target = int(sys.argv[2]) if len(sys.argv) > 2 else 2027
market_csv = sys.argv[3] if len(sys.argv) > 3 else None  # optional: Player,Price expected prices
r1 = lambda c: pl.col(c).round(1)
POS = ["QB", "RB", "WR", "TE", "K", "DEF"]
TAB = {"QB": "Quarterback", "RB": "Running Back", "WR": "Wide Receiver", "TE": "Tight End",
       "K": "Kicker", "DEF": "Defense"}
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


# per position: (players tiered, number of tier breaks); about 2 rounds' worth of draftable depth
TIER_SHAPE = {"QB": (20, 5), "RB": (45, 7), "WR": (60, 8), "TE": (20, 5), "K": (14, 3), "DEF": (14, 3)}


def name_expr():
    return (pl.when(pl.col("position") == "DEF")
            .then(pl.col("team").replace_strict(TEAMS, default=pl.col("player_display_name")))
            .otherwise(pl.col("player_display_name")))


def pos_tiers(df: pl.DataFrame) -> pl.DataFrame:
    """Tier within each position: a new tier starts at an unusually big drop in VORP."""
    parts = []
    for p in POS:
        s = df.filter(pl.col("position") == p).sort("vorp", descending=True)
        if s.height:
            parts.append(s.with_columns(pl.Series("pos_tier", _tiers(s["vorp"], *TIER_SHAPE[p]))))
    return pl.concat(parts)


note_col = pl.when(pl.col("rookie")).then(pl.lit("Rookie")).otherwise(pl.lit("")).alias("Note")

# ---- The Burbs (auction) ----
b = pos_tiers(burbs_board(target, wb_path, market_csv).with_columns(name_expr().alias("player_display_name")))
b = b.sort(["max_bid", "vorp"], descending=True)
burbs = b.select(
    pl.col("overall_rank").alias("Overall Rk"), pl.col("position").alias("Pos"), pl.col("pos_rank").alias("Pos Rk"),
    pl.col("pos_tier").alias("Pos Tier"), pl.col("player_display_name").alias("Player"), pl.col("team").alias("Team"),
    pl.col("bid").alias("Max Bid (Suggested)"), pl.col("max_bid").alias("Max Bid $"),
    pl.col("suggested_bid").alias("Suggested $"),
    pl.col("market_value").round(0).alias("League $"), pl.col("bargain").alias("Bargain $"),
    pl.col("flag").alias("Flag"),
    r1("pred_ppg_burbs").alias("PPG"), r1("exp_games").alias("Exp Games (of 13)"),
    (pl.col("pred_ppg_burbs") * pl.col("exp_games")).round(1).alias("Proj Pts (wk 1-14)"),
    r1("repl_ppg").alias("Repl PPG"), r1("vorp").alias("VORP"),
    pl.col("risk_score").alias("Injury Risk (0-100)"), note_col, pl.col("player_id").alias("nflverse ID"))

# ---- CBS (snake) ----
c = cbs_board(target).with_columns(name_expr().alias("player_display_name"))
c = pos_tiers(c).sort("vorp", descending=True)
cbs = c.select(
    pl.int_range(1, pl.len() + 1).alias("Overall Rk"), pl.col("tier").alias("Tier"),
    pl.col("position").alias("Pos"), pl.col("pos_rank").alias("Pos Rk"), pl.col("pos_tier").alias("Pos Tier"),
    pl.col("player_display_name").alias("Player"), pl.col("team").alias("Team"),
    r1("pred_ppg_cbs").alias("PPG"), r1("exp_games").alias("Exp Games (of 13)"),
    (pl.col("pred_ppg_cbs") * pl.col("exp_games")).round(1).alias("Proj Pts (wk 1-14)"),
    r1("vorp").alias("VORP"), pl.col("risk_score").alias("Injury Risk (0-100)"), note_col,
    pl.col("player_id").alias("nflverse ID"))

common_notes = [
    "Overall tab = every player ranked together. Position tabs = same rows, ranked within the position.",
    "Pos Tier: a new tier starts where value over replacement drops by an unusually large amount.",
    "VORP = expected games x (projected PPG - replacement PPG), plus an upside allowance for volatile players.",
    "Injury risk works only through expected games; the risk score (chance of missing 4+ games) is for reference.",
    "Rookies are projected from draft slot only; the 2027 class appears after the April draft.",
    "Bye weeks are blank until the 2027 schedule is released in May.",
]
preview = ["PREVIEW: the 2026 season is still in progress. Rerun in January for the real board."] if target == 2027 else []
burbs_notes = preview + [
    f"THE BURBS (auction, $200) draft board for {target}.",
    "Max Bid (Suggested): the first number is the most to pay; the number in brackets is where to aim.",
    "Max bid = the model's dollar value (value over replacement). Never bid past it.",
    "League $ = what this league usually pays for that position slot (2025-2026 auction average).",
    "Suggested = the league's usual price, capped at 90% of the max bid.",
    "Bargain $ = max bid - league price. 'Bargain' = at least $5 under model value; 'League overpays' = $5 over.",
    "League $ is a slot price by projected rank. Load real ADP / expert prices in August to make it player-specific.",
    "K and DEF are capped at $3: none sold above $4 in two years.",
] + common_notes
cbs_notes = preview + [
    f"CBS (snake) draft board for {target}, 6-pt passing TDs.",
    "Overall Rk is the draft order by VORP; Tier marks the big drops across the whole board.",
] + common_notes


def write_sheet(ws, df, freeze="C2"):
    ws.append(df.columns)
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F3864")
    for row in df.iter_rows():
        ws.append(list(row))
    ws.freeze_panes = freeze
    for col in ws.columns:
        ws.column_dimensions[col[0].column_letter].width = max(
            9, min(26, max(len(str(cc.value or "")) for cc in col[:60]) + 2))


def league_book(df, notes, path, rank_col="Pos Rk"):
    wb = Workbook()
    ws = wb.active
    ws.title = "Overall"
    write_sheet(ws, df, "F2")
    for p in POS:
        sub = df.filter(pl.col("Pos") == p).sort(rank_col).drop("Pos")
        sub = sub.select([rank_col] + [x for x in sub.columns if x != rank_col])
        write_sheet(wb.create_sheet(p), sub, "D2")
    ws = wb.create_sheet("Notes")
    for n in notes:
        ws.append([n])
    ws.column_dimensions["A"].width = 115
    wb.save(path)


out_dir = DATA / "output"
league_book(burbs, burbs_notes, out_dir / f"draft_boards_{target}_burbs.xlsx")
league_book(cbs, cbs_notes, out_dir / f"draft_boards_{target}_cbs.xlsx")

# ---- Draft Optimizer import: same columns as the workbook's position tabs ----
HEAD = [f"{target} Rank", "Player", "Tier", f"{target} Cap Value ($)", "My Assigned Bid ($)", "Value Gap ($)",
        "Proj Points", "Games", "Bye", "ADP", "Roto Proj Pts", "Blend Proj Pts", "Disagreement %", "Model $",
        "Model - My Bid ($)", "Team", "Clean Name", "Roto Auction $", "Your PPG", "Roto PPG", "Blend PPG",
        "FF Guys Auction $"]
FORMULA_COLS = set(range(10, 21)) - {16}  # K-P and R-U (0-based 10..20, except Q=16) are workbook formulas
imp = b.with_columns(
    (pl.col("exp_frac") * 17).alias("g17"),
).with_columns(
    pl.max_horizontal(pl.col("g17"), pl.lit(13.0)).round(1).alias("games"),  # workbook's 13-game floor (Risk Model B35)
    (pl.col("pred_ppg_burbs") * pl.col("g17")).round(2).alias("proj17"),
)
wb = Workbook()
ws = wb.active
ws.title = "How to import"
howto = [
    f"Draft Optimizer import for {target}: one sheet per position tab, same columns and order as the workbook.",
    "Paste columns A:I (Rank .. Bye) over the same cells in the matching tab, and column Q (Clean Name) over column Q.",
    "Leave columns K:P and R:U alone in the workbook; they are formulas (Rotoballer, Player Master, Risk Model).",
    "They are left empty here on purpose so a paste can't overwrite them if you paste A:I and Q separately.",
    "Cap Value ($) = the model's max bid for The Burbs. My Assigned Bid / Value Gap are yours to fill.",
    "Proj Points = model PPG x expected games on a 17-game basis (Risk Model B33/B34 = 17), before the floor.",
    "Games = expected games of 17 from the injury model, raised to the 13-game floor (Risk Model B35),",
    "  same convention as the workbook: Proj Points is not changed by the floor.",
    "Bye, ADP and FF Guys $ are blank: byes come with the May schedule, ADP and FF Guys prices in August.",
    "Clean Name = the plain player name used by the workbook's MATCH formulas. Defenses use the full team name.",
    "Row counts differ from the 2026 tabs: clear leftover rows below the pasted block, then re-check Player Master",
    "  and that 'Player Master'!T38 still totals $2,400 before using the Draft Optimizer.",
]
if preview:
    howto.insert(0, preview[0])
for n in howto:
    ws.append([n])
ws.column_dimensions["A"].width = 115
for p in POS:
    s = imp.filter(pl.col("position") == p).sort(["max_bid", "vorp"], descending=True)
    rows = []
    for i, r in enumerate(s.iter_rows(named=True), 1):
        row = [None] * len(HEAD)
        row[0], row[1], row[2] = i, r["player_display_name"], f"Tier {r['pos_tier']}"
        row[3] = float(r["max_bid"])
        row[6], row[7] = r["proj17"], r["games"]
        row[16] = r["player_display_name"]
        rows.append(row)
    sh = wb.create_sheet(TAB[p])
    sh.append(HEAD)
    for j, cell in enumerate(sh[1]):
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="7F7F7F" if j in FORMULA_COLS else "1F3864")
    for row in rows:
        sh.append(row)
    sh.freeze_panes = "C2"
    for col in sh.columns:
        sh.column_dimensions[col[0].column_letter].width = 14
    sh.column_dimensions["B"].width = sh.column_dimensions["Q"].width = 24
wb.save(out_dir / f"draft_optimizer_import_{target}.xlsx")
print("wrote", out_dir, burbs.height, cbs.height)
