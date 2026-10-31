"""The 2026 look-back: the finished model's pre-draft board for 2026 (data through 2025 only,
room prices from the 2025 auction only), compared with the 2026 auction and 2026 results.

Rerun in January once 2026 is complete.
PYTHONPATH=src python3 scripts/lookback_2026.py <auction workbook.xlsx>
"""
import re
import sys

import openpyxl
import polars as pl

from ffmodel.board import burbs_board
from ffmodel.config import DATA
from ffmodel.dst import dst_weeks

TARGET = 2026


def norm(name: str) -> str:
    n = name.split("\xa0")[0].lower()
    n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", n)
    return re.sub(r"[^a-z]", "", n)


board = burbs_board(TARGET, sys.argv[1])

wb = openpyxl.load_workbook(sys.argv[1], data_only=True, read_only=True)
sales = []
for r in wb["2026 Draft"].iter_rows(values_only=True):
    if not r[1]:
        continue
    name = str(r[1]).split("\xa0")[0]
    m = re.search(r"\(([A-Za-z]+)\s*-\s*([A-Z]+)\)", str(r[1]))
    key = norm(name)
    if m and m.group(2) == "DEF":
        key = "def:" + m.group(1).upper()
    sales.append({"key": key, "sold_name": name, "price": r[2], "manager": r[3]})
sales = pl.DataFrame(sales)

# actual 2026 so far: skill players + K from player seasons, DEF from team scoring
seas = pl.read_parquet(DATA / "processed" / "player_seasons.parquet").filter(pl.col("season") == TARGET)
act = seas.select("player_id", pl.col("games").alias("g26"), pl.col("ppg_burbs").alias("ppg26"))
d = (dst_weeks().filter(pl.col("season") == TARGET).group_by("team")
       .agg(pl.len().alias("g26"), pl.col("fpts_burbs").mean().alias("ppg26"))
       .rename({"team": "player_id"}))
act = pl.concat([act, d.select(act.columns)], how="vertical_relaxed")

TEAM_FIX = {"LA": "LAR", "LV": "LV", "JAX": "JAC", "WAS": "WAS"}
board = (board.with_columns(
            pl.when(pl.col("position") == "DEF")
              .then(pl.lit("def:") + pl.col("team").replace(TEAM_FIX))
              .otherwise(pl.col("player_display_name").map_elements(norm, return_dtype=pl.Utf8)).alias("key"))
              .join(sales, on="key", how="left").join(act, on="player_id", how="left")
              .with_columns(((pl.col("ppg26") - pl.col("repl_ppg")) * pl.col("g26")).alias("actual_vorp_sofar")))

cols = ["overall_rank", "pos_rank", "player_display_name", "position", "team", "rookie", "pred_ppg_burbs",
        "exp_games", "risk_score", "vorp", "auction_value", "market_value", "bid", "max_bid", "suggested_bid",
        "price", "manager", "g26", "ppg26", "actual_vorp_sofar"]
out = DATA / "output" / "lookback_2026_burbs.csv"
board.select(cols).write_csv(out)

sold = board.filter(pl.col("price").is_not_null(), pl.col("g26") >= 3)
weeks = seas["games"].max()
print(f"weeks played: {weeks}; sales matched: {board['price'].is_not_null().sum()} of {sales.height}")
sc = lambda df: df.select(
    pl.len().alias("n"),
    pl.corr("auction_value", "actual_vorp_sofar", method="spearman").round(3).alias("model"),
    pl.corr("price", "actual_vorp_sofar", method="spearman").round(3).alias("room"),
    pl.corr("max_bid", "actual_vorp_sofar", method="spearman").round(3).alias("max_bid"))
print(sc(sold))
print(sold.group_by("position").agg(
    pl.len().alias("n"),
    pl.corr("auction_value", "actual_vorp_sofar", method="spearman").round(3).alias("model"),
    pl.corr("price", "actual_vorp_sofar", method="spearman").round(3).alias("room"),
    pl.corr("max_bid", "actual_vorp_sofar", method="spearman").round(3).alias("max_bid")).sort("position"))
# discipline check: if you had only paid up to the max bid
won = sold.filter(pl.col("price") <= pl.col("max_bid"))
print("sales at or under our max bid:", won.height, "of", sold.height)
print("rookies sold:", board.filter(pl.col("rookie"), pl.col("price").is_not_null()).select(
    "player_display_name", "bid", "price", "ppg26"))
