"""Fit replacement depth per position so Model $ best matches what The Burbs actually paid.

Coordinate search over the number of rostered players per position (replacement = the next
one), minimizing average $ error over matched sales.
  PYTHONPATH=src python3 scripts/fit_replacement.py <workbook> <years, e.g. 2025 or 2025,2026> [option]
"""
import sys

import polars as pl

from ffmodel.board import projections
from ffmodel.market import _norm, sales
from ffmodel.value import value

wb, years = sys.argv[1], [int(y) for y in sys.argv[2].split(",")]
option = len(sys.argv) > 3 and sys.argv[3] == "option"
data = []
for y in years:
    pr = projections(y, "burbs").with_columns(
        pl.col("player_display_name").map_elements(_norm, return_dtype=pl.Utf8).alias("key"))
    data.append((pr, sales(wb, y)))


def err(depth):
    tot, n = 0.0, 0
    for pr, sl in data:
        b, _, _ = value(pr, "burbs", depth_override=depth, option=option)
        j = b.join(sl, on="key").filter(pl.col("position") == pl.col("pos"))
        tot += (j["auction_value"] - j["price"]).abs().sum()
        n += j.height
    return tot / n


grid = {"QB": range(10, 31, 2), "RB": range(20, 66, 3), "WR": range(28, 76, 3), "TE": range(10, 31, 2)}
depth = {"QB": 18, "RB": 55, "WR": 65, "TE": 18}
best = err(depth)
print("start", depth, round(best, 3))
for _ in range(3):
    for pos, rng in grid.items():
        for v in rng:
            trial = {**depth, pos: v}
            e = err(trial)
            if e < best - 1e-9:
                best, depth = e, trial
    print("pass", depth, round(best, 3))
print("FITTED", depth, round(best, 3))
