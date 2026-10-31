"""The room's price curve: what The Burbs pays for the k-th best player at each position,
averaged over the 2025 and 2026 auctions (from the auction workbook)."""
from __future__ import annotations

import re

import openpyxl
import polars as pl

POS = ["QB", "RB", "WR", "TE", "K", "DEF"]


def _sales(path: str) -> pl.DataFrame:
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    rows = []
    for r in wb["2026 Draft"].iter_rows(values_only=True):
        if r[1] and isinstance(r[2], (int, float)):
            m = re.search(r"-\s*([A-Z]+)\)\s*$", str(r[1]))
            if m:
                rows.append({"year": 2026, "pos": m.group(1), "price": float(r[2])})
    for i, r in enumerate(wb["2025 Draft"].iter_rows(values_only=True)):
        if i and r[0] and r[0] != "--empty--" and r[1] in POS and isinstance(r[2], (int, float)):
            rows.append({"year": 2025, "pos": r[1], "price": float(r[2])})
    return pl.DataFrame(rows)


def slot_curve(path: str, years=(2025, 2026)) -> dict[str, list[float]]:
    """{pos: [price of #1, #2, ...]} averaged across years; slots past the deepest year cost $1."""
    s = _sales(path)
    out = {}
    for pos in POS:
        per_year = [sorted(s.filter(pl.col("pos") == pos, pl.col("year") == y)["price"].to_list(), reverse=True)
                    for y in years]
        n = max(len(p) for p in per_year) if per_year else 0
        out[pos] = [sum((p[k] if k < len(p) else 1.0) for p in per_year) / len(per_year) for k in range(n)]
    return out
