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


def _norm(name: str) -> str:
    n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", str(name).split("\xa0")[0].lower())
    return re.sub(r"[^a-z]", "", n)


def sales(path: str, year: int) -> pl.DataFrame:
    """Player-level sale prices for one auction: key (normalized name), pos, price."""
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    rows = []
    if year == 2026:
        for r in wb["2026 Draft"].iter_rows(values_only=True):
            if r[1] and isinstance(r[2], (int, float)):
                m = re.search(r"-\s*([A-Z]+)\)\s*$", str(r[1]))
                if m and m.group(1) in POS[:4]:
                    rows.append({"key": _norm(r[1]), "pos": m.group(1), "price": float(r[2])})
    else:
        for i, r in enumerate(wb["2025 Draft"].iter_rows(values_only=True)):
            if i and r[0] and r[0] != "--empty--" and r[1] in POS[:4] and isinstance(r[2], (int, float)):
                rows.append({"key": _norm(r[0]), "pos": r[1], "price": float(r[2])})
    return pl.DataFrame(rows)
