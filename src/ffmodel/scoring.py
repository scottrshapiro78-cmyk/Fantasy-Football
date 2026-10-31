"""Score nflverse weekly stat lines under any league's settings.

The model projects stat components, never fantasy points, so this module is the only
place league scoring rules live. Adding a league is a YAML change.
"""
from __future__ import annotations

import polars as pl


def _sum(*cols: str) -> pl.Expr:
    return pl.sum_horizontal([pl.col(c).fill_null(0) for c in cols])


# Stat components derived from nflverse player_stats columns.
COMPONENTS: dict[str, pl.Expr] = {
    "pass_yd": _sum("passing_yards"),
    "pass_td": _sum("passing_tds"),
    "pass_int": _sum("passing_interceptions"),
    "rush_yd": _sum("rushing_yards"),
    "rush_td": _sum("rushing_tds"),
    "rec": _sum("receptions"),
    "rec_yd": _sum("receiving_yards"),
    "rec_td": _sum("receiving_tds"),
    "return_yd": _sum("punt_return_yards", "kickoff_return_yards"),
    "return_td": _sum("special_teams_tds"),
    "two_pt": _sum("passing_2pt_conversions", "rushing_2pt_conversions", "receiving_2pt_conversions"),
    # Yahoo charges every lost fumble, including on kick and punt returns; nflverse's own
    # fantasy_points leave return fumbles out, so use the all-fumbles total where it exists.
    "fumble_lost": pl.max_horizontal(
        _sum("fumbles_lost_total"),
        _sum("sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost")),
    "off_fumble_ret_td": _sum("fumble_recovery_tds"),
    "fg_0_19": _sum("fg_made_0_19"),
    "fg_20_29": _sum("fg_made_20_29"),
    "fg_30_39": _sum("fg_made_30_39"),
    "fg_40_49": _sum("fg_made_40_49"),
    "fg_50_plus": _sum("fg_made_50_59", "fg_made_60_"),
    "pat_made": _sum("pat_made"),
    "pat_missed": _sum("pat_missed"),
}


def add_components(df: pl.DataFrame) -> pl.DataFrame:
    present = set(df.columns)
    exprs = []
    for name, expr in COMPONENTS.items():
        needed = expr.meta.root_names()
        if all(c in present for c in needed):
            exprs.append(expr.alias(f"c_{name}"))
        else:
            exprs.append(pl.lit(0.0).alias(f"c_{name}"))
    return df.with_columns(exprs)


def score(df: pl.DataFrame, scoring: dict, out: str = "fpts") -> pl.DataFrame:
    """Add a fantasy points column. df must already have c_* component columns."""
    terms = [pl.col(f"c_{k}") * float(v) for k, v in scoring.items() if f"c_{k}" in df.columns]
    return df.with_columns(pl.sum_horizontal(terms).round(4).alias(out))
