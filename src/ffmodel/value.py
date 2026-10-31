"""Phase 5: replacement level, value over replacement, auction dollars and overall ranks.

Value per player = expected regular-season games x (projected PPG - replacement PPG).
Missed games are filled by a replacement-level player, so an injury only costs the gap
between the player and replacement (the injury-cost rule in the design doc).

Replacement PPG at a position = the PPG of the best player left after every team fills its
starters (flex included) and its share of bench spots: the player you could actually pick up.
"""
from __future__ import annotations

import polars as pl

from .config import load_leagues

POS = ["QB", "RB", "WR", "TE"]


def _depth(league: dict, vcfg: dict, pool: pl.DataFrame, ppg: str) -> dict[str, int]:
    """Rostered count per position: starters, flex filled by the best leftover RB/WR/TE, bench share."""
    teams, roster = league["teams"], league["roster"]
    n = {p: teams * roster.get(p, 0) for p in POS}
    flex = teams * roster.get("FLEX", 0)
    left = (pool.filter(pl.col("position").is_in(league["flex_eligible"]))
                .with_columns(pl.col(ppg).rank("ordinal", descending=True).over("position").alias("r"))
                .filter(pl.col("r") > pl.col("position").replace_strict(n, return_dtype=pl.Int64))
                .sort(ppg, descending=True).head(flex))
    for p, c in left.group_by("position").len().iter_rows():
        n[p] += c
    bench = teams * roster.get("BN", 0)
    for p in POS:
        n[p] += round(bench * vcfg["bench_split"][p])
    return n


def value(proj: pl.DataFrame, league_key: str) -> pl.DataFrame:
    """proj needs: player_id, player_display_name, position, pred_ppg_<league>, exp_frac."""
    cfg = load_leagues()
    league, vcfg = cfg[league_key], cfg["value"]
    ppg = f"pred_ppg_{league_key}"
    pool = proj.filter(pl.col("position").is_in(POS), pl.col(ppg).is_not_null())
    depth = _depth(league, vcfg, pool, ppg)
    repl = {p: (pool.filter(pl.col("position") == p).sort(ppg, descending=True)
                    .get_column(ppg).to_list() + [0.0] * 999)[depth[p]] for p in POS}
    g = vcfg["games_per_regular_season"]
    out = (pool.with_columns(
               pl.col("position").replace_strict(repl, return_dtype=pl.Float64).alias("repl_ppg"),
               (pl.col("exp_frac").fill_null(0.75) * g).alias("exp_games"))
               .with_columns(((pl.col(ppg) - pl.col("repl_ppg")) * pl.col("exp_games")).alias("vorp"))
               .with_columns(pl.col("vorp").rank("ordinal", descending=True).alias("overall_rank"),
                             pl.col("vorp").rank("ordinal", descending=True).over("position").alias("pos_rank")))
    if league.get("draft") == "auction":
        pos_vorp = out.filter(pl.col("vorp") > 0)["vorp"].sum()
        per = vcfg["burbs_discretionary_dollars"] / pos_vorp
        out = out.with_columns(pl.when(pl.col("vorp") > 0).then(1 + pl.col("vorp") * per)
                                 .otherwise(1.0).round(0).alias("auction_value"))
    return out.sort("overall_rank"), repl, depth
