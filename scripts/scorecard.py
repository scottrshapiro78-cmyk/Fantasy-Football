"""Weekly accuracy scorecard: how the week's projections did against what happened.

    PYTHONPATH=src python3 scripts/scorecard.py 2026 9

Reads the projections saved before the games (data/tracking/weekly_{season}_wk{week}_{league}.csv, or the
weekly workbook if there is no snapshot), joins actual fantasy points, and compares the model with three
simple alternatives a person could use instead: season PPG so far, last-3 PPG, and the model's own base
PPG (no matchup, Vegas or teammate adjustments). Also checks the play-chance figures against who actually
played, and lists the biggest misses. Appends one row per league/position to data/tracking/accuracy_log.csv
so the trend is visible week to week, and writes accuracy_{season}_{league}.xlsx.
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

from ffmodel.config import DATA
from ffmodel.dst import dst_weeks

TRACK = DATA / "tracking"
SHARED = Path("/mnt/project-files/ff-model/")
TOP = {"QB": 20, "RB": 40, "WR": 50, "TE": 18, "K": 14, "DEF": 14}  # fantasy-relevant: top N by projection
COLS = {"Pos": "pos", "Player": "player", "Team": "team", "Wk Proj": "proj", "Status": "status", "Play Chance": "play",
        "Base PPG": "base", "PPG So Far": "sofar", "Last 3 PPG": "last3", "Matchup Adj": "match", "Vegas Adj": "vegas",
        "Teammate Out Adj": "mate", "Injury News": "news", "nflverse ID": "player_id"}


def load_proj(season, week, lg):
    snap = TRACK / f"weekly_{season}_wk{week}_{lg}.csv"
    if snap.exists():
        return pl.read_csv(snap, infer_schema_length=2000)
    for d in (DATA / "output", SHARED):
        p = d / f"weekly_{season}_wk{week}_{lg}.xlsx"
        if p.exists():
            import openpyxl
            rows = list(openpyxl.load_workbook(p, read_only=True)["Overall"].iter_rows(values_only=True))
            df = pl.DataFrame([dict(zip(rows[0], r)) for r in rows[1:]], infer_schema_length=2000)
            return df.select([pl.col(k).alias(v) for k, v in COLS.items() if k in df.columns])
    return None


def actuals(season, week, lg):
    w = (pl.read_parquet(DATA / "processed" / "player_weeks.parquet")
           .filter(pl.col("season") == season, pl.col("week") == week)
           .select("player_id", pl.col(f"fpts_{lg}").alias("actual")))
    d = dst_weeks().filter(pl.col("season") == season, pl.col("week") == week).select(
        pl.col("team").alias("player_id"), pl.col(f"fpts_{lg}").alias("actual"))
    return pl.concat([w, d.with_columns(pl.col("actual").cast(w["actual"].dtype))])


def score(season, week, lg):
    p = load_proj(season, week, lg)
    if p is None:
        return None, None, None
    num = ["proj", "play", "base", "sofar", "last3", "match", "vegas", "mate"]
    p = p.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in num if c in p.columns])
    a = actuals(season, week, lg)
    if a.height < 200:
        return None, None, None  # week not played yet
    d = p.join(a, on="player_id", how="left").with_columns(pl.col("actual").is_not_null().alias("played"))
    d = d.with_columns(pl.col("proj").rank("ordinal", descending=True).over("pos").alias("rk"))
    rel = d.filter(pl.col("rk") <= pl.col("pos").replace_strict(TOP, return_dtype=pl.Int64))
    rows = []
    for pos in list(TOP) + ["ALL"]:
        r = rel if pos == "ALL" else rel.filter(pl.col("pos") == pos)
        g = r.filter(pl.col("played"))
        if g.height < 5:
            continue
        y = g["actual"].to_numpy()
        when_plays = g["proj"].to_numpy() / np.clip(g["play"].fill_null(100).to_numpy() / 100, 0.05, 1)  # projection if he plays
        row = {"season": season, "week": week, "league": lg, "pos": pos, "players": g.height,
               "mae_model": float(np.mean(np.abs(when_plays - y))),
               "mae_base": float(np.mean(np.abs(g["base"].fill_null(0).to_numpy() - y))),
               "mae_sofar": float(np.mean(np.abs(g["sofar"].fill_null(g["base"]).fill_null(0).to_numpy() - y))),
               "mae_last3": float(np.mean(np.abs(g["last3"].fill_null(g["sofar"]).fill_null(g["base"]).fill_null(0).to_numpy() - y))),
               "bias": float(np.mean(when_plays - y)),
               "rank_corr": float(pl.DataFrame({"a": when_plays, "b": y}).select(pl.corr("a", "b", method="spearman")).item()),
               "sat_out": r.height - g.height,
               "expected_sat_out": float((1 - r["play"].fill_null(100) / 100).sum())}
        rows.append(row)
    card = pl.DataFrame(rows)
    # play chance calibration: of players given X%, how many played
    cal = (rel.filter(pl.col("pos").is_in(["QB", "RB", "WR", "TE"]))
              .with_columns(pl.col("play").fill_null(100).alias("play"))
              .group_by("play").agg(pl.len().alias("players"), (pl.col("played").mean() * 100).round(0).alias("pct_played"))
              .sort("play", descending=True).with_columns(pl.lit(week).alias("week"), pl.lit(lg).alias("league")))
    miss = (rel.with_columns((pl.col("actual").fill_null(0) - pl.col("proj")).alias("miss"))
               .sort(pl.col("miss").abs(), descending=True).head(20)
               .select("pos", "player", "team", "proj", "actual", "miss", "play", "status", "news"))
    return card, cal, miss


if __name__ == "__main__":
    season, week = int(sys.argv[1]), int(sys.argv[2])
    TRACK.mkdir(exist_ok=True)
    log_path = TRACK / "accuracy_log.csv"
    for lg in ("burbs", "cbs"):
        card, cal, miss = score(season, week, lg)
        if card is None:
            print(f"{lg}: no saved projections or no results yet for week {week}")
            continue
        log = card
        if log_path.exists():
            old = pl.read_csv(log_path).filter(~((pl.col("season") == season) & (pl.col("week") == week) & (pl.col("league") == lg)))
            log = pl.concat([old, card], how="diagonal_relaxed")
        log.sort("season", "league", "week").write_csv(log_path)
        print(f"\n{lg.upper()} week {week}: avg miss in points per player (lower is better)")
        print(card.select("pos", "players", pl.col("mae_model", "mae_base", "mae_sofar", "mae_last3", "bias", "rank_corr").round(2),
                          "sat_out", pl.col("expected_sat_out").round(1)))
        import xlsxwriter
        path = DATA / "output" / f"accuracy_{season}_{lg}.xlsx"
        full = pl.read_csv(log_path).filter(pl.col("league") == lg, pl.col("season") == season)
        with xlsxwriter.Workbook(path) as wb:
            full.with_columns(pl.col(pl.Float64).round(2)).write_excel(wb, worksheet="By Week", autofit=True)
            cal.write_excel(wb, worksheet=f"Play Chance wk{week}", autofit=True)
            miss.with_columns(pl.col(pl.Float64).round(1)).write_excel(wb, worksheet=f"Biggest Misses wk{week}", autofit=True)
        print("wrote", path)
