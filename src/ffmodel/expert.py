"""FantasyPros expert consensus rankings (ECR), blended into the weekly and ROS rankings.

Source: dynastyprocess db_fpecr.parquet (free, updated weekly; FantasyPros PPR consensus, weekly snapshots).
ECR is a ranking, not points: within each position the expert order is given the model's own sorted numbers
(same scale, expert order), then blended: (1 - w) x model + w x expert.

Weights from scripts/study_expert_blend.py (Nov 19 2026), leave-one-season-out, adopted only where the blend
cut the average miss 2%+ (rules.md rule 5); a position not listed stays model-only:
- weekly (ROS-based weekly model, weeks 10-12, 2021-25): QB -5.4%, RB -2.9%, WR -3.5%; TE -0.1% (not used)
- ROS after week 10 (2020-25): QB -2.1%, RB -2.9%, TE -3.2%; WR -1.7% (not used)
If the feed can't be fetched or is stale, everything stays model-only and the build says so.
"""
from __future__ import annotations

import datetime as dt
import time

import polars as pl
import requests

from .config import DATA

URL = "https://raw.githubusercontent.com/dynastyprocess/data/master/files/db_fpecr.parquet"
CACHE = DATA / "external" / "db_fpecr.parquet"
WEEKLY_W = {"QB": 1.0, "RB": 0.9, "WR": 1.0}
ROS_W = {"QB": 1.0, "RB": 0.65, "TE": 0.9}
MAX_AGE_DAYS = 8  # a snapshot older than this is ignored


def _fetch() -> pl.DataFrame | None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not CACHE.exists() or time.time() - CACHE.stat().st_mtime > 12 * 3600:
        try:
            r = requests.get(URL, timeout=60)
            r.raise_for_status()
            CACHE.write_bytes(r.content)
        except Exception as e:  # keep an older copy if there is one
            print(f"expert feed fetch failed: {e}")
    if not CACHE.exists():
        return None
    return pl.read_parquet(CACHE)


def _ids() -> tuple[pl.DataFrame, pl.DataFrame]:
    ids = (pl.read_parquet(DATA / "processed" / "player_seasons.parquet").sort("season")
           .select("player_id", "yahoo_id", "merge_name", "position").unique(["player_id"], keep="last"))
    yid = ids.drop_nulls("yahoo_id").unique("yahoo_id").select("yahoo_id", pl.col("player_id").alias("pid_y"))
    nm = (ids.group_by("merge_name", "position").agg(pl.col("player_id").first().alias("pid_n"), pl.len().alias("k"))
          .filter(pl.col("k") == 1).drop("k"))
    return yid, nm


def latest(kind: str, as_of: dt.date | None = None, week_sunday: dt.date | None = None) -> tuple[pl.DataFrame | None, str]:
    """kind 'weekly' or 'ros'. Returns (player_id, position, ecr) for the newest snapshot, and a note.
    Weekly needs week_sunday: only a snapshot from the 6 days before that Sunday is for that week's games
    (FantasyPros snapshots land on Fridays, so a Tuesday build is model-only and a Saturday build is blended)."""
    raw = _fetch()
    if raw is None:
        return None, "expert feed unavailable: model only"
    as_of = as_of or dt.date.today()
    raw = raw.with_columns(pl.col("scrape_date").str.to_date()).filter(pl.col("pos").is_in(["QB", "RB", "WR", "TE"]),
                                                                       pl.col("scrape_date") <= as_of)
    if kind == "weekly":
        e = raw.filter(pl.col("page_type").str.starts_with("weekly-"), pl.col("ecr_type") == "wp")
    else:
        e = raw.filter(pl.col("fp_page").str.contains("/ros-"), pl.col("ecr_type") == "rp")
    if e.is_empty():
        return None, f"no {kind} expert snapshot: model only"
    d = e["scrape_date"].max()
    if kind == "weekly" and (week_sunday is None or not 0 <= (week_sunday - d).days <= 6):
        return None, f"no weekly expert snapshot for this week yet (newest {d}): model only"
    if (as_of - d).days > MAX_AGE_DAYS:
        return None, f"{kind} expert snapshot {d} is stale: model only"
    yid, nm = _ids()
    e = (e.filter(pl.col("scrape_date") == d).join(yid, on="yahoo_id", how="left")
         .with_columns(pl.col("mergename").str.to_lowercase().str.replace_all(r"[.'’]", "")
                       .str.replace_all(r"\s+(jr|sr|ii|iii|iv)$", "").alias("mn"))
         .join(nm, left_on=["mn", "pos"], right_on=["merge_name", "position"], how="left")
         .with_columns(pl.coalesce("pid_y", "pid_n").alias("player_id")).drop_nulls("player_id")
         .sort("ecr").unique("player_id", keep="first").select("player_id", pl.col("pos").alias("position"), "ecr"))
    return e, f"{kind} expert consensus {d} ({e.height} players)"


def blend(df: pl.DataFrame, col: str, weights: dict, ecr: pl.DataFrame | None, eligible: pl.Expr | None = None) -> pl.DataFrame:
    """Blend df[col] with the expert order. Adds 'expert_pts' (null where not blended); rewrites col."""
    if ecr is None:
        return df.with_columns(pl.lit(None, dtype=pl.Float64).alias("expert_pts"))
    ok = pl.col("position").is_in(list(weights)) & pl.col("ecr").is_not_null()
    if eligible is not None:
        ok = ok & eligible
    d = df.join(ecr.select("player_id", "ecr"), on="player_id", how="left").with_row_index("_i")
    pool = d.filter(ok)
    by_ecr = pool.sort("position", "ecr").with_columns(pl.int_range(pl.len()).over("position").alias("_r")).select("_i", "position", "_r")
    by_val = (pool.sort(["position", col], descending=[False, True]).with_columns(pl.int_range(pl.len()).over("position").alias("_r"))
              .select("position", "_r", pl.col(col).alias("expert_pts")))
    m = by_ecr.join(by_val, on=["position", "_r"]).select("_i", "expert_pts")
    d = d.join(m, on="_i", how="left")
    w = pl.col("position").replace_strict(weights, default=0.0, return_dtype=pl.Float64)
    d = d.with_columns(pl.when(pl.col("expert_pts").is_not_null())
                       .then((1 - w) * pl.col(col) + w * pl.col("expert_pts")).otherwise(pl.col(col)).alias(col))
    return d.drop("_i", "ecr")
