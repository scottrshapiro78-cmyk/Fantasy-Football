"""Player ID crosswalk: nflverse gsis_id is the key everywhere; Yahoo ID joins to the leagues."""
from __future__ import annotations

import polars as pl

from . import data


def crosswalk() -> pl.DataFrame:
    ids = data.player_ids()
    keep = [c for c in ["gsis_id", "yahoo_id", "sleeper_id", "pfr_id", "espn_id", "name",
                        "merge_name", "position", "team", "birthdate", "draft_year",
                        "draft_round", "draft_pick", "college"] if c in ids.columns]
    return (ids.select(keep)
               .filter(pl.col("gsis_id").is_not_null())
               .unique(subset="gsis_id", keep="first"))
