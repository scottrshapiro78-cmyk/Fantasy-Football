"""Phase 1: load every season 2012-2026, score it for each league, write player-game and
player-season tables to data/processed/.

Run:  PYTHONPATH=src python3 scripts/build_history.py
"""
import polars as pl

from ffmodel import data, ids
from ffmodel.config import DATA, load_leagues
from ffmodel.scoring import add_components, score

LEAGUES = ["burbs", "cbs"]
SKILL = ["QB", "RB", "WR", "TE", "K"]


def main() -> None:
    leagues = load_leagues()
    weeks = data.player_weeks(refresh_current=True)
    weeks = weeks.filter(pl.col("season_type") == "REG", pl.col("position").is_in(SKILL))
    weeks = add_components(weeks)
    for key in LEAGUES:
        weeks = score(weeks, leagues[key]["scoring"], f"fpts_{key}")

    out = DATA / "processed"
    out.mkdir(parents=True, exist_ok=True)
    keep = (["player_id", "player_display_name", "position", "season", "week", "team",
             "opponent_team", "game_id"] + [c for c in weeks.columns if c.startswith("c_")]
            + ["targets", "carries", "attempts", "target_share", "air_yards_share", "wopr"]
            + [f"fpts_{k}" for k in LEAGUES])
    weeks.select([c for c in keep if c in weeks.columns]).write_parquet(out / "player_weeks.parquet")

    agg = [pl.len().alias("games"), pl.col("team").last().alias("team")]
    agg += [pl.col(c).sum() for c in weeks.columns if c.startswith("c_")]
    agg += [pl.col(c).sum() for c in ["targets", "carries", "attempts"]]
    agg += [pl.col(f"fpts_{k}").sum().alias(f"fpts_{k}") for k in LEAGUES]
    seasons = (weeks.group_by(["player_id", "player_display_name", "position", "season"])
                    .agg(agg)
                    .with_columns([(pl.col(f"fpts_{k}") / pl.col("games")).round(2).alias(f"ppg_{k}")
                                   for k in LEAGUES]))
    xw = ids.crosswalk().rename({"position": "xw_position", "team": "xw_team", "name": "xw_name"})
    seasons = seasons.join(xw, left_on="player_id", right_on="gsis_id", how="left")
    seasons.write_parquet(out / "player_seasons.parquet")

    matched = seasons.filter(pl.col("season") >= 2022)
    rate = matched["yahoo_id"].is_not_null().mean()
    print(f"player-weeks: {weeks.height:,}  player-seasons: {seasons.height:,}")
    print(f"seasons {seasons['season'].min()}-{seasons['season'].max()}, "
          f"Yahoo ID match 2022+: {rate:.1%}")


if __name__ == "__main__":
    main()
