import polars as pl
import pytest

from ffmodel import data
from ffmodel.config import league
from ffmodel.scoring import add_components, score


@pytest.fixture(scope="module")
def weeks():
    df = data.player_weeks([2023, 2024, 2025])
    return add_components(df.filter(pl.col("position").is_in(["QB", "RB", "WR", "TE"])))


def test_matches_nflverse_ppr(weeks):
    """Our scorer, set to nflverse's own rules, must reproduce its fantasy_points_ppr."""
    s = score(weeks, league("nflverse_ppr_check")["scoring"])
    diff = (s["fpts"] - s["fantasy_points_ppr"]).abs()
    assert (diff > 0.05).mean() < 0.001, s.filter(diff > 0.05).select(
        "player_display_name", "season", "week", "fpts", "fantasy_points_ppr").head(10)


def test_six_point_pass_td_adds_two_per_td(weeks):
    burbs = score(weeks, league("burbs")["scoring"], "b")
    cbs = score(burbs, league("cbs")["scoring"], "c")
    qb = cbs.filter(pl.col("position") == "QB")
    # CBS vs Burbs for a QB: +2 per pass TD, +1 per INT (−1 vs −2), minus Burbs return yards / PAT misses
    expected = 2 * qb["c_pass_td"] + qb["c_pass_int"] - 0.04 * qb["c_return_yd"]
    assert ((qb["c"] - qb["b"]) - expected).abs().max() < 1e-6
