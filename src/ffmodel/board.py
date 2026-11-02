"""Assemble the draft boards for a target season: projections for every position (rookies,
kickers and defenses included), expected games, value over replacement, then

  The Burbs (auction):  max bid = the model's value (walk-away price; Scott's call 31 Oct:
                        the model is not fitted to the league's prices)
                        suggested bid = the league's usual price, capped at 90% of the max bid
                        league $ is shown beside it; bargain $ = model value - league price
  Note: until real 2027 market prices (ADP / expert $) are loaded, the room price is a slot
  price by projected position rank, not a player-specific price.
  CBS (snake):          overall rank by value over replacement, with tiers
"""
from __future__ import annotations

import numpy as np
import polars as pl

from . import availability as av
from . import dst, rookies
from .config import DATA, load_leagues
from .market import slot_curve
from .projection import Projector, project_kickers
from .value import value

CAL_POOL = {"RB": 30, "WR": 40, "TE": 12}  # QBs are not rescaled (see findings)


def _calibration(f: pl.DataFrame, target: int, league: str) -> dict:
    cal = []
    for t in range(target - 5, target):
        p = Projector(league).fit(f.filter(pl.col("season") + 1 < t)).predict(f.filter(pl.col("season") == t - 1))
        cal.append(p.join(f.filter(pl.col("season") == t - 1).select("player_id", f"next_ppg_{league}", "next_games"),
                          on="player_id").filter(pl.col("next_games") >= 6))
    cal = pl.concat(cal).to_pandas()
    out = {}
    for pos, n in CAL_POOL.items():
        q = cal[cal.position == pos].sort_values(f"pred_ppg_{league}", ascending=False).groupby("season").head(n)
        out[pos] = np.polyfit(q[f"pred_ppg_{league}"], q[f"next_ppg_{league}"], 1)
    return out


def suspensions(target: int) -> pl.DataFrame:
    """Games a player is known to miss at the start of the season (suspension / exempt list),
    within the fantasy regular season (weeks 1-14).

    From the roster: consecutive weeks from week 1 on Reserve/Suspended (code R40) or the exempt list;
    suspension lengths are announced before the season, so this is draft-night information.
    For a season not yet played, list announced suspensions in data/input/suspensions_<target>.csv
    (columns: player_id or Player, Games); that file also overrides the roster count.
    """
    out = pl.DataFrame(schema={"player_id": pl.Utf8, "susp_games": pl.Int64})
    path = DATA / "raw" / "rosters_weekly" / f"{target}.parquet"
    if path.exists():
        r = (pl.read_parquet(path).filter(pl.col("week") <= 14, pl.col("gsis_id").is_not_null())
               .select(pl.col("gsis_id").alias("player_id"), "week",
                       (pl.col("status").is_in(["SUS", "EXE"])
                        | (pl.col("status_description_abbr") == "R40")).alias("out")))
        r = r.group_by("player_id", "week").agg(pl.col("out").any()).sort("week")
        out = (r.group_by("player_id").agg(
                   pl.col("out").cast(pl.Int8).cum_prod().sum().alias("susp_games"))
                .filter(pl.col("susp_games") > 0).with_columns(pl.col("susp_games").cast(pl.Int64)))
    manual = DATA / "input" / f"suspensions_{target}.csv"
    if manual.exists():
        m = pl.read_csv(manual)
        if "player_id" not in m.columns:
            ids = pl.read_parquet(DATA / "processed" / "player_seasons.parquet").select(
                "player_id", pl.col("player_display_name").alias("Player")).unique("Player", keep="last")
            m = m.join(ids, on="Player", how="left")
        m = m.select("player_id", pl.col("Games").cast(pl.Int64).alias("susp_games"))
        out = pl.concat([out.filter(~pl.col("player_id").is_in(m["player_id"].to_list())), m])
    return out


def projections(target: int, league: str) -> pl.DataFrame:
    f = pl.read_parquet(DATA / "processed" / "features.parquet")
    rows = f.filter(pl.col("season") == target - 1)
    col = f"pred_ppg_{league}"
    proj = Projector(league).fit(f.filter(pl.col("season") + 1 < target)).predict(rows)
    cal = _calibration(f, target, league)
    proj = proj.with_columns(pl.struct("position", col).map_elements(
        lambda r: float(np.polyval(cal[r["position"]], r[col])) if r["position"] in cal else r[col],
        return_dtype=pl.Float64).alias(col))

    a = av.build()
    m = av.fit(a.filter(pl.col("season") + 1 < target))
    eg = pl.from_pandas(av.predict(m, a.filter(pl.col("season") == target - 1))[["player_id", "exp_frac", "risk_score"]])
    proj = proj.join(eg, on="player_id", how="left").with_columns(pl.lit(False).alias("rookie"))

    parts = [proj.select("player_id", "player_display_name", "position", "team", col, "exp_frac", "risk_score", "rookie")]
    rk = rookies.project(target)
    if rk.height:
        known = set(proj["player_id"].to_list())
        rk = rk.filter(~pl.col("player_id").is_in(known))
        parts.append(rk.select("player_id", "player_display_name", "position", "team", col, "exp_frac",
                               pl.lit(None, dtype=pl.Float64).alias("risk_score"), "rookie"))
    seasons = pl.read_parquet(DATA / "processed" / "player_seasons.parquet")
    k = project_kickers(seasons, target, league)
    parts.append(k.select("player_id", "player_display_name", "position", "team", col,
                          pl.lit(0.97).alias("exp_frac"), pl.lit(None, dtype=pl.Float64).alias("risk_score"),
                          pl.lit(False).alias("rookie")))
    d = dst.project(pl.read_parquet(DATA / "processed" / "dst_seasons.parquet"), target, league)
    parts.append(d.select(pl.col("team").alias("player_id"), (pl.col("team") + " DEF").alias("player_display_name"),
                          pl.lit("DEF").alias("position"), "team", col, pl.lit(1.0).alias("exp_frac"),
                          pl.lit(None, dtype=pl.Float64).alias("risk_score"), pl.lit(False).alias("rookie")))
    allp = pl.concat([p.with_columns(pl.col(col).cast(pl.Float64), pl.col("exp_frac").cast(pl.Float64))
                      for p in parts], how="vertical_relaxed")
    # known suspensions: those weeks are lost on top of normal injury risk (of 14 regular-season weeks)
    return (allp.join(suspensions(target), on="player_id", how="left")
                .with_columns(pl.col("susp_games").fill_null(0))
                .with_columns((pl.col("exp_frac") * (1 - pl.col("susp_games") / 14).clip(0, 1)).alias("exp_frac")))


def _tiers(vorp: pl.Series, top: int = 156, breaks: int = 11) -> list[int]:
    """Tiers from the largest drops in value. `vorp` must be sorted high to low.

    Among the top `top` players, the `breaks` largest gaps start a new tier; everyone
    below `top` shares one final tier (waiver-level depth).
    """
    v = vorp.to_list()
    n = min(top, len(v))
    gaps = [(v[i - 1] - v[i], i) for i in range(1, n)]
    cuts = {i for _, i in sorted(gaps, reverse=True)[:max(0, min(breaks, n - 1))]}
    tier, out = 1, []
    for i in range(len(v)):
        if i in cuts or (i == n and i < len(v)):
            tier += 1
        out.append(tier)
    return out


def _window_tiers(vorp: pl.Series, lo: int = 6, hi: int = 18, top: int = 156) -> list[int]:
    """Whole-board tiers: after each cut, the next cut is the biggest drop 6-18 picks later.

    Keeps tiers roughly round-sized while still breaking at natural drops; past `top`
    (13 rounds x 12 teams) everyone shares one final tier.
    """
    v = vorp.to_list()
    n = min(top, len(v))
    cuts, start = set(), 0
    while start + lo < n:
        window = range(start + lo, min(start + hi, n - 1) + 1)
        cut = max(window, key=lambda i: v[i - 1] - v[i])
        cuts.add(cut)
        start = cut
    tier, out = 1, []
    for i in range(len(v)):
        if i in cuts or (i == n and i < len(v)):
            tier += 1
        out.append(tier)
    return out


def _norm(name: str) -> str:
    import re
    n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", name.split("\xa0")[0].lower())
    return re.sub(r"[^a-z]", "", n)


def burbs_board(target: int, workbook: str, market_csv: str | None = None) -> pl.DataFrame:
    """Room prices only from auctions before the target season (no peeking in look-backs).

    market_csv: optional player-specific expected prices for the target season (columns
    Player, Price), e.g. Yahoo average auction values or the workbook's FF & Roto $ in August.
    Players it covers use that price instead of the slot price."""
    b, repl, depth = value(projections(target, "burbs"), "burbs", option=True)
    curve = slot_curve(workbook, years=tuple(y for y in (2025, 2026) if y < target))
    if market_csv:
        mk = pl.read_csv(market_csv).select(
            pl.col("Player").map_elements(_norm, return_dtype=pl.Utf8).alias("k"), pl.col("Price").cast(pl.Float64).alias("mkt"))
        b = b.with_columns(pl.col("player_display_name").map_elements(_norm, return_dtype=pl.Utf8).alias("k")).join(
            mk, on="k", how="left")
    b = b.with_columns(pl.struct("position", "pos_rank").map_elements(
        lambda r: (curve[r["position"]] + [1.0] * 500)[r["pos_rank"] - 1], return_dtype=pl.Float64).alias("market_value"))
    if market_csv:
        b = b.with_columns(pl.coalesce("mkt", "market_value").alias("market_value")).drop("k", "mkt")
    b = b.with_columns(pl.col("auction_value").clip(lower_bound=1).alias("max_bid"),
                       (pl.col("auction_value") - pl.col("market_value")).round(0).alias("bargain"))
    b = b.with_columns(pl.min_horizontal(pl.col("market_value"), 0.9 * pl.col("max_bid"))
                         .round(0).clip(lower_bound=1).alias("suggested_bid"),
                       pl.when(pl.col("bargain") >= 5).then(pl.lit("Bargain"))
                         .when(pl.col("bargain") <= -5).then(pl.lit("League overpays"))
                         .otherwise(pl.lit("")).alias("flag"))
    b = b.with_columns(
        pl.format("${} (${})", pl.col("max_bid").cast(pl.Int64), pl.col("suggested_bid").cast(pl.Int64)).alias("bid"))
    return b.sort("max_bid", "vorp", descending=True).with_columns(pl.int_range(1, pl.len() + 1).alias("overall_rank"))


def cbs_board(target: int) -> pl.DataFrame:
    b, repl, depth = value(projections(target, "cbs"), "cbs", option=True)
    b = b.sort("vorp", descending=True)
    return b.with_columns(pl.Series("tier", _window_tiers(b["vorp"])))
