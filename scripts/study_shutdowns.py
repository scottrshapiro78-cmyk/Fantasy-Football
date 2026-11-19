"""Do NFL teams out of the playoff race sit injury-nagged stars late in the season?

For each season 2013-2025 (snap counts start 2013):
  * star = top fantasy scorer by PPR points per game in weeks 1-13 (min 6 games):
    QB top 12, RB top 24, WR top 30, TE top 10.
  * nagged = listed Questionable / Doubtful / Out on the official report in 2+ weeks of weeks 1-13.
  * must have played in one of his team's last two games before week 14 (so long-term IR cases
    are not counted as shutdowns).
  * outcome per team game in week 14 to the end of the regular season: played (offense snaps > 0)
    or missed, and snap share when he played vs. his weeks 1-13 average.
  * contention when the week starts, from standings at that point (approximation, no tiebreakers,
    no division-winner path):
      out      = can't reach the current last playoff seed's win total even by winning out
      longshot = 2+ games behind the last seed, not out
      alive    = within 2 games, not clinched
      clinched = more points than the first team outside the playoffs could reach

Run:  PYTHONPATH=src python3 scripts/study_shutdowns.py [current_week_for_2026]
"""
import sys

import nflreadpy as nfl
import polars as pl

from ffmodel import data

SEASONS = list(range(2013, 2026))
LATE = 14
TOP = {"QB": 12, "RB": 24, "WR": 30, "TE": 10}


def _k(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([pl.col(c).cast(pl.Int32) for c in ("season", "week") if c in df.columns])


def team_games(sched: pl.DataFrame) -> pl.DataFrame:
    s = sched.filter(pl.col("game_type") == "REG")
    home = s.select("season", "week", "game_id", pl.col("home_team").alias("team"),
                    pl.col("home_score").alias("pf"), pl.col("away_score").alias("pa"))
    away = s.select("season", "week", "game_id", pl.col("away_team").alias("team"),
                    pl.col("away_score").alias("pf"), pl.col("home_score").alias("pa"))
    return _k(pl.concat([home, away])).with_columns(
        pl.when(pl.col("pf") > pl.col("pa")).then(1.0)
          .when(pl.col("pf") == pl.col("pa")).then(0.5).otherwise(0.0).alias("w"))


def contention(tg: pl.DataFrame, conf: dict) -> pl.DataFrame:
    """Status of every team entering every week."""
    rows = []
    for season in tg["season"].unique().sort():
        g = tg.filter(pl.col("season") == season)
        total = g.group_by("team").len().rename({"len": "n"})
        seeds = 6 if season <= 2019 else 7
        for wk in sorted(g["week"].unique()):
            done = g.filter(pl.col("week") < wk, pl.col("pf").is_not_null())
            rec = (total.join(done.group_by("team").agg(pl.col("w").sum().alias("pts"), pl.len().alias("gp")),
                              on="team", how="left").fill_null(0)
                   .with_columns((pl.col("n") - pl.col("gp")).alias("rem"),
                                 pl.col("team").replace_strict(conf, default=None).alias("conf")))
            for c in ("AFC", "NFC"):
                r = rec.filter(pl.col("conf") == c).sort("pts", descending=True)
                if r.height < seeds + 1:
                    continue
                cut = r["pts"][seeds - 1]
                maxes = (r["pts"] + r["rem"]).to_list()
                for t, pts, rem in zip(r["team"], r["pts"], r["rem"]):
                    others = [m for tt, m in zip(r["team"], maxes) if tt != t]
                    # clinched (approx): fewer than `seeds` other teams can reach this team's current points
                    clinched = sum(m >= pts for m in others) < seeds
                    if pts + rem < cut:
                        st = "out"
                    elif cut - pts >= 2:
                        st = "longshot"
                    elif clinched:
                        st = "clinched"
                    else:
                        st = "alive"
                    rows.append((int(season), int(wk), t, st, float(cut - pts)))
    return _k(pl.DataFrame(rows, schema=["season", "week", "team", "status", "gb"], orient="row"))


def build(ss: list[int], late_from: int = LATE):
    sched = pl.read_parquet(data.DATA / "raw" / "sched.parquet") if (data.DATA / "raw" / "sched.parquet").exists() \
        else data.schedules(data.seasons())
    sched = sched.filter(pl.col("season").is_in(ss))
    teams = nfl.load_teams()
    conf = dict(zip(teams["team_abbr"], teams["team_conf"]))
    conf.update({"OAK": "AFC", "SD": "AFC", "STL": "NFC", "LA": "NFC", "LV": "AFC", "LAC": "AFC"})
    tg = team_games(sched)
    cont = contention(tg, conf)

    pw = _k(data.player_weeks(ss)).filter(pl.col("season_type") == "REG", pl.col("position").is_in(list(TOP)))
    early = pw.filter(pl.col("week") < late_from)
    ppg = (early.group_by("player_id", "player_display_name", "position", "season")
           .agg(pl.len().alias("g"), pl.col("fantasy_points_ppr").mean().alias("ppg"))
           .filter(pl.col("g") >= 6)
           .with_columns(pl.col("ppg").rank("ordinal", descending=True).over("season", "position").alias("rk")))
    stars = ppg.filter(pl.col("rk") <= pl.col("position").replace_strict(TOP, return_dtype=pl.Int64))

    inj = _k(data.injuries(ss)).filter(pl.col("game_type") == "REG", pl.col("week") < late_from,
                                   pl.col("report_status").is_in(["Questionable", "Doubtful", "Out"]))
    tags = inj.group_by("gsis_id", "season").agg(pl.col("week").n_unique().alias("tag_weeks"))
    stars = stars.join(tags, left_on=["player_id", "season"], right_on=["gsis_id", "season"], how="left") \
                 .with_columns(pl.col("tag_weeks").fill_null(0))

    ids = data.player_ids().select("gsis_id", "pfr_id").drop_nulls().unique("pfr_id")
    snaps = _k(data.snap_counts(ss)).filter(pl.col("game_type") == "REG") \
        .join(ids, left_on="pfr_player_id", right_on="pfr_id", how="inner") \
        .select("season", "week", "game_id", "team", "gsis_id", "offense_snaps", "offense_pct")
    snaps = snaps.filter(pl.col("gsis_id").is_in(stars["player_id"].implode()))

    # player's team = team of his last early-season snap row; must have played in team's last 2 early games
    last = (snaps.filter(pl.col("week") < late_from, pl.col("offense_snaps") > 0)
            .sort("week").group_by("gsis_id", "season")
            .agg(pl.col("team").last(), pl.col("week").last().alias("last_wk"),
                 pl.col("offense_pct").mean().alias("early_pct")))
    team_early = (tg.filter(pl.col("week") < late_from, pl.col("pf").is_not_null())
                  .sort("week").group_by("season", "team").agg(pl.col("week").tail(2).min().alias("cut_wk")))
    last = last.join(team_early, on=["season", "team"]).filter(pl.col("last_wk") >= pl.col("cut_wk"))
    stars = stars.join(last, left_on=["player_id", "season"], right_on=["gsis_id", "season"], how="inner")

    late_games = tg.filter(pl.col("week") >= late_from).select("season", "week", "game_id", "team", "pf")
    out = (stars.join(late_games, on=["season", "team"])
           .join(cont, on=["season", "week", "team"], how="left")
           .join(snaps.select("game_id", "gsis_id", "offense_snaps", "offense_pct"),
                 left_on=["game_id", "player_id"], right_on=["game_id", "gsis_id"], how="left")
           .with_columns((pl.col("offense_snaps").fill_null(0) == 0).alias("missed"),
                         (pl.col("tag_weeks") >= 2).alias("nagged"),
                         pl.col("week").max().over("season").alias("last_week")))
    return out, cont, stars


def summarize(out: pl.DataFrame) -> None:
    out = out.with_columns(
        pl.when(pl.col("status").is_in(["out", "longshot"])).then(pl.lit("out/longshot"))
          .otherwise(pl.col("status")).alias("grp"),
        pl.when(pl.col("week") == pl.col("last_week")).then(pl.lit("final wk")).otherwise(pl.lit("wk14-17")).alias("phase"))
    played = pl.col("offense_pct").filter(~pl.col("missed"))
    agg = [pl.col("player_id").n_unique().alias("players"), pl.len().alias("games"),
           pl.col("missed").mean().round(3).alias("miss_rate"),
           (played.mean() - pl.col("early_pct").filter(~pl.col("missed")).mean()).round(3).alias("snap_chg")]
    with pl.Config(tbl_rows=50, tbl_cols=20, tbl_width_chars=200):
        print("\n== by contention x nagged (all late games) ==")
        print(out.group_by("grp", "nagged").agg(agg).sort("grp", "nagged"))
        print("\n== split weeks 14-17 vs final week ==")
        print(out.group_by("phase", "grp", "nagged").agg(agg).sort("phase", "grp", "nagged"))
        print("\n== by position, out/longshot vs alive, nagged only, weeks before final ==")
        print(out.filter(pl.col("nagged"), pl.col("phase") == "wk14-17", pl.col("grp").is_in(["out/longshot", "alive"]))
              .group_by("position", "grp").agg(agg).sort("position", "grp"))
        print("\n== by era ==")
        print(out.with_columns(pl.when(pl.col("season") >= 2019).then(pl.lit("2019-25")).otherwise(pl.lit("2013-18")).alias("era"))
              .filter(pl.col("grp").is_in(["out/longshot", "alive"]))
              .group_by("era", "grp", "nagged").agg(agg).sort("era", "grp", "nagged"))
        # who got shut down: nagged stars on out/longshot teams who missed 2+ late games
        sd = (out.filter(pl.col("nagged"), pl.col("grp") == "out/longshot")
              .group_by("season", "player_display_name", "position", "team", "tag_weeks")
              .agg(pl.len().alias("late_g"), pl.col("missed").sum().alias("missed"))
              .filter(pl.col("missed") >= 2).sort("season", "missed", descending=[False, True]))
        print("\n== nagged stars on out/longshot teams who missed 2+ late games ==")
        print(sd)
        # player-level: missed 2+ late games (one row per player-season)
        ps = (out.filter(pl.col("grp").is_in(["out/longshot", "alive"]))
              .group_by("player_id", "season", "nagged")
              .agg(pl.col("grp").first(), pl.col("missed").sum().alias("m")))
        print("\n== player-seasons missing 2+ late games (team status at first late week) ==")
        print(ps.group_by("grp", "nagged").agg(pl.len().alias("n"), (pl.col("m") >= 2).mean().round(3).alias("share_2plus"))
              .sort("grp", "nagged"))


def main() -> None:
    out, cont, stars = build(SEASONS)
    out.write_parquet(data.DATA / "processed" / "shutdowns.parquet") if (data.DATA / "processed").exists() else None
    summarize(out)


if __name__ == "__main__":
    main()
