"""QB context for pass catchers and backs: who is throwing to him now, and how he has scored with that QB vs others.

Starter per team-game = the QB with the most pass attempts. "Current QB" = the starter of the team's latest game.
Text example: "Brissett: 1 g, 16.2 PPG | other QBs: 9 g, 6.4 PPG (small sample)".
Added Nov 21 after a reply wrongly said Brissett had started most of the season (he had started 1 game).
"""
import polars as pl

from ffmodel.config import DATA


def qb_context(season: int, thru: int, pts_col: str) -> pl.DataFrame:
    w = pl.read_parquet(DATA / "processed" / "player_weeks.parquet").filter(pl.col("season") == season, pl.col("week") <= thru)
    qb = (w.filter(pl.col("position") == "QB").sort("attempts", descending=True, nulls_last=True)
            .group_by("team", "week").agg(pl.col("player_display_name").first().alias("qb")))
    cur = qb.sort("week").group_by("team").agg(pl.col("qb").last().alias("cur_qb"))
    s = (w.filter(pl.col("position").is_in(["RB", "WR", "TE"])).join(qb, on=["team", "week"], how="left").join(cur, on="team", how="left")
           .with_columns((pl.col("qb") == pl.col("cur_qb")).alias("same")))
    g = s.group_by("player_id").agg(
        pl.col("cur_qb").last(),
        pl.col("same").sum().alias("n_cur"), pl.col(pts_col).filter(pl.col("same")).mean().alias("ppg_cur"),
        (~pl.col("same")).sum().alias("n_oth"), pl.col(pts_col).filter(~pl.col("same")).mean().alias("ppg_oth"))
    last = pl.col("cur_qb").str.split(" ").list.last()
    txt = pl.concat_str([last, pl.lit(": "), pl.col("n_cur").cast(pl.Utf8), pl.lit(" g, "),
                         pl.col("ppg_cur").round(1).cast(pl.Utf8), pl.lit(" PPG")])
    txt = (pl.when(pl.col("n_oth") > 0)
             .then(pl.concat_str([txt, pl.lit(" | other QBs: "), pl.col("n_oth").cast(pl.Utf8), pl.lit(" g, "),
                                  pl.col("ppg_oth").round(1).cast(pl.Utf8), pl.lit(" PPG"),
                                  pl.when(pl.col("n_cur") <= 2).then(pl.lit(" (new QB, small sample)")).otherwise(pl.lit(""))]))
             .otherwise(txt))
    return g.select(pl.col("player_id").alias("nflverse ID"), txt.alias("QB Context"))
