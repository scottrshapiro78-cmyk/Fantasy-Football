"""Project 2027 PPG for both leagues from every completed season (2026 is partial until January).

Writes data/output/projections_2027.csv. Season totals, injury-adjusted games and replacement
value come in Phases 4-5; this is points per game only.
"""
import polars as pl

from ffmodel.config import DATA
from ffmodel.projection import Projector

f = pl.read_parquet(DATA / "processed" / "features.parquet")
rows = f.filter(pl.col("season") == 2026)
out = None
for lg in ["burbs", "cbs"]:
    p = Projector(lg).fit(f.filter(pl.col("season") <= 2024)).predict(rows)
    p = p.select("player_id", "player_display_name", "position", "team", f"pred_ppg_{lg}")
    out = p if out is None else out.join(p.select("player_id", f"pred_ppg_{lg}"), on="player_id")
out = (out.join(f.filter(pl.col("season") == 2026).select("player_id", "games", "ppg_burbs", "yahoo_id")
                if "yahoo_id" in f.columns else
                f.filter(pl.col("season") == 2026).select("player_id", "games", "ppg_burbs"), on="player_id")
          .with_columns(pl.col("pred_ppg_burbs").rank("ordinal", descending=True).over("position").alias("pos_rank_burbs"),
                        pl.col("pred_ppg_cbs").rank("ordinal", descending=True).over("position").alias("pos_rank_cbs"))
          .sort(["position", "pos_rank_burbs"]))
(DATA / "output").mkdir(parents=True, exist_ok=True)
out.write_csv(DATA / "output" / "projections_2027.csv")
pl.Config.set_tbl_rows(60)
for pos in ["QB", "RB", "WR", "TE"]:
    print(out.filter(pl.col("position") == pos).head(12).select(
        "player_display_name", "team", "games", "ppg_burbs", "pred_ppg_burbs", "pred_ppg_cbs").with_columns(pl.col(pl.Float64).round(1)))
