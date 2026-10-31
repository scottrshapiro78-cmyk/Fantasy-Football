"""Out-of-sample test of value variants on the 2026 auction: depth fitted on 2025 only."""
import sys

import polars as pl

from ffmodel.board import projections
from ffmodel.config import DATA
from ffmodel.market import _norm, sales
from ffmodel.value import value

wb = sys.argv[1]
fitted = {"QB": 10, "RB": 35, "WR": 55, "TE": 10}
pr = projections(2026, "burbs").with_columns(pl.col("player_display_name").map_elements(_norm, return_dtype=pl.Utf8).alias("key"))
sl = sales(wb, 2026)
act = pl.read_parquet(DATA / "processed" / "player_seasons.parquet").filter(pl.col("season") == 2026).select(
    "player_id", pl.col("games").alias("g26"), pl.col("ppg_burbs").alias("ppg26"))
_, repl0, _ = value(pr, "burbs")  # one fixed yardstick for actual value across variants
rows = []
for name, kw in [("current", {}), ("fitted depth", {"depth_override": fitted}),
                 ("option value", {"option": True}), ("fitted + option", {"depth_override": fitted, "option": True})]:
    b, repl, _ = value(pr, "burbs", **kw)
    j = (b.join(sl, on="key").filter(pl.col("position") == pl.col("pos")).join(act, on="player_id", how="left")
          .with_columns(((pl.col("ppg26") - pl.col("position").replace_strict(repl0, return_dtype=pl.Float64))
                          * pl.col("g26")).alias("av")))
    k = j.filter(pl.col("g26") >= 3)
    blend = k.with_columns((0.3 * pl.col("auction_value") + 0.7 * pl.col("price")).alias("bl"))
    rows.append({"variant": name, "n": j.height,
                 "price_error_$": round((j["auction_value"] - j["price"]).abs().mean(), 2),
                 "price_bias_$": round((j["auction_value"] - j["price"]).mean(), 2),
                 "rank_vs_actual": round(k.select(pl.corr("auction_value", "av", method="spearman")).item(), 3),
                 "blend_vs_actual": round(blend.select(pl.corr("bl", "av", method="spearman")).item(), 3),
                 "top10_model_$": round(b.filter(pl.col("position").is_in(["QB","RB","WR","TE"])).head(10)["auction_value"].mean(), 1)})
print(pl.DataFrame(rows))
