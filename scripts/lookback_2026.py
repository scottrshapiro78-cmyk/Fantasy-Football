"""The 2026 look-back: rankings the model would have produced before the 2026 draft, using only
data through the 2025 season, compared with the 2026 auction and with 2026 results so far.

Rerun in January once 2026 is complete.  PYTHONPATH=src python3 scripts/lookback_2026.py <workbook.xlsx>
"""
import re
import sys

import numpy as np
import openpyxl
import polars as pl

from ffmodel import availability as av
from ffmodel.config import DATA
from ffmodel.projection import Projector
from ffmodel.value import value

TARGET = 2026
f = pl.read_parquet(DATA / "processed" / "features.parquet")
rows = f.filter(pl.col("season") == TARGET - 1)


def norm(name: str) -> str:
    n = name.split("\xa0")[0].lower()
    n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", n)
    return re.sub(r"[^a-z]", "", n)


# 1. projection, calibrated on out-of-sample predictions from earlier seasons only
proj = Projector("burbs").fit(f.filter(pl.col("season") + 1 < TARGET)).predict(rows)
cal = []
for t in range(TARGET - 5, TARGET):
    p = Projector("burbs").fit(f.filter(pl.col("season") + 1 < t)).predict(f.filter(pl.col("season") == t - 1))
    cal.append(p.join(f.filter(pl.col("season") == t - 1).select("player_id", "next_ppg_burbs", "next_games"),
                      on="player_id").filter(pl.col("next_games") >= 6))
cal = pl.concat(cal).to_pandas()
coef = {}
for pos, n in {"QB": 12, "RB": 30, "WR": 40, "TE": 12}.items():
    q = cal[cal.position == pos].sort_values("pred_ppg_burbs", ascending=False).groupby("season").head(n)
    coef[pos] = np.polyfit(q.pred_ppg_burbs, q.next_ppg_burbs, 1)
proj = proj.with_columns(pl.struct("position", "pred_ppg_burbs").map_elements(
    lambda r: float(np.polyval(coef[r["position"]], r["pred_ppg_burbs"])) if r["position"] in coef
    else r["pred_ppg_burbs"], return_dtype=pl.Float64).alias("pred_ppg_burbs"))

# 2. expected games
avail_df = av.build()
m = av.fit(avail_df.filter(pl.col("season") + 1 < TARGET))
eg = pl.from_pandas(av.predict(m, avail_df.filter(pl.col("season") == TARGET - 1))[
    ["player_id", "exp_frac", "risk_score"]])
proj = proj.join(eg, on="player_id", how="left")

# 3. value
board, repl, depth = value(proj, "burbs")

# 4. the 2026 auction and 2026 results so far
wb = openpyxl.load_workbook(sys.argv[1], data_only=True, read_only=True)
sales = pl.DataFrame([{"key": norm(r[1]), "sold_name": r[1].split("\xa0")[0], "price": r[2], "manager": r[3]}
                      for r in wb["2026 Draft"].iter_rows(values_only=True) if r[1]])
seas = pl.read_parquet(DATA / "processed" / "player_seasons.parquet")
actual = seas.filter(pl.col("season") == TARGET).select("player_id", pl.col("games").alias("g26"),
                                                        pl.col("ppg_burbs").alias("ppg26"))
board = (board.with_columns(pl.col("player_display_name").map_elements(norm, return_dtype=pl.Utf8).alias("key"))
              .join(sales, on="key", how="left").join(actual, on="player_id", how="left"))
weeks_played = seas.filter(pl.col("season") == TARGET)["games"].max()
# actual value so far, against the same replacement levels
board = board.with_columns(((pl.col("ppg26") - pl.col("repl_ppg")) * pl.col("g26")).alias("actual_vorp_sofar"))

out = DATA / "output"
out.mkdir(parents=True, exist_ok=True)
cols = ["overall_rank", "pos_rank", "player_display_name", "position", "team", "pred_ppg_burbs", "exp_games",
        "risk_score", "vorp", "auction_value", "price", "manager", "g26", "ppg26", "actual_vorp_sofar"]
board.select(cols).write_csv(out / "lookback_2026_burbs.csv")

# 5. scorecard: who predicted 2026 better, the model or the room?
sold = board.filter(pl.col("price").is_not_null(), pl.col("g26") >= 3)
sc = (sold.group_by("position")
          .agg(pl.len().alias("n"),
               pl.corr("auction_value", "actual_vorp_sofar", method="spearman").round(3).alias("model_vs_actual"),
               pl.corr("price", "actual_vorp_sofar", method="spearman").round(3).alias("room_vs_actual"))
          .sort("position"))
allc = sold.select(pl.corr("auction_value", "actual_vorp_sofar", method="spearman").alias("model"),
                   pl.corr("price", "actual_vorp_sofar", method="spearman").alias("room"))
print(f"2026 weeks played so far: {weeks_played}; matched sales: {board['price'].is_not_null().sum()} of {sales.height}")
print("replacement PPG:", {k: round(v, 1) for k, v in repl.items()}, "depth:", depth)
print(sc); print(allc)
pl.Config.set_tbl_rows(40)
show = ["player_display_name", "position", "auction_value", "price", "ppg26", "g26"]
diff = sold.with_columns((pl.col("auction_value") - pl.col("price")).alias("gap"))
print("Model liked more than the room:"); print(diff.sort("gap", descending=True).select(show + ["gap"]).head(12))
print("Room liked more than the model:"); print(diff.sort("gap").select(show + ["gap"]).head(12))
print(board.select(cols[:10]).head(30))
